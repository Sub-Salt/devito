import os
import sys
import importlib
from glob import glob
from subprocess import call
from collections import OrderedDict

import numpy as np

from devito.compiler import make
from devito.exceptions import CompilationError
from devito.logger import debug, yask as log
from devito.tools import as_tuple

from devito.yask import cfac, nfac, ofac, namespace, exit, configuration
from devito.yask.utils import rawpointer


class YaskGrid(object):

    """
    A ``YaskGrid`` wraps a YASK grid.

    A ``YaskGrid`` implements a subset of the ``numpy.ndarray`` API. The subset of
    API implemented should suffice to transition between Devito backends w/o changes
    to the user code. It was not possible to subclass ``numpy.ndarray``, as this
    would have led to shadow data copies, since YASK employs a storage layout
    different than what users expect (row-major).

    The storage layout of a YASK grid looks as follows: ::

        --------------------------------------------------------------
        | extra_padding | halo |              | halo | extra_padding |
        ------------------------    domain    ------------------------
        |       padding        |              |       padding        |
        --------------------------------------------------------------
        |                         allocation                         |
        --------------------------------------------------------------

    :param grid: The YASK yk::grid that will be wrapped. Data storage will be
                 allocated if not yet allocated.
    :param shape: The "visibility region" of the YaskGrid. The shape should be
                  at least as big as the domain (in each dimension). If larger,
                  then users will be allowed to access more data entries,
                  such as those lying on the halo region.
    :param dimensions: A tuple of :class:`Dimension`s, representing the dimensions
                       of the ``YaskGrid``.
    :param radius: An integer indicating the extent of the halo region.
    :param dtype: A ``numpy.dtype`` for the raw data.
    """

    # Force __rOP__ methods (OP={add,mul,...) to get arrays, not scalars, for efficiency
    __array_priority__ = 1000

    def __init__(self, grid, shape, dimensions, radius, dtype):
        self.grid = grid
        self.dimensions = dimensions
        self.dtype = dtype

        self._modulo = tuple(i.modulo if i.is_Stepping else None for i in dimensions)

        # TODO: the initialization below will (slightly) change after the
        # domain-allocation switch will have happened in Devito. E.g.,
        # currently shape == domain

        if not grid.is_storage_allocated():
            # Set up halo sizes
            for i, j in zip(dimensions, shape):
                if i.is_Time:
                    assert self.grid.is_dim_used(i.name)
                    assert self.grid.get_alloc_size(i.name) == j
                else:
                    # Note, from the YASK docs:
                    # "If the halo is set to a value larger than the padding size,
                    # the padding size will be automatically increase to accomodate it."
                    self.grid.set_halo_size(i.name, radius)

            # Allocate memory
            self.grid.alloc_storage()

            self._halo = []
            self._ofs = [0 if i.is_Time else self.get_first_rank_domain_index(i.name)
                         for i in dimensions]
            self._shape = shape

            # `self` will actually act as a view of `self.base`
            self.base = YaskGrid(grid, shape, dimensions, 0, dtype)

            # Initialize memory to 0
            self.reset()
        else:
            self._halo = [0 if i.is_Time else self.get_halo_size(i.name)
                          for i in dimensions]
            self._ofs = [0 if i.is_Time else (self.get_first_rank_domain_index(i.name)-j)
                         for i, j in zip(dimensions, self._halo)]
            self._shape = [i + 2*j for i, j in zip(shape, self._halo)]

            # Like numpy.ndarray, `base = None` indicates that this is the real
            # array, i.e., it's not a view
            self.base = None

    def __getitem__(self, index):
        start, stop, shape = self._convert_index(index)
        if not shape:
            log("YaskGrid: Getting single entry %s" % str(start))
            assert start == stop
            out = self.grid.get_element(start)
        else:
            log("YaskGrid: Getting full-array/block via index [%s]" % str(index))
            out = np.empty(shape, self.dtype, 'C')
            self.grid.get_elements_in_slice(out.data, start, stop)
        return out

    def __setitem__(self, index, val):
        start, stop, shape = self._convert_index(index, 'set')
        if all(i == 1 for i in shape):
            log("YaskGrid: Setting single entry %s" % str(start))
            assert start == stop
            self.grid.set_element(val, start)
        elif isinstance(val, np.ndarray):
            log("YaskGrid: Setting full-array/block via index [%s]" % str(index))
            self.grid.set_elements_in_slice(val, start, stop)
        elif all(i == j-1 for i, j in zip(shape, self.shape)):
            log("YaskGrid: Setting full-array to given scalar via single grid sweep")
            self.grid.set_all_elements_same(val)
        else:
            log("YaskGrid: Setting block to given scalar via index [%s]" % str(index))
            self.grid.set_elements_in_slice_same(val, start, stop, True)

    def __getslice__(self, start, stop):
        if stop == sys.maxint:
            # Emulate default NumPy behaviour
            stop = None
        return self.__getitem__(slice(start, stop))

    def __setslice__(self, start, stop, val):
        if stop == sys.maxint:
            # Emulate default NumPy behaviour
            stop = None
        self.__setitem__(slice(start, stop), val)

    def _convert_index(self, index, mode='get'):
        """
        Convert an ``index`` into a format suitable for YASK's get_elements_{...}
        and set_elements_{...} routines.

        ``index`` can be of any type out of the types supported by NumPy's
        ``ndarray.__getitem__`` and ``ndarray.__setitem__``.

        In particular, an ``index`` is either a single element or an iterable of
        elements. An element can be a slice object, an integer index, or a tuple
        of integer indices.

        In the general case in which ``index`` is an iterable, each element in
        the iterable corresponds to a dimension in ``shape``. In this case, an element
        can be either a slice or an integer, but not a tuple of integers.

        If ``index`` is a single element,  then it is interpreted as follows: ::

            * slice object: the slice spans the whole shape;
            * single integer: shape is one-dimensional, and the index represents
              a specific entry;
            * a tuple of integers: it must be ``len(index) == len(shape)``,
              and each entry in ``index`` corresponds to a specific entry in a
              dimension in ``shape``.

        The returned value is a 3-tuple ``(starts, ends, shapes)``, where ``starts,
        ends, shapes`` are lists of length ``len(shape)``. By taking ``starts[i]`` and
        `` ends[i]``, one gets the start and end points of the section of elements to
        be accessed along dimension ``i``; ``shapes[i]`` gives the size of the section.
        """

        # Note: the '-1' below are because YASK uses '<=', rather than '<', to check
        # bounds when iterating over grid dimensions

        assert mode in ['get', 'set']
        index = as_tuple(index)

        # Index conversion
        cstart = []
        cstop = []
        cshape = []
        for i, size, mod in zip(index, self.shape, self._modulo):
            if isinstance(i, slice):
                if i.step is not None:
                    raise NotImplementedError("Unsupported stepping != 1.")
                if i.start is None:
                    start = 0
                elif i.start < 0:
                    start = size + i.start
                else:
                    start = i.start
                if i.stop is None:
                    stop = size - 1
                elif i.stop < 0:
                    stop = size + i.stop
                else:
                    stop = i.stop - 1
                shape = stop - start + 1
            else:
                if i is None:
                    start = 0
                    stop = size - 1
                elif i < 0:
                    start = size + i
                    stop = size + i
                else:
                    start = i
                    stop = i
                shape = 1 if mode == 'set' else None
            # Apply logical indexing
            if mod is not None:
                start %= mod
                stop %= mod
            # Finally append the converted index
            cstart.append(start)
            cstop.append(stop)
            if shape is not None:
                cshape.append(shape)

        # Remainder (e.g., requesting A[1] and A has shape (3,3))
        nremainder = len(self.shape) - len(index)
        cstart.extend([0]*nremainder)
        cstop.extend([self.shape[len(index) + j] - 1 for j in range(nremainder)])
        cshape.extend([self.shape[len(index) + j] for j in range(nremainder)])

        assert len(self.shape) == len(cstart) == len(cstop) == len(self._ofs)

        # Shift by the specified offsets
        cstart = [int(j + i) for i, j in zip(self._ofs, cstart)]
        cstop = [int(j + i) for i, j in zip(self._ofs, cstop)]

        return cstart, cstop, cshape

    def __getattr__(self, name):
        """Proxy to yk::grid methods."""
        return getattr(self.grid, name)

    def __repr__(self):
        return repr(self[:])

    def __meta_op__(op, reverse=False):
        # Used to build all binary operations such as __eq__, __add__, etc.
        # These all boil down to calling the numpy equivalents
        def f(self, other):
            o1, o2 = (self[:], other) if reverse is False else (other, self[:])
            return getattr(o1, op)(o2)
        return f
    __eq__ = __meta_op__('__eq__')
    __ne__ = __meta_op__('__ne__')
    __le__ = __meta_op__('__le__')
    __lt__ = __meta_op__('__lt__')
    __ge__ = __meta_op__('__ge__')
    __gt__ = __meta_op__('__gt__')
    __add__ = __meta_op__('__add__')
    __radd__ = __meta_op__('__add__')
    __sub__ = __meta_op__('__sub__')
    __rsub__ = __meta_op__('__sub__', True)
    __mul__ = __meta_op__('__mul__')
    __rmul__ = __meta_op__('__mul__', True)
    __div__ = __meta_op__('__div__')
    __rdiv__ = __meta_op__('__div__', True)
    __truediv__ = __meta_op__('__truediv__')
    __rtruediv__ = __meta_op__('__truediv__', True)
    __mod__ = __meta_op__('__mod__')
    __rmod__ = __meta_op__('__mod__', True)

    @property
    def with_halo(self):
        if self.base is None:
            raise ValueError("Cannot access the halo of a non-view Data")
        return self.base

    @property
    def name(self):
        return self.grid.get_name()

    @property
    def shape(self):
        return self._shape

    @property
    def rawpointer(self):
        return rawpointer(self.grid)

    def give_storage(self, target):
        """
        Share self's storage with ``target``.
        """
        for i in self.dimensions:
            if i.is_Time:
                target.set_alloc_size(i.name, self.get_alloc_size(i.name))
            else:
                target.set_halo_size(i.name, self.get_halo_size(i.name))
        target.share_storage(self.grid)

    def reset(self):
        """
        Set all grid entries to 0.
        """
        self[:] = 0.0

    def view(self):
        """
        View of the YASK grid in standard (i.e., Devito) row-major layout,
        returned as a :class:`numpy.ndarray`.
        """
        return self[:]


class YaskKernel(object):

    """
    A ``YaskKernel`` wraps a YASK kernel solution.
    """

    def __init__(self, name, yc_soln, local_grids=None):
        """
        Write out a YASK kernel, build it using YASK's Makefiles,
        import the corresponding SWIG-generated Python module, and finally
        create a YASK kernel solution object.

        :param name: Unique name of this YaskKernel.
        :param yc_soln: YaskCompiler solution.
        :param local_grids: A local grid is necessary to run the YaskKernel,
                            but its final content can be ditched. Indeed, local
                            grids are hidden to users -- for example, they could
                            represent temporary arrays introduced by the DSE.
                            This parameter tells which of the ``yc_soln``'s grids
                            are local.
        """
        self.name = name

        # Shared object name
        self.soname = "%s.%s.%s" % (name, yc_soln.get_name(), configuration['platform'])

        # It's necessary to `clean` the YASK kernel directory *before*
        # writing out the first `yask_stencil_code.hpp`
        make(namespace['path'], ['-C', namespace['kernel-path'], 'clean'])

        # Write out the stencil file
        if not os.path.exists(namespace['kernel-path-gen']):
            os.makedirs(namespace['kernel-path-gen'])
        yc_soln.format(configuration['isa'],
                       ofac.new_file_output(namespace['kernel-output']))

        # JIT-compile it
        try:
            compiler = configuration.yask['compiler']
            opt_level = 1 if configuration.yask['develop-mode'] else 3
            make(namespace['path'], ['-j3', 'YK_CXX=%s' % compiler.cc,
                                     'YK_CXXOPT=-O%d' % opt_level,
                                     'mpi=0',  # Disable MPI for now
                                     # "EXTRA_MACROS=TRACE",
                                     'YK_BASE=%s' % str(name),
                                     'stencil=%s' % yc_soln.get_name(),
                                     'arch=%s' % configuration['platform'],
                                     '-C', namespace['kernel-path'], 'api'])
        except CompilationError:
            exit("Kernel solution compilation")

        # Import the corresponding Python (SWIG-generated) module
        try:
            yk = getattr(__import__('yask', fromlist=[name]), name)
        except ImportError:
            exit("Python YASK kernel bindings")
        try:
            yk = reload(yk)
        except NameError:
            # Python 3.5 compatibility
            yk = importlib.reload(yk)

        # Create the YASK solution object
        kfac = yk.yk_factory()
        self.env = kfac.new_env()
        self.soln = kfac.new_solution(self.env)

        # MPI setup: simple rank configuration in 1st dim only.
        # TODO: in production runs, the ranks would be distributed along all
        # domain dimensions.
        self.soln.set_num_ranks(self.space_dimensions[0], self.env.get_num_ranks())

        # Redirect stdout/strerr to a string or file
        if configuration.yask['dump']:
            filename = 'yk_dump.%s.%s.%s.txt' % (self.name,
                                                 configuration['platform'],
                                                 configuration['isa'])
            filename = os.path.join(configuration.yask['dump'], filename)
            self.output = yk.yask_output_factory().new_file_output(filename)
        else:
            self.output = yk.yask_output_factory().new_string_output()
        self.soln.set_debug_output(self.output)

        # Users may want to run the same Operator (same domain etc.) with
        # different grids.
        self.grids = {i.get_name(): i for i in self.soln.get_grids()}
        self.local_grids = {i.name: self.grids[i.name] for i in (local_grids or [])}

    def new_grid(self, name, obj):
        """
        Create a new YASK grid.
        """
        return self.soln.new_fixed_size_grid(name, [str(i) for i in obj.indices],
                                             [int(i) for i in obj.shape])  # cast np.int

    def run(self, cfunction, arguments, toshare):
        """
        Run the YaskKernel through a JIT-compiled function.

        :param cfunction: The JIT-compiler function, of type :class:`ctypes.FuncPtr`
        :param arguments: Mapper from function/dimension/... names to run-time values
               to be passed to ``cfunction``.
        :param toshare: Mapper from functions to :class:`YaskGrid`s for sharing
                        grid storage.
        """
        # Sanity check
        grids = {i.grid for i in toshare if i.is_TensorFunction}
        assert len(grids) == 1
        grid = grids.pop()

        # Set the domain size, apply grid sharing, more sanity checks
        for k, v in zip(self.space_dimensions, grid.shape):
            self.soln.set_rank_domain_size(k, int(v))
        for k, v in toshare.items():
            target = self.grids.get(k.name)
            if target is not None:
                v.give_storage(target)
        assert all(not i.is_storage_allocated() for i in self.local_grids.values())
        assert all(v.is_storage_allocated() for k, v in self.grids.items()
                   if k not in self.local_grids)

        # Debug info
        debug("%s<%s,%s>" % (self.name, self.time_dimension, self.space_dimensions))
        for i in list(self.grids.values()) + list(self.local_grids.values()):
            if i.get_num_dims() == 0:
                debug("    Scalar: %s", i.get_name())
            elif not i.is_storage_allocated():
                size = [i.get_rank_domain_size(j) for j in self.space_dimensions]
                debug("    LocalGrid: %s%s, size=%s" %
                      (i.get_name(), str(i.get_dim_names()), size))
            else:
                size = [i.get_rank_domain_size(j) for j in self.space_dimensions]
                pad = [i.get_pad_size(j) for j in self.space_dimensions]
                debug("    Grid: %s%s, size=%s, pad=%s" %
                      (i.get_name(), str(i.get_dim_names()), size, pad))

        # Apply any user-provided option, if any
        self.soln.apply_command_line_options(configuration.yask['options'] or '')
        # Set up the block shape for loop blocking
        for i, j in zip(self.space_dimensions, configuration.yask['blockshape']):
            self.soln.set_block_size(i, j)

        # This, amongst other things, allocates storage for the temporary grids
        self.soln.prepare_solution()

        # Set up auto-tuning
        if configuration.yask['autotuning'] == 'off':
            self.soln.reset_auto_tuner(False)
        elif configuration.yask['autotuning'] == 'preemptive':
            self.soln.run_auto_tuner_now()

        # Run the kernel
        cfunction(*list(arguments.values()))

        # Release grid storage. Note: this *will not* cause deallocation, as these
        # grids are actually shared with the hook solution
        for i in self.grids.values():
            i.release_storage()
        # Release local grid storage. This *will* cause deallocation
        for i in self.local_grids.values():
            i.release_storage()
        # Dump performance data
        self.soln.get_stats()

    @property
    def space_dimensions(self):
        return tuple(self.soln.get_domain_dim_names())

    @property
    def time_dimension(self):
        return self.soln.get_step_dim_name()

    @property
    def rawpointer(self):
        return rawpointer(self.soln)

    def __repr__(self):
        return "YaskKernel [%s]" % self.name


class YaskContext(object):

    def __init__(self, name, grid, dtype):
        """
        Proxy between Devito and YASK.

        A YaskContext contains N YaskKernel and M YaskGrids, which have space
        and time dimensions in common.

        :param name: Unique name of the context.
        :param grid: A :class:`Grid` carrying the context dimensions.
        :param dtype: The data type used in kernels, as a NumPy dtype.
        """
        self.name = name
        self.space_dimensions = grid.dimensions
        self.time_dimension = grid.stepping_dim
        self.dtype = dtype

        # All known solutions and grids in this context
        self.solutions = []
        self.grids = {}

        # Build the hook kernel solution (wrapper) to create grids
        yc_hook = self.make_yc_solution(namespace['jit-yc-hook'])
        # Need to add dummy grids to make YASK happy
        # TODO: improve me
        handle = [nfac.new_domain_index(str(i)) for i in self.space_dimensions]
        yc_hook.new_grid('dummy_wo_time', handle)
        handle = [nfac.new_step_index(str(self.time_dimension))] + handle
        yc_hook.new_grid('dummy_w_time', handle)
        self.yk_hook = YaskKernel(namespace['jit-yk-hook'](name, 0), yc_hook)

    @property
    def dimensions(self):
        return (self.time_dimension,) + self.space_dimensions

    @property
    def nsolutions(self):
        return len(self.solutions)

    @property
    def ngrids(self):
        return len(self.grids)

    def make_grid(self, obj):
        """
        Create and return a new :class:`YaskGrid`, a YASK grid wrapper. Memory
        is allocated.

        :param obj: The symbolic data object for which a YASK grid is allocated.
        """
        if set(obj.indices) < set(self.space_dimensions):
            exit("Need a Function[x,y,z] to create a YASK grid.")
        name = 'devito_%s_%d' % (obj.name, contexts.ngrids)
        log("Allocating YaskGrid for %s (%s)" % (obj.name, str(obj.shape)))
        grid = self.yk_hook.new_grid(name, obj)
        wrapper = YaskGrid(grid, obj.shape, obj.indices, obj.space_order, obj.dtype)
        self.grids[name] = wrapper
        return wrapper

    def make_yc_solution(self, namer):
        """
        Create and return a YASK compiler solution object.
        """
        name = namer(self.name, self.nsolutions)

        yc_soln = cfac.new_solution(name)

        # Redirect stdout/strerr to a string or file
        if configuration.yask['dump']:
            filename = 'yc_dump.%s.%s.%s.txt' % (name, configuration['platform'],
                                                 configuration['isa'])
            filename = os.path.join(configuration.yask['dump'], filename)
            yc_soln.set_debug_output(ofac.new_file_output(filename))
        else:
            yc_soln.set_debug_output(ofac.new_null_output())

        # Set data type size
        yc_soln.set_element_bytes(self.dtype().itemsize)

        # Apply compile-time optimizations
        if configuration['isa'] != 'cpp':
            dimensions = [nfac.new_domain_index(str(i)) for i in self.space_dimensions]
            # Vector folding
            for i, j in zip(dimensions, configuration.yask['folding']):
                yc_soln.set_fold_len(i, j)
            # Unrolling
            for i, j in zip(dimensions, configuration.yask['clustering']):
                yc_soln.set_cluster_mult(i, j)

        return yc_soln

    def make_yk_solution(self, namer, yc_soln, local_grids):
        """
        Create and return a new :class:`YaskKernel` using ``self`` as context
        and ``yc_soln`` as YASK compiler ("stencil") solution.
        """
        soln = YaskKernel(namer(self.name, self.nsolutions), yc_soln, local_grids)
        self.solutions.append(soln)
        return soln

    def __repr__(self):
        return ("YaskContext: %s\n"
                "- domain: %s\n"
                "- grids: [%s]\n"
                "- solns: [%s]\n") % (self.name, str(self.space_dimensions),
                                      ', '.join([i for i in list(self.grids)]),
                                      ', '.join([i.name for i in self.solutions]))


class ContextManager(OrderedDict):

    def __init__(self, *args, **kwargs):
        super(ContextManager, self).__init__(*args, **kwargs)
        self.ncontexts = 0

    def dump(self):
        """
        Drop all known contexts and clean up the relevant YASK directories.
        """
        self.clear()
        call(['rm', '-f'] + glob(os.path.join(namespace['path'], 'yask', '*devito*')))
        call(['rm', '-f'] + glob(os.path.join(namespace['path'], 'lib', '*devito*')))
        call(['rm', '-f'] + glob(os.path.join(namespace['path'], 'lib', '*hook*')))

    def fetch(self, grid, dtype):
        """
        Fetch the :class:`YaskContext` in ``self`` uniquely identified by
        ``grid`` and ``dtype``. Create a new (empty) :class:`YaskContext` on miss.
        """
        # A unique key for this context.
        key = (configuration['isa'], dtype, grid.dimensions,
               grid.time_dim, grid.stepping_dim)

        # Fetch or create a YaskContext
        if key in self:
            log("Fetched existing context from cache")
        else:
            self[key] = YaskContext('ctx%d' % self.ncontexts, grid, dtype)
            self.ncontexts += 1
            log("Context successfully created!")
        return self[key]

    @property
    def ngrids(self):
        return sum(i.ngrids for i in self.values())


contexts = ContextManager()
"""All known YASK contexts."""


# Helpers

class YaskGridConst(np.float64):

    """A YASK grid wrapper for scalar values."""

    def give_storage(self, target):
        if not target.is_storage_allocated():
            target.alloc_storage()
        target.set_element(float(self.real), [])

    @property
    def rawpointer(self):
        return None


class YaskNullKernel(object):

    """Used when an Operator doesn't actually have a YASK-offloadable tree."""

    def __init__(self):
        self.name = 'null solution'
        self.grids = {}
        self.local_grids = {}

    def run(self, cfunction, arguments, toshare):
        cfunction(*list(arguments.values()))


class YaskNullContext(object):

    """Used when an Operator doesn't actually have a YASK-offloadable tree."""

    @property
    def space_dimensions(self):
        return '?'

    @property
    def time_dimension(self):
        return '?'
