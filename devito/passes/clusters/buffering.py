from collections import OrderedDict, defaultdict, namedtuple
from itertools import combinations

from cached_property import cached_property

from devito.ir import (Cluster, Forward, Interval, IntervalGroup, IterationSpace,
                       DataSpace, Queue, Vector, lower_exprs, detect_accesses,
                       build_intervals, vmax, vmin, PARALLEL)
from devito.exceptions import InvalidOperator
from devito.logger import warning
from devito.symbolics import retrieve_function_carriers, uxreplace
from devito.tools import (Bunch, DefaultOrderedDict, as_tuple, filter_ordered, flatten,
                          is_integer, timed_pass)
from devito.types import Array, CustomDimension, Dimension, Eq, ModuloDimension

__all__ = ['buffering']


@timed_pass()
def buffering(clusters, callback, sregistry, options):
    """
    Replace written Functions with Arrays. This gives the compiler more control
    over storage layout, data movement (e.g. between host and device), etc.

    Parameters
    ----------
    cluster : list of Cluster
        Input Clusters, subject of the optimization pass.
    callback : callable, optional
        A mechanism to express what the buffering candidates are, and what
        Dimensions, if any, should be replaced by ModuloDimensions, such that
        the buffer has a smaller footprint than that of the Function it stems
        from. The callable takes a Function as input and returns either None
        or a list. If the output is None, then the input is not a buffering
        candidate. Otherwise, the output is a buffering candidate and the list
        contains the Dimensions to be replaced by new ModuloDimensions. If
        unspecified, by default all DiscreteFunctions are turned into buffers,
        but no Dimension replacement occurs.
    sregistry : SymbolRegistry
        The symbol registry, to create unique names for buffers and Dimensions.
    options : dict
        The optimization options.
        Accepted: ['buf-async-degree'].
        * 'buf-async-degree': Specify the size of the buffer. By default, the
          buffer size is the minimal one, inferred from the memory accesses in
          the ``clusters`` themselves. An asynchronous degree equals to `k`
          means that the buffer will be enforced to size=`k` along the introduced
          ModuloDimensions. This might help relieving the synchronization
          overhead when asynchronous operations are used (these are however
          implemented by other passes).

    Examples
    --------
    Assume `callback(f)=[time]`, meaning that `f`, a Function defined over `time`,
    should be replaced by a buffer using a ModuloDimension in place of `time`.
    Consider the Eq below:

        Eq(u[time+1, x], u[time, x] + u[time-1, x] + 1)

    We see that `u(time, x)` is both read and written. Applying buffering boils
    down to:

        1. Prepending an Eq to initialize the buffer `ub`, i.e.

            Eq(ub[d, x], u[d, x])

        2. Appending an Eq to copy `ub` back into `u`

            Eq(u[time+1, x], ub[d+1, x])

        3. Replacing all other occurrences of `u` with `ub`

    So eventually we have three Eqs:

        Eq(ub[d, x], u[d, x]),
        Eq(ub[d+1, x], ub[d, x] + ub[d-1, x] + 1)
        Eq(u[time+1, x], ub[d+1, x])
    """
    if callback is None:
        def callback(f):
            if f.is_DiscreteFunction:
                return []
            else:
                return None
    assert callable(callback)

    return Buffering(callback, sregistry, options).process(clusters)


class Buffering(Queue):

    def __init__(self, callback0, sregistry, options):
        super().__init__()

        self.callback0 = callback0
        self.sregistry = sregistry
        self.options = options

    def process(self, clusters):
        return self._process_fatd(clusters, 1, cache={})

    def callback(self, clusters, prefix, cache=None):
        async_degree = self.options['buf-async-degree']

        # Locate all Function accesses within the provided `clusters`
        accessmap = AccessMapper(clusters)

        # Create the buffers
        buffers = []
        for f, accessv in accessmap.items():
            # Has a buffer already been produced for `f`?
            if f in cache:
                continue

            # Is `f` really a buffering candidate?
            dims = self.callback0(f)
            if dims is None:
                continue
            if not all(any([i.dim in d._defines for i in prefix]) for d in dims):
                continue

            b = cache[f] = Buffer(f, dims, accessv, async_degree, self.sregistry)
            buffers.append(b)

        if not buffers:
            return clusters

        # Create Eqs to initialize buffers. Note: a buffer needs to be initialized
        # only if the buffered Function is read in at least one place or in the case
        # of non-uniform SubDimensions, to avoid uninitialized values to be copied-back
        # into the buffered Function
        processed = []
        for b in buffers:
            if b.is_read or not b.has_uniform_subdims:
                lhs = b.indexed[b.buffer.dimensions]
                rhs = b.function[[b.initmap.get(d, d) for d in b.function.dimensions]]

                expr = lower_exprs(Eq(lhs, rhs))
                ispace = b.writeto
                dspace = derive_dspace(expr, ispace)
                properties = {d: {PARALLEL} for d in ispace.itdimensions}

                processed.append(Cluster(expr, ispace, dspace, properties=properties))

        # Substitution rules to replace buffered Functions with buffers
        subs = {}
        for b in buffers:
            for a in b.accessv.accesses:
                subs[a] = b.indexed[[b.index_mapper_flat.get(i, i) for i in a.indices]]

        for c in clusters:
            # If a buffer is read but never written, then we need to add
            # an Eq to step through the next slot
            # E.g., `ub[0, x] = u[time+2, x]`
            for b in buffers:
                if not b.is_readonly:
                    continue
                try:
                    c.exprs.index(b.lastread)
                except ValueError:
                    continue

                dims = b.function.dimensions
                lhs = b.indexed[[b.lastmap.get(d, Map(d, d)).b for d in dims]]
                rhs = b.function[[b.lastmap.get(d, Map(d, d)).f for d in dims]]

                expr = lower_exprs(uxreplace(Eq(lhs, rhs), b.subdims_mapper))
                ispace = b.written
                dspace = derive_dspace(expr, ispace)

                processed.append(c.rebuild(exprs=expr, ispace=ispace, dspace=dspace))

            # Substitute buffered Functions with the newly created buffers
            exprs = [uxreplace(e, subs) for e in c.exprs]
            ispace = c.ispace
            for b in buffers:
                ispace = ispace.augment(b.sub_iterators)
            processed.append(c.rebuild(exprs=exprs, ispace=ispace))

            # Also append the copy-back if `e` is the last-write of some buffers
            # E.g., `u[time + 1, x] = ub[sb1, x]`
            for b in buffers:
                if b.is_readonly:
                    continue
                try:
                    c.exprs.index(b.lastwrite)
                except ValueError:
                    continue

                dims = b.function.dimensions
                lhs = b.function[[b.lastmap.get(d, Map(d, d)).f for d in dims]]
                rhs = b.indexed[[b.lastmap.get(d, Map(d, d)).b for d in dims]]

                expr = lower_exprs(uxreplace(Eq(lhs, rhs), b.subdims_mapper))
                ispace = b.written
                dspace = derive_dspace(expr, ispace)

                processed.append(c.rebuild(exprs=expr, ispace=ispace, dspace=dspace))

        return processed


class Buffer(object):

    """
    A buffer with metadata attached.

    Parameters
    ----------
    function : DiscreteFunction
        The object for which the buffer is created.
    contracted_dims : list of Dimension
        The Dimensions in `function` to be contracted, that is to be replaced
        by ModuloDimensions.
    accessv : AccessValue
        All accesses involving `function`.
    async_degree : int, optional
        Enforce a size of `async_degree` along the contracted Dimensions.
    sregistry : SymbolRegistry
        The symbol registry, to create unique names for buffers and Dimensions.
    """

    def __init__(self, function, contracted_dims, accessv, async_degree, sregistry):
        self.function = function
        self.accessv = accessv

        self.contraction_mapper = {}
        self.index_mapper = defaultdict(dict)
        self.sub_iterators = defaultdict(list)
        self.subdims_mapper = DefaultOrderedDict(set)

        # Create the necessary ModuloDimensions for indexing into the buffer
        # E.g., `u[time,x] + u[time+1,x] -> `ub[sb0,x] + ub[sb1,x]`, where `sb0`
        # and `sb1` are ModuloDimensions starting at `time` and `time+1` respectively
        dims = list(function.dimensions)
        for d in contracted_dims:
            assert d in function.dimensions

            # Determine the buffer size, and therefore the span of the ModuloDimension,
            # along the contracting Dimension `d`
            indices = filter_ordered(i.indices[d] for i in accessv.accesses)
            slots = [i.subs({d: 0, d.spacing: 1}) for i in indices]
            try:
                size = max(slots) - min(slots) + 1
            except TypeError:
                # E.g., special case `slots=[-1 + time/factor, 2 + time/factor]`
                # Resort to the fast vector-based comparison machinery (rather than
                # the slower sympy.simplify)
                slots = [Vector(i) for i in slots]
                size = int((vmax(*slots) - vmin(*slots) + 1)[0])

            if async_degree is not None:
                if async_degree < size:
                    warning("Ignoring provided asynchronous degree as it'd be "
                            "too small for the required buffer (provided %d, "
                            "but need at least %d for `%s`)"
                            % (async_degree, size, function.name))
                else:
                    size = async_degree

            # Replace `d` with a suitable CustomDimension `bd`
            name = sregistry.make_name(prefix='db')
            bd = CustomDimension(name, 0, size-1, size, d)
            self.contraction_mapper[d] = dims[dims.index(d)] = bd

            # Finally create the ModuloDimensions as children of `bd`
            for i in indices:
                name = sregistry.make_name(prefix='sb')
                md = self.index_mapper[d][i] = ModuloDimension(name, bd, i, size)
                self.sub_iterators[d.root].append(md)

        # Track the SubDimensions used to index into `function`
        for e in accessv.mapper:
            m = {i.root: i for i in e.free_symbols
                 if isinstance(i, Dimension) and (i.is_Sub or not i.is_Derived)}
            for d, v in m.items():
                self.subdims_mapper[d].add(v)
        if any(len(v) > 1 for v in self.subdims_mapper.values()):
            # Non-uniform SubDimensions. At this point we're going to raise
            # an exception. It's either illegal or still unsupported
            for v in self.subdims_mapper.values():
                for d0, d1 in combinations(v, 2):
                    if d0.overlap(d1):
                        raise InvalidOperator("Cannot apply `buffering` to `%s` as it "
                                              "is accessed over the overlapping "
                                              " SubDimensions `<%s, %s>`" %
                                              (function, d0, d1))
            raise NotImplementedError("`buffering` does not support multiple "
                                      "non-overlapping SubDimensions yet.")
        else:
            self.subdims_mapper = {d: v.pop() for d, v in self.subdims_mapper.items()}

        # Build and sanity-check the buffer IterationIntervals
        self.itintervals_mapper = {}
        for e in accessv.mapper:
            for i in e.ispace.itintervals:
                v = self.itintervals_mapper.setdefault(i.dim, i.args)
                if v != self.itintervals_mapper[i.dim]:
                    raise NotImplementedError("Cannot apply `buffering` as the buffered "
                                              "function `%s` is accessed over multiple, "
                                              "non-compatible iteration spaces along the "
                                              "Dimension `%s`" % (function.name, i.dim))
        # Also add IterationIntervals for initialization along `x`, should `xi` be
        # the only written Dimension in the `x` hierarchy
        for d, (interval, _, _) in list(self.itintervals_mapper.items()):
            for i in d._defines:
                self.itintervals_mapper.setdefault(i, (interval.relaxed, (), Forward))

        # Finally create the actual buffer
        self.buffer = Array(name='%sb' % function.name,
                            dimensions=dims,
                            dtype=function.dtype,
                            halo=function.halo,
                            space='mapped')

    def __repr__(self):
        return "Buffer[%s,<%s>]" % (self.buffer.name,
                                    ','.join(str(i) for i in self.contraction_mapper))

    @property
    def is_read(self):
        return self.accessv.is_read

    @property
    def is_readonly(self):
        return self.is_read and self.lastwrite is None

    @property
    def lastread(self):
        return self.accessv.lastread

    @property
    def lastwrite(self):
        return self.accessv.lastwrite

    @property
    def has_uniform_subdims(self):
        return self.subdims_mapper is not None

    @cached_property
    def indexed(self):
        return self.buffer.indexed

    @cached_property
    def index_mapper_flat(self):
        ret = {}
        for mapper in self.index_mapper.values():
            ret.update(mapper)
        return ret

    @cached_property
    def writeto(self):
        """
        The `writeto` IterationSpace, that is the iteration space that must be
        iterated over in order to initialize the buffer.
        """
        intervals = []
        sub_iterators = {}
        directions = {}
        for d in self.buffer.dimensions:
            try:
                interval, si, direction = self.itintervals_mapper[d]
            except KeyError:
                # E.g., the contraction Dimension `db0`
                assert d in self.contraction_mapper.values()
                interval, si, direction = Interval(d, 0, 0), (), Forward
            intervals.append(interval)
            sub_iterators[d] = si
            directions[d] = direction

        relations = (self.buffer.dimensions,)
        intervals = IntervalGroup(intervals, relations=relations)

        return IterationSpace(intervals, sub_iterators, directions)

    @cached_property
    def written(self):
        """
        The `written` IterationSpace, that is the iteration space that must be
        iterated over in order to read all of the written buffer values.
        """
        intervals = []
        sub_iterators = {}
        directions = {}
        for dd in self.function.dimensions:
            d = dd.xreplace(self.subdims_mapper)
            try:
                interval, si, direction = self.itintervals_mapper[d]
            except KeyError:
                # E.g., d=time_sub
                assert d.is_NonlinearDerived
                d = d.root
                interval, si, direction = self.itintervals_mapper[d]
            intervals.append(interval)
            sub_iterators[d] = si + as_tuple(self.sub_iterators.get(d))
            directions[d] = direction

        relations = (tuple(i.dim for i in intervals),)
        intervals = IntervalGroup(intervals, relations=relations)

        return IterationSpace(intervals, sub_iterators, directions)

    @cached_property
    def lastmap(self):
        """
        A mapper from contracted Dimensions to a 2-tuple of indices representing,
        respectively, the "last" write to the buffer and the "last" read from the
        buffered Function. For example, `{time: (sb1, time+1)}`.
        """
        mapper = {}
        for d, m in self.index_mapper.items():
            try:
                func = max if self.written.directions[d.root] is Forward else min
                v = func(m)
            except TypeError:
                func = vmax if self.written.directions[d.root] is Forward else vmin
                v = func(*[Vector(i) for i in m])[0]
            mapper[d] = Map(m[v], v)

        return mapper

    @cached_property
    def initmap(self):
        """
        A mapper from contracted Dimensions to indices representing the min point
        for buffer initialization. For example, `{time: db0}` in the case of a
        forward-propagating `time`, or `{time: time_M - 2 + db0}` in the case
        of a backward-propagating `time`.
        """
        mapper = {}
        for d, bd in self.contraction_mapper.items():
            if self.written.directions[d.root] is Forward:
                mapper[d] = bd
            else:
                m = self.index_mapper[d]
                if d in m:
                    offset = len([i for i in m if i < d])
                    mapper[d] = d.symbolic_max - offset + bd
                else:
                    # E.g., `time/factor-1` and `time/factor+1` present, but not
                    # `time/factor`. We reconstruct `time/factor` -- the mid point
                    # logically corresponding to time_M
                    assert len(m) > 0
                    v = list(m).pop()

                    if len(m) == 1:
                        mapper[d] = v.subs(d.root, d.root.symbolic_max) + bd
                        continue

                    try:
                        p = v.args[1]
                        if len(v.args) != 2 or \
                           not is_integer(v.args[0]) or \
                           not is_integer(p - v):
                            raise ValueError
                    except (IndexError, ValueError):
                        raise NotImplementedError("Cannot apply buffering with nonlinear "
                                                  "index functions (found `%s`)" % v)
                    try:
                        # Start assuming e.g. `list(m) = [time - 1, time + 2]`
                        offset = len([i for i in m if i < p])
                    except TypeError:
                        # Actually, e.g. `list(m) = [time/factor - 1, time/factor + 2]`
                        offset = len([i for i in m if Vector(i) < Vector(p)])
                    mapper[d] = p.subs(d.root, d.root.symbolic_max) - offset + bd

        return mapper


class AccessValue(object):

    """
    A simple data structure tracking the accesses performed by a given Function
    in a sequence of Clusters.

    Parameters
    ----------
    function : Function
        The target Function.
    mapper : dict
        A mapper from expressions to Indexeds, representing all accesses to
        `function` in a sequence of expressions.
    """

    def __init__(self, function, mapper):
        self.function = function
        self.mapper = mapper

    @cached_property
    def exprs(self):
        return tuple(self.mapper)

    @cached_property
    def accesses(self):
        return tuple(flatten(as_tuple(i.reads) + as_tuple(i.write)
                             for i in self.mapper.values()))

    @cached_property
    def is_read(self):
        return any(av.reads for av in self.mapper.values())

    @cached_property
    def lastread(self):
        for e, av in reversed(self.mapper.items()):
            if av.reads:
                return e
        return None

    @cached_property
    def lastwrite(self):
        for e, av in reversed(self.mapper.items()):
            if av.write is not None:
                return e
        return None


AccessTuple = lambda: Bunch(reads=[], write=None)
Map = namedtuple('Map', 'b f')


class AccessMapper(OrderedDict):

    def __init__(self, clusters):
        mapper = DefaultOrderedDict(lambda: DefaultOrderedDict(AccessTuple))
        for c in clusters:
            for e in c.exprs:
                for i in retrieve_function_carriers(e.rhs):
                    mapper[i.function][e].reads.append(i)
                mapper[e.lhs.function][e].write = e.lhs

        super().__init__([(f, AccessValue(f, mapper[f])) for f in mapper])


def derive_dspace(expr, ispace):
    accesses = detect_accesses(expr)
    parts = {k: IntervalGroup(build_intervals(v)).relaxed
             for k, v in accesses.items() if k}
    dspace = DataSpace(ispace.intervals, parts)

    return dspace