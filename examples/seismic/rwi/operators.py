from devito.types import dense
from devito import (Eq, Operator, Function, TimeFunction, NODE, Inc, solve,
                    cos, sin, sqrt)
from examples.seismic import PointSource, Receiver


def update_kernel(name, dimension, func):
    kernels.update((name, dimension), func)


def second_order_stencil(model, u, v, H0, Hz, qu, qv, forward=True):
    """
    Creates the stencil corresponding to the second order TTI wave equation
    m * u.dt2 =  (epsilon * H0 + delta * Hz) - damp * u.dt
    m * v.dt2 =  (delta * H0 + Hz) - damp * v.dt
    """
    m, damp = model.m, model.damp

    unext = u.forward if forward else u.backward
    vnext = v.forward if forward else v.backward
    udt = u.dt if forward else u.dt.T
    vdt = v.dt if forward else v.dt.T

    # Stencils
    stencilp = solve(m * u.dt2 - H0 - qu + damp * udt, unext)
    stencilr = solve(m * v.dt2 - Hz - qv + damp * vdt, vnext)

    first_stencil = Eq(unext, stencilp)
    second_stencil = Eq(vnext, stencilr)

    stencils = [first_stencil, second_stencil]
    return stencils


def kernel_tti_2d(model, u, v, space_order, **kwargs):
    # Forward or backward
    forward = kwargs.get('forward', True)

    # Tilt and azymuth setup

    density = model.rho
    delta, epsilon = model.delta, model.epsilon


    # Get source
    qu = kwargs.get('qu', 0)
    qv = kwargs.get('qv', 0)

    if forward:
        hd2 = lambda field: density * (1/density * field.dx).dx
        vd2 = lambda field: density * (1/density * field.dy).dy
        H0 = (1+2*epsilon)*hd2(u) + sqrt(1 + 2*delta)*vd2(v)
        Hz = sqrt(1 + 2*delta)*hd2(u) + vd2(v) 
        return second_order_stencil(model, u, v, H0, Hz, qu, qv)
    else:
        H0 = ((((1+2*epsilon)*u + sqrt(1 + 2*delta)*v) * density).dx / density).dx
        Hz = (((sqrt(1 + 2*delta)*u + v) * density).dy / density).dy
        return second_order_stencil(model, u, v, H0, Hz, qu, qv, forward=forward)



def kernel_centered_2d(model, u, v, space_order, **kwargs):
    # Forward or backward
    forward = kwargs.get('forward', True)

    density = model.rho
    delta, epsilon = model.delta, model.epsilon

    # Get source
    qu = kwargs.get('qu', 0)
    qv = kwargs.get('qv', 0)

    if forward:
        hd2 = lambda field: density * (1/density * field.dx2 + field.dx * (1/density).dx)
        vd2 = lambda field: density * (1/density * field.dy2 + field.dy * (1/density).dy)
        H0 = (1+2*epsilon)*hd2(u) + 1/(1+epsilon-delta)*vd2(v) 
        Hz = (1+epsilon-delta)*(1+2*delta)*hd2(u) + vd2(v)
        return second_order_stencil(model, u, v, H0, Hz, qu, qv)
    else:
        Ha = ((1+2*epsilon)*u + (1+epsilon-delta)*(1+2*delta)*v) * density
        Hb = (1/(1+epsilon-delta)*u + v) * density
        H0 = Ha.dx * (1/density).dx + Ha.dx2 / density
        Hz = Hb.dy * (1/density).dy + Hb.dy2 / density
        return second_order_stencil(model, u, v, H0, Hz, qu, qv, forward=forward)


def ForwardOperator(model, geometry, space_order=4,
                    save=False, kernel='centered', **kwargs):
    """
    Construct an forward modelling operator in an tti media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    kernel : str, optional
        Type of discretization, centered or shifted
    """

    dt = model.grid.time_dim.spacing
    m = model.m
    time_order = 2
    stagg_u = stagg_v = None

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid, staggered=stagg_u,
                     save=geometry.nt if save else None,
                     time_order=time_order, space_order=space_order)
    v = TimeFunction(name='v', grid=model.grid, staggered=stagg_v,
                     save=geometry.nt if save else None,
                     time_order=time_order, space_order=space_order)
    src = PointSource(name='src', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # FD kernels of the PDE
    FD_kernel = kernels[(kernel, len(model.shape))]
    stencils = FD_kernel(model, u, v, space_order)

    # Source and receivers
    expr = src * dt**2 / m
    stencils += src.inject(field=u.forward, expr=expr)
    stencils += src.inject(field=v.forward, expr=expr)
    stencils += rec.interpolate(expr=u + v)

    # Substitute spacing terms to reduce flops
    return Operator(stencils, subs=model.spacing_map, name='ForwardRWI', **kwargs)


def AdjointOperator(model, geometry, space_order=4,
                    kernel='centered', **kwargs):
    """
    Construct an adjoint modelling operator in an tti media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, centered or shifted
    """

    dt = model.grid.time_dim.spacing
    m = model.m
    time_order = 1 if kernel == 'staggered' else 2
    if kernel == 'staggered':
        stagg_p = stagg_r = NODE
    else:
        stagg_p = stagg_r = None

    # Create symbols for forward wavefield, source and receivers
    p = TimeFunction(name='p', grid=model.grid, staggered=stagg_p,
                     time_order=time_order, space_order=space_order)
    r = TimeFunction(name='r', grid=model.grid, staggered=stagg_r,
                     time_order=time_order, space_order=space_order)
    srca = PointSource(name='srca', grid=model.grid, time_range=geometry.time_axis,
                       npoint=geometry.nsrc)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # FD kernels of the PDE
    expr = rec * dt / m if kernel == 'staggered' else rec * dt**2 / m
    FD_kernel = kernels[(kernel, len(model.shape))]
    if kernel == "iso":
        stencils = FD_kernel(model, p, "OT4")
        stencils += rec.inject(field=p.backward, expr=expr)
    else:
        stencils = FD_kernel(model, p, r, space_order, forward=False)
        stencils += rec.inject(field=p.backward, expr=expr)
        stencils += rec.inject(field=r.backward, expr=expr)

    # Construct expression to inject receiver values

    # Create interpolation expression for the adjoint-source
    stencils += srca.interpolate(expr=p + r)

    # Substitute spacing terms to reduce flops
    return Operator(stencils, subs=model.spacing_map, name='AdjointRWI', **kwargs)


def JacobianOperator(model, geometry, space_order=4,
                     **kwargs):
    """
    Construct a Linearized Born operator in a TTI media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, centered or staggered.
    """
    dt = model.grid.stepping_dim.spacing
    m = model.m
    time_order = 2

    # Create source and receiver symbols
    src = Receiver(name='src', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nsrc)

    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # Create wavefields and a dm field
    u0 = TimeFunction(name='u0', grid=model.grid, save=None, time_order=time_order,
                      space_order=space_order)
    v0 = TimeFunction(name='v0', grid=model.grid, save=None, time_order=time_order,
                      space_order=space_order)
    du = TimeFunction(name="du", grid=model.grid, save=None,
                      time_order=2, space_order=space_order)
    dv = TimeFunction(name="dv", grid=model.grid, save=None,
                      time_order=2, space_order=space_order)
    dm = Function(name="dm", grid=model.grid, space_order=0)

    # FD kernels of the PDE
    FD_kernel = kernels[('centered', len(model.shape))]
    eqn1 = FD_kernel(model, u0, v0, space_order)

    # Linearized source and stencil
    lin_usrc = -dm * u0.dt2
    lin_vsrc = -dm * v0.dt2

    eqn2 = FD_kernel(model, du, dv, space_order, qu=lin_usrc, qv=lin_vsrc)

    # Construct expression to inject source values, injecting at u0(t+dt)/v0(t+dt)
    src_term = src.inject(field=u0.forward, expr=src * dt**2 / m)
    src_term += src.inject(field=v0.forward, expr=src * dt**2 / m)

    # Create interpolation expression for receivers, extracting at du(t)+dv(t)
    rec_term = rec.interpolate(expr=du + dv)

    # Substitute spacing terms to reduce flops
    return Operator(eqn1 + src_term + eqn2 + rec_term, subs=model.spacing_map,
                    name='BornRWI', **kwargs)


def JacobianAdjOperator(model, geometry, space_order=4,
                        save=True, **kwargs):
    """
    Construct a linearized JacobianAdjoint modeling Operator in a TTI media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Option to store the entire (unrolled) wavefield.
    """
    dt = model.grid.stepping_dim.spacing
    m = model.m
    time_order = 2

    # Gradient symbol and wavefield symbols
    u0 = TimeFunction(name='u0', grid=model.grid, save=geometry.nt if save
                      else None, time_order=time_order, space_order=space_order)
    v0 = TimeFunction(name='v0', grid=model.grid, save=geometry.nt if save
                      else None, time_order=time_order, space_order=space_order)

    du = TimeFunction(name="du", grid=model.grid, save=None,
                      time_order=time_order, space_order=space_order)
    dv = TimeFunction(name="dv", grid=model.grid, save=None,
                      time_order=time_order, space_order=space_order)

    dm = Function(name="dm", grid=model.grid)

    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # FD kernels of the PDE
    FD_kernel = kernels[('centered', len(model.shape))]
    eqn = FD_kernel(model, du, dv, space_order, forward=False)

    dm_update = Inc(dm, - (u0 * du.dt2 + v0 * dv.dt2))

    # Add expression for receiver injection
    rec_term = rec.inject(field=du.backward, expr=rec * dt**2 / m)
    rec_term += rec.inject(field=dv.backward, expr=rec * dt**2 / m)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + rec_term + [dm_update], subs=model.spacing_map,
                    name='GradientRWI', **kwargs)


kernels = {('centered', 2): kernel_centered_2d, ('tti', 2): kernel_tti_2d}
