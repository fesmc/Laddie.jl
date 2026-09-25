
# Periodic neighbour indices along each axis.  Arrays are [ix, iy]: `_xp1`/`_xm1`
# step the first index, `_yp1`/`_ym1` the second.  These are index directions, not
# compass directions — the grid axes of a projected domain need not align with
# east/north (see the note on the shift primitives in utils.jl).  The time-step
# stencils are launched over the interior (`launch_interior!`) and use plain
# `i ± 1`; the wrap remains only where a stencil must also cover the border ring:
# the collocated cross-velocities of `NonlinearLateralViscosity` (a u-point at
# index 1 is a real face), the second-neighbour reads of the upstream momentum
# advection, and the host-side diagnostics and output averages.
# Written as a comparison of the shifted index, not `i == N ? 1 : i + 1`: Reactant
# raises this form of a wrapped read to a rotation, which shards as a halo exchange,
# and the equality form to a gather, which shards as an all-gather of the whole
# field.  So does a read wrapped along both axes at once (see `_v_half_kernel!`),
# and one inside a branch (see `_face_mass_x`).
@inline _xp1(i, N) = ifelse(i + 1 > N, i + 1 - N, i + 1)
@inline _xm1(i, N) = ifelse(i - 1 < 1, i - 1 + N, i - 1)
@inline _yp1(j, N) = ifelse(j + 1 > N, j + 1 - N, j + 1)
@inline _ym1(j, N) = ifelse(j - 1 < 1, j - 1 + N, j - 1)
# Zero where `b` is zero.  The divisor of the unused branch is swapped for one as well:
# reverse-mode AD (Enzyme) still differentiates that branch, and its zero adjoint
# times 1/0 would give NaN.
@inline _safe_div(a, b) = ifelse(iszero(b), zero(a), a / ifelse(iszero(b), one(b), b))
# The plain value of a number, without derivative parts.  Used where a value
# leaves the differentiable computation: integer indices, the dt controller,
# display and file output.  The ForwardDiff extension strips `Dual` numbers.
@inline _primal(x) = x
_float64(x) = Float64(_primal(x))
# `sqrt` for arguments that reach exactly zero (a speed at rest, a clipped
# discriminant), with a zero derivative there instead of the NaN of `0 × Inf`.  The
# same double `ifelse` as `_safe_div`, for Enzyme; the ForwardDiff extension has its
# own method for `Dual`s.
@inline _safe_sqrt(x) = ifelse(iszero(x), zero(x), sqrt(ifelse(iszero(x), one(x), x)))
# A kernel's scalar argument as a plain value.  The kernels unwrap their scalars
# with it on entry.  `getindex(x::Number)` is `x` for floats and `Dual`s; a scalar
# traced by Reactant arrives as a device reference, which `x[]` loads, so that the
# kernel body computes with a plain float (`FT(…)`, `clamp`, `promote` do not work
# on the reference).  Arrays pass through: some slots are scalars or fields.
@inline _val(x::Number) = x[]
@inline _val(x::NamedTuple) = map(_val, x)
@inline _val(x) = x

# The slip factor of a wall face, from the grounding-line and land indicators `gl`
# and `lnd` of the face and the slip factors in `walls` (`_u_walls`/`_v_walls`); zero
# off the walls.  The indicators partition the wall faces (the grounding line takes
# precedence at mixed corners), so exactly one of the two factors applies on a wall.
@inline _slip(walls, gl, lnd, i, j) = walls.slip_gl * gl[i, j] + walls.slip_land * lnd[i, j]

# Masked two-point average of `a` over the cells (i, j) and (i2, j2): the mean of the
# active ones, zero if neither is active.  The kernels form their staggered
# velocities and thicknesses with it instead of reading pre-computed fields.
@inline _face_avg(a, mask, i, j, i2, j2) =
    _safe_div(a[i, j] + a[i2, j2], mask[i, j] + mask[i2, j2])
# The same average for the cell (i, j) and its neighbour (i + di, j + dj), but zero on
# the border ring, where the pre-computed fields were never written.  For a stencil
# that reads the average at a neighbouring cell, which may be on the ring.  Both reads
# stay in bounds at every call site (the cell is a neighbour of an interior cell and
# the offset points inwards), so the average is always formed and then selected with
# `ifelse`: a branch here is raised by Reactant to a reduction Enzyme cannot
# differentiate.
@inline function _face_avg_ring0(a, mask, i, j, di, dj)
    Nx, Ny = size(a)
    inner = (1 < i < Nx) & (1 < j < Ny)
    return ifelse(inner, _face_avg(a, mask, i, j, i + di, j + dj), zero(eltype(a)))
end

# Tracer Laplacian ∇·(D ∇var), with D the Laplacian thickness averaged onto each face.
@kernel function _lapT_kernel!(
    out,
    @Const(var),
    @Const(D),
    @Const(tmask),
    dy2,
    dx2,
)
    dy2, dx2 = _val(dy2), _val(dx2)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        jp1 = j + 1
        jm1 = j - 1
        ip1 = i + 1
        im1 = i - 1
        D0jp = _face_avg(D, tmask, i, j, i, jp1)
        D0jm = _face_avg(D, tmask, i, j, i, jm1)
        D0ip = _face_avg(D, tmask, i, j, ip1, j)
        D0im = _face_avg(D, tmask, i, j, im1, j)
        flux_N = D0jp * (var[i, jp1] - var[i, j]) * tmask[i, jp1] / dy2
        flux_S = D0jm * (var[i, jm1] - var[i, j]) * tmask[i, jm1] / dy2
        flux_E = D0ip * (var[ip1, j] - var[i, j]) * tmask[ip1, j] / dx2
        flux_W = D0im * (var[im1, j] - var[i, j]) * tmask[im1, j] / dx2
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

# Upwind flux-form advection of a T-point quantity `var` (D, D·T, D·S).  The
# upwind split of each face velocity and the masked neighbour values are formed
# here rather than pre-computed, so the kernel reads only the fields themselves.
# At an open-ocean face the upwind value is the zero-gradient extrapolation of the
# cell itself when `inflow = 1` (ZeroGradientInflow), and the stored — masked,
# hence zero — neighbour value when `inflow = 0` (NoInflow).
@kernel function _upwind_advection_T_kernel!(
    out,
    @Const(var),
    @Const(U),
    @Const(V),
    @Const(tmask),
    @Const(ocn),
    @Const(umask),
    @Const(vmask),
    inflow,
    dx,
    dy,
)
    inflow, dx, dy = _val(inflow), _val(dx), _val(dy)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        jp1 = j + 1
        jm1 = j - 1
        ip1 = i + 1
        im1 = i - 1
        FT = typeof(inflow)
        z = zero(FT)
        keep = one(FT) - inflow          # weight of the raw neighbour value
        v = var[i, j]
        u_e = U[i, j]
        u_w = U[im1, j]
        v_n = V[i, j]
        v_s = V[i, jm1]
        var_n = var[i, jp1] * (inflow * tmask[i, jp1] + keep) + inflow * v * ocn[i, jp1]
        var_s = var[i, jm1] * (inflow * tmask[i, jm1] + keep) + inflow * v * ocn[i, jm1]
        var_e = var[ip1, j] * (inflow * tmask[ip1, j] + keep) + inflow * v * ocn[ip1, j]
        var_w = var[im1, j] * (inflow * tmask[im1, j] + keep) + inflow * v * ocn[im1, j]
        flux_N = -(max(v_n, z) * v + min(v_n, z) * var_n) / dy * vmask[i, j]
        flux_S = (max(v_s, z) * var_s + min(v_s, z) * v) / dy * vmask[i, jm1]
        flux_E = -(max(u_e, z) * v + min(u_e, z) * var_e) / dx * umask[i, j]
        flux_W = (max(u_w, z) * var_w + min(u_w, z) * v) / dx * umask[im1, j]
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

_inflow_weight(::ZeroGradientInflow, FT) = one(FT)
_inflow_weight(::NoInflow, FT) = zero(FT)

# Momentum advection of U.  D is masked on the fly (`Dt(i, j) = D·tmask`), and the
# staggered velocity averages are formed here from U and V (`_face_avg`).
@kernel function _upwind_advection_U_kernel!(
    out,
    @Const(D),
    @Const(tmask),
    @Const(ocn),
    @Const(V),
    @Const(vmask),
    @Const(U),
    @Const(umask),
    walls,
    dx,
    dy,
)
    walls, dx, dy = _val(walls), _val(dx), _val(dy)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        jp1 = j + 1
        jm1 = j - 1
        ip1 = i + 1
        im1 = i - 1
        FT = typeof(dx)
        Vip = _face_avg(V, vmask, i, j, ip1, j)
        Vip_jm = _face_avg_ring0(V, vmask, i, jm1, 1, 0)
        Ujp = _face_avg(U, umask, i, j, i, jp1)
        Ujm = _face_avg(U, umask, i, j, i, jm1)
        Uip = _face_avg(U, umask, i, j, ip1, j)
        Uim = _face_avg(U, umask, i, j, im1, j)
        D0 = D[i, j] * tmask[i, j]
        De = D[ip1, j] * tmask[ip1, j]
        Dn = D[i, jp1] * tmask[i, jp1]
        Ds = D[i, jm1] * tmask[i, jm1]
        Dne = D[ip1, jp1] * tmask[ip1, jp1]
        Dse = D[ip1, jm1] * tmask[ip1, jm1]
        D_N = _safe_div(
            D0 + De + Dn + Dne,
            tmask[i, j] + tmask[ip1, j] + tmask[i, jp1] + tmask[ip1, jp1],
        )
        D_S = _safe_div(
            D0 + De + Ds + Dse,
            tmask[i, j] + tmask[ip1, j] + tmask[i, jm1] + tmask[ip1, jm1],
        )
        ocn_e = ocn[ip1, j]
        D_E = De + ocn_e * D0
        D_W = D0 + ocn[i, j] * De
        u = U[i, j]
        sU = sign(u)
        wallN = _slip(walls, walls.glN, walls.lndN, i, j)
        wallS = _slip(walls, walls.glS, walls.lndS, i, j)
        flux_N = -D_N * Vip * (Ujp - wallN * u) / dy
        flux_S = D_S * Vip_jm * (Ujm - wallS * u) / dy
        flux_E = -D_E * Uip * (Uip - (one(FT) - sU) * u * ocn_e) / dx
        flux_W = D_W * Uim * (Uim - sU * u * ocn[i, j]) / dx
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

# Momentum advection of V; the mirror image of `_upwind_advection_U_kernel!`.
@kernel function _upwind_advection_V_kernel!(
    out,
    @Const(D),
    @Const(tmask),
    @Const(ocn),
    @Const(V),
    @Const(vmask),
    @Const(U),
    @Const(umask),
    walls,
    dx,
    dy,
)
    walls, dx, dy = _val(walls), _val(dx), _val(dy)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        jp1 = j + 1
        jm1 = j - 1
        ip1 = i + 1
        im1 = i - 1
        FT = typeof(dx)
        Vjp = _face_avg(V, vmask, i, j, i, jp1)
        Vjm = _face_avg(V, vmask, i, j, i, jm1)
        Vip = _face_avg(V, vmask, i, j, ip1, j)
        Vim = _face_avg(V, vmask, i, j, im1, j)
        Ujp = _face_avg(U, umask, i, j, i, jp1)
        Ujp_im = _face_avg_ring0(U, umask, im1, j, 0, 1)
        D0 = D[i, j] * tmask[i, j]
        Dn = D[i, jp1] * tmask[i, jp1]
        De = D[ip1, j] * tmask[ip1, j]
        Dw = D[im1, j] * tmask[im1, j]
        Dne = D[ip1, jp1] * tmask[ip1, jp1]
        Dnw = D[im1, jp1] * tmask[im1, jp1]
        D_E = _safe_div(
            D0 + Dn + De + Dne,
            tmask[i, j] + tmask[i, jp1] + tmask[ip1, j] + tmask[ip1, jp1],
        )
        D_W = _safe_div(
            D0 + Dn + Dw + Dnw,
            tmask[i, j] + tmask[i, jp1] + tmask[im1, j] + tmask[im1, jp1],
        )
        ocn_n = ocn[i, jp1]
        D_N = Dn + ocn_n * D0
        D_S = D0 + ocn[i, j] * Dn
        v = V[i, j]
        sV = sign(v)
        flux_N = -D_N * Vjp * (Vjp - (one(FT) - sV) * v * ocn_n) / dy
        flux_S = D_S * Vjm * (Vjm - sV * v * ocn[i, j]) / dy
        wallE = _slip(walls, walls.glE, walls.lndE, i, j)
        wallW = _slip(walls, walls.glW, walls.lndW, i, j)
        flux_E = -D_E * Ujp * (Vip - wallE * v) / dx
        flux_W = D_W * Ujp_im * (Vim - wallW * v) / dx
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

@kernel function _laplace_U_kernel!(
    out,
    @Const(var),
    @Const(D0),
    @Const(tmask),
    @Const(ocn),
    walls,
    A_h,
    dx2,
    dy2,
)
    walls, A_h, dx2, dy2 = _val(walls), _val(A_h), _val(dx2), _val(dy2)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    FT = typeof(A_h)
    @inbounds begin
        jp1 = j + 1
        jm1 = j - 1
        ip1 = i + 1
        im1 = i - 1
        o = one(FT)
        v = var[i, j]
        # D on the u-points (and at the two y-neighbours), masked: the face average
        # of the Laplacian thickness D0, zero on the border ring.
        DU = _face_avg(D0, tmask, i, j, ip1, j) * tmask[i, j]
        DU_jp = _face_avg_ring0(D0, tmask, i, jp1, 1, 0) * tmask[i, jp1]
        DU_jm = _face_avg_ring0(D0, tmask, i, jm1, 1, 0) * tmask[i, jm1]
        # Per-face wall drag, zero off the walls.
        dragN = _slip(walls, walls.glN, walls.lndN, i, j) * DU * v / dy2
        dragS = _slip(walls, walls.glS, walls.lndS, i, j) * DU * v / dy2
        jpD = _safe_div(DU + DU_jp, tmask[i, j] + tmask[i, jp1])
        jmD = _safe_div(DU + DU_jm, tmask[i, j] + tmask[i, jm1])
        flux_N = jpD * (var[i, jp1] - v) / dy2 * (o - ocn[i, jp1]) - dragN
        flux_S = jmD * (var[i, jm1] - v) / dy2 * (o - ocn[i, jm1]) - dragS
        flux_E = D0[ip1, j] * (var[ip1, j] - v) / dx2 * (o - ocn[ip1, j])
        flux_W = D0[i, j] * (var[im1, j] - v) / dx2 * (o - ocn[i, j])
        out[i, j] = (flux_N + flux_S + flux_E + flux_W) * A_h
    end
end

@kernel function _laplace_V_kernel!(
    out,
    @Const(var),
    @Const(D0),
    @Const(tmask),
    @Const(ocn),
    walls,
    A_h,
    dx2,
    dy2,
)
    walls, A_h, dx2, dy2 = _val(walls), _val(A_h), _val(dx2), _val(dy2)
    FT = typeof(A_h)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        jp1 = j + 1
        jm1 = j - 1
        ip1 = i + 1
        im1 = i - 1
        o = one(FT)
        v = var[i, j]
        # D on the v-points (and at the two x-neighbours); see _laplace_U_kernel!.
        DV = _face_avg(D0, tmask, i, j, i, jp1) * tmask[i, j]
        DV_ip = _face_avg_ring0(D0, tmask, ip1, j, 0, 1) * tmask[ip1, j]
        DV_im = _face_avg_ring0(D0, tmask, im1, j, 0, 1) * tmask[im1, j]
        dragE = _slip(walls, walls.glE, walls.lndE, i, j) * DV * v / dx2
        dragW = _slip(walls, walls.glW, walls.lndW, i, j) * DV * v / dx2
        ipD = _safe_div(DV + DV_ip, tmask[i, j] + tmask[ip1, j])
        imD = _safe_div(DV + DV_im, tmask[i, j] + tmask[im1, j])
        flux_N = D0[i, jp1] * (var[i, jp1] - v) / dy2 * (o - ocn[i, jp1])
        flux_S = D0[i, j] * (var[i, jm1] - v) / dy2 * (o - ocn[i, j])
        flux_E = ipD * (var[ip1, j] - v) / dx2 * (o - ocn[ip1, j]) - dragE
        flux_W = imD * (var[im1, j] - v) / dx2 * (o - ocn[im1, j]) - dragW
        out[i, j] = (flux_N + flux_S + flux_E + flux_W) * A_h
    end
end

# Shear-scaled (NonlinearLateralViscosity) counterparts of _laplace_U_kernel!/
# _laplace_V_kernel!: same geometry and masking, but each interior flux term gets
# its own coefficient visc_? * |Δu| in place of a single constant A_h, where |Δu|
# is the magnitude of the full velocity-difference *vector* across that face
# (the reference's dUabs), not just the component being diffused — hence `other`,
# the cross-component collocated onto this component's points.  The kernels get it
# half-collocated (averaged along the other axis only, `_v_half_kernel!`) and finish
# the average on the fly (`_other_x`, `_other_y`): stored whole, it takes a read
# wrapped along both axes at once, which Reactant raises to a gather.  Where
# visc_x/visc_y = C_visc * dx/100 and C_visc * dy/100 carry the reference's
# dUabs * triCw / 100 scaling (laddie_velocity.f90:260).  The grounding-line/land
# wall-drag terms keep the plain, unscaled A_h_wall — mirroring the reference,
# which never scales its border term by dUabs (laddie_velocity.f90:249-254).
@kernel function _nonlinear_laplace_U_kernel!(
    out,
    @Const(var),
    @Const(half),
    @Const(D0),
    @Const(tmask),
    @Const(ocn),
    walls,
    A_h_wall,
    visc_x,
    visc_y,
    dx2,
    dy2,
    Nx,
)
    walls, A_h_wall, visc_x, visc_y, dx2, dy2 =
        _val(walls), _val(A_h_wall), _val(visc_x), _val(visc_y), _val(dx2), _val(dy2)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    FT = typeof(A_h_wall)
    @inbounds begin
        jp1 = j + 1
        jm1 = j - 1
        ip1 = i + 1
        im1 = i - 1
        o = one(FT)
        v = var[i, j]
        # D on the u-points (and at the two y-neighbours), masked: the face average
        # of the Laplacian thickness D0, zero on the border ring.
        DU = _face_avg(D0, tmask, i, j, ip1, j) * tmask[i, j]
        DU_jp = _face_avg_ring0(D0, tmask, i, jp1, 1, 0) * tmask[i, jp1]
        DU_jm = _face_avg_ring0(D0, tmask, i, jm1, 1, 0) * tmask[i, jm1]
        dragN = A_h_wall * _slip(walls, walls.glN, walls.lndN, i, j) * DU * v / dy2
        dragS = A_h_wall * _slip(walls, walls.glS, walls.lndS, i, j) * DU * v / dy2
        jpD = _safe_div(DU + DU_jp, tmask[i, j] + tmask[i, jp1])
        jmD = _safe_div(DU + DU_jm, tmask[i, j] + tmask[i, jm1])
        ov = _other_x(half, i, j, Nx)
        dN = var[i, jp1] - v
        dS = var[i, jm1] - v
        dE = var[ip1, j] - v
        dW = var[im1, j] - v
        # |Δu| across each face: this component's difference combined with the
        # cross-component's difference over the same displacement.
        aN = _safe_sqrt(dN * dN + (_other_x(half, i, jp1, Nx) - ov)^2)
        aS = _safe_sqrt(dS * dS + (_other_x(half, i, jm1, Nx) - ov)^2)
        aE = _safe_sqrt(dE * dE + (_other_x(half, ip1, j, Nx) - ov)^2)
        aW = _safe_sqrt(dW * dW + (_other_x(half, im1, j, Nx) - ov)^2)
        flux_N = visc_y * aN * jpD * dN / dy2 * (o - ocn[i, jp1]) - dragN
        flux_S = visc_y * aS * jmD * dS / dy2 * (o - ocn[i, jm1]) - dragS
        flux_E = visc_x * aE * D0[ip1, j] * dE / dx2 * (o - ocn[ip1, j])
        flux_W = visc_x * aW * D0[i, j] * dW / dx2 * (o - ocn[i, j])
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

@kernel function _nonlinear_laplace_V_kernel!(
    out,
    @Const(var),
    @Const(half),
    @Const(D0),
    @Const(tmask),
    @Const(ocn),
    walls,
    A_h_wall,
    visc_x,
    visc_y,
    dx2,
    dy2,
    Ny,
)
    walls, A_h_wall, visc_x, visc_y, dx2, dy2 =
        _val(walls), _val(A_h_wall), _val(visc_x), _val(visc_y), _val(dx2), _val(dy2)
    FT = typeof(A_h_wall)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        jp1 = j + 1
        jm1 = j - 1
        ip1 = i + 1
        im1 = i - 1
        o = one(FT)
        v = var[i, j]
        # D on the v-points (and at the two x-neighbours); see _laplace_U_kernel!.
        DV = _face_avg(D0, tmask, i, j, i, jp1) * tmask[i, j]
        DV_ip = _face_avg_ring0(D0, tmask, ip1, j, 0, 1) * tmask[ip1, j]
        DV_im = _face_avg_ring0(D0, tmask, im1, j, 0, 1) * tmask[im1, j]
        dragE = A_h_wall * _slip(walls, walls.glE, walls.lndE, i, j) * DV * v / dx2
        dragW = A_h_wall * _slip(walls, walls.glW, walls.lndW, i, j) * DV * v / dx2
        ipD = _safe_div(DV + DV_ip, tmask[i, j] + tmask[ip1, j])
        imD = _safe_div(DV + DV_im, tmask[i, j] + tmask[im1, j])
        ov = _other_y(half, i, j, Ny)
        dN = var[i, jp1] - v
        dS = var[i, jm1] - v
        dE = var[ip1, j] - v
        dW = var[im1, j] - v
        # See _nonlinear_laplace_U_kernel! for the |Δu| composition.
        aN = _safe_sqrt(dN * dN + (_other_y(half, i, jp1, Ny) - ov)^2)
        aS = _safe_sqrt(dS * dS + (_other_y(half, i, jm1, Ny) - ov)^2)
        aE = _safe_sqrt(dE * dE + (_other_y(half, ip1, j, Ny) - ov)^2)
        aW = _safe_sqrt(dW * dW + (_other_y(half, im1, j, Ny) - ov)^2)
        flux_N = visc_y * aN * D0[i, jp1] * dN / dy2 * (o - ocn[i, jp1])
        flux_S = visc_y * aS * D0[i, j] * dS / dy2 * (o - ocn[i, j])
        flux_E = visc_x * aE * ipD * dE / dx2 * (o - ocn[ip1, j]) - dragE
        flux_W = visc_x * aW * imD * dW / dx2 * (o - ocn[im1, j]) - dragW
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

function upwind_advection_T(out, m, var)
    launch_interior!(
        _upwind_advection_T_kernel!,
        out,
        var,
        m.U.present,
        m.V.present,
        m.tmask,
        m.ocn,
        m.umask,
        m.vmask,
        _inflow_weight(m.boundary.open_ocean, m.FT),
        m.dx,
        m.dy,
    )
    return out
end

# ---------------------------------------------------------------------------
# Donor-cell momentum advection (UpstreamMomentumAdvection)
#
# Flux form, built so the momentum fluxes ride on the *mass* fluxes of the
# thickness equation: `_face_mass_x`/`_face_mass_y` are the donor-cell `D·u` of
# `_upwind_advection_T_kernel!` (same zero-gradient treatment of an ocean
# neighbour, same `umask`/`vmask` gate), averaged onto the staggered face from
# the two adjacent T-cell faces.  Momentum comes from the upstream side of that
# flux.  Consistency with discrete continuity is the point: a centred face
# thickness with an upstream velocity lets momentum cross faces that carry no
# mass, and blows up on a real geometry.
# ---------------------------------------------------------------------------

# Mass flux through the x-face at u-point (i, j), i.e. between T(i, j) and
# T(i+1, j).  Zero-gradient thickness when the donor side is open ocean.  `ifelse`,
# not a branch: the read at `ip1` wraps on the last column, and Reactant raises a
# wrapped read inside a branch to a gather (an all-gather when sharded).
@inline function _face_mass_x(U, D, tmask, ocn, umask, i, j, Nx)
    @inbounds begin
        ip1 = _xp1(i, Nx)
        u = U[i, j]
        Dd = ifelse(
            u > zero(u),
            D[i, j] * tmask[i, j] + D[ip1, j] * ocn[i, j],
            D[ip1, j] * tmask[ip1, j] + D[i, j] * ocn[ip1, j],
        )
        return umask[i, j] * u * Dd
    end
end

# Mass flux through the y-face at v-point (i, j), between T(i, j) and T(i, j+1).
@inline function _face_mass_y(V, D, tmask, ocn, vmask, i, j, Ny)
    @inbounds begin
        jp1 = _yp1(j, Ny)
        v = V[i, j]
        Dd = ifelse(
            v > zero(v),
            D[i, j] * tmask[i, j] + D[i, jp1] * ocn[i, j],
            D[i, jp1] * tmask[i, jp1] + D[i, j] * ocn[i, jp1],
        )
        return vmask[i, j] * v * Dd
    end
end

# Upstream velocity on a face, with a zero-gradient fallback when the donor
# point is inactive (ice front); at a wall the flux is zero anyway.
@inline _donor(q_own, q_nb, mask_nb, take_own) =
    take_own ? q_own : (mask_nb > zero(mask_nb) ? q_nb : q_own)

@kernel function _upstream_advection_U_kernel!(
    out,
    @Const(D),
    @Const(U),
    @Const(V),
    @Const(tmask),
    @Const(ocn),
    @Const(umask),
    @Const(vmask),
    dx,
    dy,
    Nx,
    Ny,
)
    dx, dy = _val(dx), _val(dy)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        ip1 = i + 1
        im1 = i - 1
        jp1 = j + 1
        jm1 = j - 1
        u = U[i, j]
        half = one(u) / 2
        # Face mass fluxes of the U control volume: east and west sit on T-points,
        # north and south on corners, each the mean of the two T-cell faces there.
        MxE =
            half * (
                _face_mass_x(U, D, tmask, ocn, umask, i, j, Nx) +
                _face_mass_x(U, D, tmask, ocn, umask, ip1, j, Nx)
            )
        MxW =
            half * (
                _face_mass_x(U, D, tmask, ocn, umask, im1, j, Nx) +
                _face_mass_x(U, D, tmask, ocn, umask, i, j, Nx)
            )
        MyN =
            half * (
                _face_mass_y(V, D, tmask, ocn, vmask, i, j, Ny) +
                _face_mass_y(V, D, tmask, ocn, vmask, ip1, j, Ny)
            )
        MyS =
            half * (
                _face_mass_y(V, D, tmask, ocn, vmask, i, jm1, Ny) +
                _face_mass_y(V, D, tmask, ocn, vmask, ip1, jm1, Ny)
            )
        z = zero(u)
        qE = _donor(u, U[ip1, j], umask[ip1, j], MxE > z)
        qW = _donor(u, U[im1, j], umask[im1, j], MxW <= z)
        qN = _donor(u, U[i, jp1], umask[i, jp1], MyN > z)
        qS = _donor(u, U[i, jm1], umask[i, jm1], MyS <= z)
        out[i, j] = -((MxE * qE - MxW * qW) / dx + (MyN * qN - MyS * qS) / dy)
    end
end

@kernel function _upstream_advection_V_kernel!(
    out,
    @Const(D),
    @Const(U),
    @Const(V),
    @Const(tmask),
    @Const(ocn),
    @Const(umask),
    @Const(vmask),
    dx,
    dy,
    Nx,
    Ny,
)
    dx, dy = _val(dx), _val(dy)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        ip1 = i + 1
        im1 = i - 1
        jp1 = j + 1
        jm1 = j - 1
        v = V[i, j]
        half = one(v) / 2
        MyN =
            half * (
                _face_mass_y(V, D, tmask, ocn, vmask, i, j, Ny) +
                _face_mass_y(V, D, tmask, ocn, vmask, i, jp1, Ny)
            )
        MyS =
            half * (
                _face_mass_y(V, D, tmask, ocn, vmask, i, jm1, Ny) +
                _face_mass_y(V, D, tmask, ocn, vmask, i, j, Ny)
            )
        MxE =
            half * (
                _face_mass_x(U, D, tmask, ocn, umask, i, j, Nx) +
                _face_mass_x(U, D, tmask, ocn, umask, i, jp1, Nx)
            )
        MxW =
            half * (
                _face_mass_x(U, D, tmask, ocn, umask, im1, j, Nx) +
                _face_mass_x(U, D, tmask, ocn, umask, im1, jp1, Nx)
            )
        z = zero(v)
        qE = _donor(v, V[ip1, j], vmask[ip1, j], MxE > z)
        qW = _donor(v, V[im1, j], vmask[im1, j], MxW <= z)
        qN = _donor(v, V[i, jp1], vmask[i, jp1], MyN > z)
        qS = _donor(v, V[i, jm1], vmask[i, jm1], MyS <= z)
        out[i, j] = -((MxE * qE - MxW * qW) / dx + (MyN * qN - MyS * qS) / dy)
    end
end

# Dispatch on `Params.momentum_advection`; the centred path is v1.x's.
upwind_advection_U(m) = _advect_U(m, m.momentum_advection)
upwind_advection_V(m) = _advect_V(m, m.momentum_advection)

function _advect_U(m, ::UpstreamMomentumAdvection)
    nx, ny = size(m.U.present)
    launch_interior!(
        _upstream_advection_U_kernel!,
        m.adv,
        m.D.present,
        m.U.present,
        m.V.present,
        m.tmask,
        m.ocn,
        m.umask,
        m.vmask,
        m.dx,
        m.dy,
        nx,
        ny,
    )
    return m.adv
end

function _advect_V(m, ::UpstreamMomentumAdvection)
    nx, ny = size(m.V.present)
    launch_interior!(
        _upstream_advection_V_kernel!,
        m.adv,
        m.D.present,
        m.U.present,
        m.V.present,
        m.tmask,
        m.ocn,
        m.umask,
        m.vmask,
        m.dx,
        m.dy,
        nx,
        ny,
    )
    return m.adv
end

function _advect_U(m, ::CentredMomentumAdvection)
    launch_interior!(
        _upwind_advection_U_kernel!,
        m.adv,
        m.D.present,
        m.tmask,
        m.ocn,
        m.V.present,
        m.vmask,
        m.U.present,
        m.umask,
        _u_walls(m, _advection_slips(m)),
        m.dx,
        m.dy,
    )
    return m.adv
end
function _advect_V(m, ::CentredMomentumAdvection)
    launch_interior!(
        _upwind_advection_V_kernel!,
        m.adv,
        m.D.present,
        m.tmask,
        m.ocn,
        m.V.present,
        m.vmask,
        m.U.present,
        m.umask,
        _v_walls(m, _advection_slips(m)),
        m.dx,
        m.dy,
    )
    return m.adv
end
function laplace_T(out, m, var)
    launch_interior!(
        _lapT_kernel!,
        out,
        var,
        laplacian_thickness(m),
        m.tmask,
        m.dy^2,
        m.dx^2,
    )
    return out
end
laplace_U(m) = laplace_U(m, m.lateral_viscosity)
laplace_V(m) = laplace_V(m, m.lateral_viscosity)

# PrescribedLateralViscosity: the plain Laplacian kernel, with the global A_h
# folded into its output.
function laplace_U(m, ::PrescribedLateralViscosity)
    launch_interior!(
        _laplace_U_kernel!,
        m.lap,
        m.U.past,
        laplacian_thickness(m),
        m.tmask,
        m.ocn,
        _u_walls(m, _wall_slips(m)),
        m.A_h,
        m.dx^2,
        m.dy^2,
    )
    return m.lap
end
function laplace_V(m, ::PrescribedLateralViscosity)
    launch_interior!(
        _laplace_V_kernel!,
        m.lap,
        m.V.past,
        laplacian_thickness(m),
        m.tmask,
        m.ocn,
        _v_walls(m, _wall_slips(m)),
        m.A_h,
        m.dx^2,
        m.dy^2,
    )
    return m.lap
end

# NonlinearLateralViscosity: the shear-scaled coefficient is per-face, so it must
# be applied inside the kernel rather than as a post-hoc scalar multiply; wall
# drag keeps using the plain, unscaled m.A_h (see _nonlinear_laplace_U_kernel!).
function laplace_U(m, lv::NonlinearLateralViscosity)
    nx, ny = size(m.V.past)
    # V collocated onto the U points, so the kernel can form |Δu| across a face.
    # Same 4-point average the drag term uses for the speed magnitude; its y half
    # is stored in the `Dq` work buffer, which only the tracer steps use.
    Vy = m.Dq
    launch!(_v_half_kernel!, Vy, m.V.past, ny)
    launch_interior!(
        _nonlinear_laplace_U_kernel!,
        m.lap,
        m.U.past,
        Vy,
        laplacian_thickness(m),
        m.tmask,
        m.ocn,
        _u_walls(m, _wall_slips(m)),
        m.A_h,
        lv.C_visc * m.dx / 100,
        lv.C_visc * m.dy / 100,
        m.dx^2,
        m.dy^2,
        nx,
    )
    return m.lap
end
function laplace_V(m, lv::NonlinearLateralViscosity)
    nx, ny = size(m.U.past)
    # U collocated onto the V points (x half in `Dq`); mirrors laplace_U above.
    Ux = m.Dq
    launch!(_u_half_kernel!, Ux, m.U.past, nx)
    launch_interior!(
        _nonlinear_laplace_V_kernel!,
        m.lap,
        m.V.past,
        Ux,
        laplacian_thickness(m),
        m.tmask,
        m.ocn,
        _v_walls(m, _wall_slips(m)),
        m.A_h,
        lv.C_visc * m.dx / 100,
        lv.C_visc * m.dy / 100,
        m.dx^2,
        m.dy^2,
        ny,
    )
    return m.lap
end

# 4-point collocation of the cross-velocity component, as `ip_half(jm_half(V))`
# and `jp_half(im_half(U))` without the intermediate arrays, in two halves: the
# kernels below store the average along the one axis, and the Laplacian kernels
# take the average of that along the other (`_other_x`, `_other_y`).
@kernel function _v_half_kernel!(out, @Const(V), Ny)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = (V[i, j] + V[i, _ym1(j, Ny)]) / 2
end
@kernel function _u_half_kernel!(out, @Const(U), Nx)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = (U[i, j] + U[_xm1(i, Nx), j]) / 2
end
@inline _other_x(half, i, j, Nx) = @inbounds (half[i, j] + half[_xp1(i, Nx), j]) / 2
@inline _other_y(half, i, j, Ny) = @inbounds (half[i, j] + half[i, _yp1(j, Ny)]) / 2
