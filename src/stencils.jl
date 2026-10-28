
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
# and `lnd` of the face and the slip factors `slip_gl`, `slip_land` (`_u_walls`/
# `_v_walls`); zero off the walls.  The indicators partition the wall faces (the
# grounding line takes precedence at mixed corners), so exactly one of the two
# factors applies on a wall.
#
# The helpers here that read arrays carry their own `@inbounds`, like `_other_x`
# and `_face_mass_x`: a kernel's `@inbounds` block does not reach into a called
# function, even an inlined one, and the bounds checks left there stop the CPU
# kernels from vectorising (3–4x slower).
@inline _slip(slip_gl, slip_land, gl, lnd, i, j) =
    @inbounds slip_gl * gl[i, j] + slip_land * lnd[i, j]

# Masked two-point average of `a` over the cells (i, j) and (i2, j2): the mean of the
# active ones, zero if neither is active.  The kernels form their staggered
# velocities and thicknesses with it instead of reading pre-computed fields.
@inline _face_avg(a, mask, i, j, i2, j2) =
    @inbounds _safe_div(a[i, j] + a[i2, j2], mask[i, j] + mask[i2, j2])
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

# Tracer Laplacian ∇·(H ∇var), with H the Laplacian thickness averaged onto each face.
@kernel function _lapT_kernel!(
    out,
    @Const(var),
    @Const(H),
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
        H0jp = _face_avg(H, tmask, i, j, i, jp1)
        H0jm = _face_avg(H, tmask, i, j, i, jm1)
        H0ip = _face_avg(H, tmask, i, j, ip1, j)
        H0im = _face_avg(H, tmask, i, j, im1, j)
        flux_N = H0jp * (var[i, jp1] - var[i, j]) * tmask[i, jp1] / dy2
        flux_S = H0jm * (var[i, jm1] - var[i, j]) * tmask[i, jm1] / dy2
        flux_E = H0ip * (var[ip1, j] - var[i, j]) * tmask[ip1, j] / dx2
        flux_W = H0im * (var[im1, j] - var[i, j]) * tmask[im1, j] / dx2
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

# Upwind flux-form advection of a T-point quantity `var` (H, H·T, H·S).  The
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
    kinks,
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
        eu = _kink_width(kinks, :u)
        flux_N = -(_smax0(kinks, v_n, eu) * v + _sneg(kinks, v_n, eu) * var_n) / dy * vmask[i, j]
        flux_S = (_smax0(kinks, v_s, eu) * var_s + _sneg(kinks, v_s, eu) * v) / dy * vmask[i, jm1]
        flux_E = -(_smax0(kinks, u_e, eu) * v + _sneg(kinks, u_e, eu) * var_e) / dx * umask[i, j]
        flux_W = (_smax0(kinks, u_w, eu) * var_w + _sneg(kinks, u_w, eu) * v) / dx * umask[im1, j]
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

_inflow_weight(::ZeroGradientInflow, FT) = one(FT)
_inflow_weight(::NoInflow, FT) = zero(FT)

# Momentum advection of U.  H is masked on the fly (`Ht(i, j) = H·tmask`), and the
# staggered velocity averages are formed here from U and V (`_face_avg`).
@kernel function _upwind_advection_U_kernel!(
    out,
    @Const(H),
    @Const(tmask),
    @Const(ocn),
    @Const(V),
    @Const(vmask),
    @Const(U),
    @Const(umask),
    @Const(glN),
    @Const(glS),
    @Const(lndN),
    @Const(lndS),
    slip_gl,
    slip_land,
    dx,
    dy,
    kinks,
)
    slip_gl, slip_land, dx, dy = _val(slip_gl), _val(slip_land), _val(dx), _val(dy)
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
        H0 = H[i, j] * tmask[i, j]
        He = H[ip1, j] * tmask[ip1, j]
        Hn = H[i, jp1] * tmask[i, jp1]
        Hs = H[i, jm1] * tmask[i, jm1]
        Hne = H[ip1, jp1] * tmask[ip1, jp1]
        Hse = H[ip1, jm1] * tmask[ip1, jm1]
        H_N = _safe_div(
            H0 + He + Hn + Hne,
            tmask[i, j] + tmask[ip1, j] + tmask[i, jp1] + tmask[ip1, jp1],
        )
        H_S = _safe_div(
            H0 + He + Hs + Hse,
            tmask[i, j] + tmask[ip1, j] + tmask[i, jm1] + tmask[ip1, jm1],
        )
        ocn_e = ocn[ip1, j]
        H_E = He + ocn_e * H0
        H_W = H0 + ocn[i, j] * He
        u = U[i, j]
        absu = _sabs(kinks, u, _kink_width(kinks, :u))   # sign(u)·u
        wallN = _slip(slip_gl, slip_land, glN, lndN, i, j)
        wallS = _slip(slip_gl, slip_land, glS, lndS, i, j)
        flux_N = -H_N * Vip * (Ujp - wallN * u) / dy
        flux_S = H_S * Vip_jm * (Ujm - wallS * u) / dy
        flux_E = -H_E * Uip * (Uip - (u - absu) * ocn_e) / dx
        flux_W = H_W * Uim * (Uim - absu * ocn[i, j]) / dx
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

# Momentum advection of V; the mirror image of `_upwind_advection_U_kernel!`.
@kernel function _upwind_advection_V_kernel!(
    out,
    @Const(H),
    @Const(tmask),
    @Const(ocn),
    @Const(V),
    @Const(vmask),
    @Const(U),
    @Const(umask),
    @Const(glE),
    @Const(glW),
    @Const(lndE),
    @Const(lndW),
    slip_gl,
    slip_land,
    dx,
    dy,
    kinks,
)
    slip_gl, slip_land, dx, dy = _val(slip_gl), _val(slip_land), _val(dx), _val(dy)
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
        H0 = H[i, j] * tmask[i, j]
        Hn = H[i, jp1] * tmask[i, jp1]
        He = H[ip1, j] * tmask[ip1, j]
        Hw = H[im1, j] * tmask[im1, j]
        Hne = H[ip1, jp1] * tmask[ip1, jp1]
        Hnw = H[im1, jp1] * tmask[im1, jp1]
        H_E = _safe_div(
            H0 + Hn + He + Hne,
            tmask[i, j] + tmask[i, jp1] + tmask[ip1, j] + tmask[ip1, jp1],
        )
        H_W = _safe_div(
            H0 + Hn + Hw + Hnw,
            tmask[i, j] + tmask[i, jp1] + tmask[im1, j] + tmask[im1, jp1],
        )
        ocn_n = ocn[i, jp1]
        H_N = Hn + ocn_n * H0
        H_S = H0 + ocn[i, j] * Hn
        v = V[i, j]
        absv = _sabs(kinks, v, _kink_width(kinks, :u))   # sign(v)·v
        flux_N = -H_N * Vjp * (Vjp - (v - absv) * ocn_n) / dy
        flux_S = H_S * Vjm * (Vjm - absv * ocn[i, j]) / dy
        wallE = _slip(slip_gl, slip_land, glE, lndE, i, j)
        wallW = _slip(slip_gl, slip_land, glW, lndW, i, j)
        flux_E = -H_E * Ujp * (Vip - wallE * v) / dx
        flux_W = H_W * Ujp_im * (Vim - wallW * v) / dx
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

@kernel function _laplace_U_kernel!(
    out,
    @Const(var),
    @Const(H0),
    @Const(tmask),
    @Const(ocn),
    @Const(glN),
    @Const(glS),
    @Const(lndN),
    @Const(lndS),
    slip_gl,
    slip_land,
    A_h,
    dx2,
    dy2,
)
    slip_gl, slip_land, A_h, dx2, dy2 = _val(slip_gl), _val(slip_land), _val(A_h), _val(dx2), _val(dy2)
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
        # H on the u-points (and at the two y-neighbours), masked: the face average
        # of the Laplacian thickness H0.  A neighbour on the border ring is masked
        # out by its `tmask`, so its average needs no ring guard.
        HU = _face_avg(H0, tmask, i, j, ip1, j) * tmask[i, j]
        HU_jp = _face_avg(H0, tmask, i, jp1, ip1, jp1) * tmask[i, jp1]
        HU_jm = _face_avg(H0, tmask, i, jm1, ip1, jm1) * tmask[i, jm1]
        # Per-face wall drag, zero off the walls.
        dragN = _slip(slip_gl, slip_land, glN, lndN, i, j) * HU * v / dy2
        dragS = _slip(slip_gl, slip_land, glS, lndS, i, j) * HU * v / dy2
        jpH = _safe_div(HU + HU_jp, tmask[i, j] + tmask[i, jp1])
        jmH = _safe_div(HU + HU_jm, tmask[i, j] + tmask[i, jm1])
        flux_N = jpH * (var[i, jp1] - v) / dy2 * (o - ocn[i, jp1]) - dragN
        flux_S = jmH * (var[i, jm1] - v) / dy2 * (o - ocn[i, jm1]) - dragS
        flux_E = H0[ip1, j] * (var[ip1, j] - v) / dx2 * (o - ocn[ip1, j])
        flux_W = H0[i, j] * (var[im1, j] - v) / dx2 * (o - ocn[i, j])
        out[i, j] = (flux_N + flux_S + flux_E + flux_W) * A_h
    end
end

@kernel function _laplace_V_kernel!(
    out,
    @Const(var),
    @Const(H0),
    @Const(tmask),
    @Const(ocn),
    @Const(glE),
    @Const(glW),
    @Const(lndE),
    @Const(lndW),
    slip_gl,
    slip_land,
    A_h,
    dx2,
    dy2,
)
    slip_gl, slip_land, A_h, dx2, dy2 = _val(slip_gl), _val(slip_land), _val(A_h), _val(dx2), _val(dy2)
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
        # H on the v-points (and at the two x-neighbours); see _laplace_U_kernel!.
        HV = _face_avg(H0, tmask, i, j, i, jp1) * tmask[i, j]
        HV_ip = _face_avg(H0, tmask, ip1, j, ip1, jp1) * tmask[ip1, j]
        HV_im = _face_avg(H0, tmask, im1, j, im1, jp1) * tmask[im1, j]
        dragE = _slip(slip_gl, slip_land, glE, lndE, i, j) * HV * v / dx2
        dragW = _slip(slip_gl, slip_land, glW, lndW, i, j) * HV * v / dx2
        ipH = _safe_div(HV + HV_ip, tmask[i, j] + tmask[ip1, j])
        imH = _safe_div(HV + HV_im, tmask[i, j] + tmask[im1, j])
        flux_N = H0[i, jp1] * (var[i, jp1] - v) / dy2 * (o - ocn[i, jp1])
        flux_S = H0[i, j] * (var[i, jm1] - v) / dy2 * (o - ocn[i, j])
        flux_E = ipH * (var[ip1, j] - v) / dx2 * (o - ocn[ip1, j]) - dragE
        flux_W = imH * (var[im1, j] - v) / dx2 * (o - ocn[im1, j]) - dragW
        out[i, j] = (flux_N + flux_S + flux_E + flux_W) * A_h
    end
end

# Shear-scaled (NonlinearLateralViscosity) counterparts of _laplace_U_kernel!/
# _laplace_V_kernel!: same geometry and masking, but each interior flux term gets
# its own coefficient visc_? * |Δu| in place of a single constant A_h, where |Δu|
# is the magnitude of the full velocity-difference *vector* across that face
# (the reference's dUabs), not just the component being diffused — hence `other`,
# the cross-component collocated onto this component's points.  Where
# visc_x/visc_y = C_visc * dx/100 and C_visc * dy/100 carry the reference's
# dUabs * triCw / 100 scaling (laddie_velocity.f90:260).  The grounding-line/land
# wall-drag terms keep the plain, unscaled A_h_wall — mirroring the reference,
# which never scales its border term by dUabs (laddie_velocity.f90:249-254).
#
# `other` and `Hface` (H on this component's points, masked) are read from fields
# that `laplace_U`/`laplace_V` fill first, since each is needed at several stencil
# points: forming them inline triples the kernels' CPU cost.  On the GPU the extra
# passes cost about as much as they save.
@kernel function _nonlinear_laplace_U_kernel!(
    out,
    @Const(var),
    @Const(other),
    @Const(Hface),
    @Const(H0),
    @Const(tmask),
    @Const(ocn),
    @Const(glN),
    @Const(glS),
    @Const(lndN),
    @Const(lndS),
    slip_gl,
    slip_land,
    A_h_wall,
    visc_x,
    visc_y,
    dx2,
    dy2,
    kinks,
)
    slip_gl, slip_land, A_h_wall, visc_x, visc_y, dx2, dy2 = _val(slip_gl),
    _val(slip_land), _val(A_h_wall), _val(visc_x), _val(visc_y), _val(dx2), _val(dy2)
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
        HU = Hface[i, j]
        dragN = A_h_wall * _slip(slip_gl, slip_land, glN, lndN, i, j) * HU * v / dy2
        dragS = A_h_wall * _slip(slip_gl, slip_land, glS, lndS, i, j) * HU * v / dy2
        jpH = _safe_div(HU + Hface[i, jp1], tmask[i, j] + tmask[i, jp1])
        jmH = _safe_div(HU + Hface[i, jm1], tmask[i, j] + tmask[i, jm1])
        ov = other[i, j]
        eu = _kink_width(kinks, :u)
        dN = var[i, jp1] - v
        dS = var[i, jm1] - v
        dE = var[ip1, j] - v
        dW = var[im1, j] - v
        # |Δu| across each face: this component's difference combined with the
        # cross-component's difference over the same displacement.
        aN = _ssqrt(kinks, dN * dN + (other[i, jp1] - ov)^2, eu)
        aS = _ssqrt(kinks, dS * dS + (other[i, jm1] - ov)^2, eu)
        aE = _ssqrt(kinks, dE * dE + (other[ip1, j] - ov)^2, eu)
        aW = _ssqrt(kinks, dW * dW + (other[im1, j] - ov)^2, eu)
        flux_N = visc_y * aN * jpH * dN / dy2 * (o - ocn[i, jp1]) - dragN
        flux_S = visc_y * aS * jmH * dS / dy2 * (o - ocn[i, jm1]) - dragS
        flux_E = visc_x * aE * H0[ip1, j] * dE / dx2 * (o - ocn[ip1, j])
        flux_W = visc_x * aW * H0[i, j] * dW / dx2 * (o - ocn[i, j])
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

@kernel function _nonlinear_laplace_V_kernel!(
    out,
    @Const(var),
    @Const(other),
    @Const(Hface),
    @Const(H0),
    @Const(tmask),
    @Const(ocn),
    @Const(glE),
    @Const(glW),
    @Const(lndE),
    @Const(lndW),
    slip_gl,
    slip_land,
    A_h_wall,
    visc_x,
    visc_y,
    dx2,
    dy2,
    kinks,
)
    slip_gl, slip_land, A_h_wall, visc_x, visc_y, dx2, dy2 = _val(slip_gl),
    _val(slip_land), _val(A_h_wall), _val(visc_x), _val(visc_y), _val(dx2), _val(dy2)
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
        HV = Hface[i, j]
        dragE = A_h_wall * _slip(slip_gl, slip_land, glE, lndE, i, j) * HV * v / dx2
        dragW = A_h_wall * _slip(slip_gl, slip_land, glW, lndW, i, j) * HV * v / dx2
        ipH = _safe_div(HV + Hface[ip1, j], tmask[i, j] + tmask[ip1, j])
        imH = _safe_div(HV + Hface[im1, j], tmask[i, j] + tmask[im1, j])
        ov = other[i, j]
        eu = _kink_width(kinks, :u)
        dN = var[i, jp1] - v
        dS = var[i, jm1] - v
        dE = var[ip1, j] - v
        dW = var[im1, j] - v
        # See _nonlinear_laplace_U_kernel! for the |Δu| composition.
        aN = _ssqrt(kinks, dN * dN + (other[i, jp1] - ov)^2, eu)
        aS = _ssqrt(kinks, dS * dS + (other[i, jm1] - ov)^2, eu)
        aE = _ssqrt(kinks, dE * dE + (other[ip1, j] - ov)^2, eu)
        aW = _ssqrt(kinks, dW * dW + (other[im1, j] - ov)^2, eu)
        flux_N = visc_y * aN * H0[i, jp1] * dN / dy2 * (o - ocn[i, jp1])
        flux_S = visc_y * aS * H0[i, j] * dS / dy2 * (o - ocn[i, j])
        flux_E = visc_x * aE * ipH * dE / dx2 * (o - ocn[ip1, j]) - dragE
        flux_W = visc_x * aW * imH * dW / dx2 * (o - ocn[im1, j]) - dragW
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
        m.params.kinks,
    )
    return out
end

# ---------------------------------------------------------------------------
# Donor-cell momentum advection (UpstreamMomentumAdvection)
#
# Flux form, built so the momentum fluxes ride on the *mass* fluxes of the
# thickness equation: `_face_mass_x`/`_face_mass_y` are the donor-cell `H·u` of
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
@inline function _face_mass_x(U, H, tmask, ocn, umask, i, j, Nx, kinks)
    @inbounds begin
        ip1 = _xp1(i, Nx)
        Hd = _supwind(
            kinks,
            U[i, j],
            H[i, j] * tmask[i, j] + H[ip1, j] * ocn[i, j],
            H[ip1, j] * tmask[ip1, j] + H[i, j] * ocn[ip1, j],
            _kink_width(kinks, :u),
        )
        return umask[i, j] * Hd
    end
end

# Mass flux through the y-face at v-point (i, j), between T(i, j) and T(i, j+1).
@inline function _face_mass_y(V, H, tmask, ocn, vmask, i, j, Ny, kinks)
    @inbounds begin
        jp1 = _yp1(j, Ny)
        Hd = _supwind(
            kinks,
            V[i, j],
            H[i, j] * tmask[i, j] + H[i, jp1] * ocn[i, j],
            H[i, jp1] * tmask[i, jp1] + H[i, j] * ocn[i, jp1],
            _kink_width(kinks, :u),
        )
        return vmask[i, j] * Hd
    end
end

# The neighbour's velocity on a face, for when it is upstream, with a zero-gradient
# fallback when that point is inactive (ice front); at a wall the flux is zero
# anyway.  `_supwind` picks the upstream side from the sign of the mass flux.
@inline _donor(q_own, q_nb, mask_nb) = mask_nb > zero(mask_nb) ? q_nb : q_own

@kernel function _upstream_advection_U_kernel!(
    out,
    @Const(H),
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
    kinks,
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
                _face_mass_x(U, H, tmask, ocn, umask, i, j, Nx, kinks) +
                _face_mass_x(U, H, tmask, ocn, umask, ip1, j, Nx, kinks)
            )
        MxW =
            half * (
                _face_mass_x(U, H, tmask, ocn, umask, im1, j, Nx, kinks) +
                _face_mass_x(U, H, tmask, ocn, umask, i, j, Nx, kinks)
            )
        MyN =
            half * (
                _face_mass_y(V, H, tmask, ocn, vmask, i, j, Ny, kinks) +
                _face_mass_y(V, H, tmask, ocn, vmask, ip1, j, Ny, kinks)
            )
        MyS =
            half * (
                _face_mass_y(V, H, tmask, ocn, vmask, i, jm1, Ny, kinks) +
                _face_mass_y(V, H, tmask, ocn, vmask, ip1, jm1, Ny, kinks)
            )
        eM = _kink_width(kinks, :uH)
        FE = _supwind(kinks, MxE, u, _donor(u, U[ip1, j], umask[ip1, j]), eM)
        FW = _supwind(kinks, MxW, _donor(u, U[im1, j], umask[im1, j]), u, eM)
        FN = _supwind(kinks, MyN, u, _donor(u, U[i, jp1], umask[i, jp1]), eM)
        FS = _supwind(kinks, MyS, _donor(u, U[i, jm1], umask[i, jm1]), u, eM)
        out[i, j] = -((FE - FW) / dx + (FN - FS) / dy)
    end
end

@kernel function _upstream_advection_V_kernel!(
    out,
    @Const(H),
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
    kinks,
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
                _face_mass_y(V, H, tmask, ocn, vmask, i, j, Ny, kinks) +
                _face_mass_y(V, H, tmask, ocn, vmask, i, jp1, Ny, kinks)
            )
        MyS =
            half * (
                _face_mass_y(V, H, tmask, ocn, vmask, i, jm1, Ny, kinks) +
                _face_mass_y(V, H, tmask, ocn, vmask, i, j, Ny, kinks)
            )
        MxE =
            half * (
                _face_mass_x(U, H, tmask, ocn, umask, i, j, Nx, kinks) +
                _face_mass_x(U, H, tmask, ocn, umask, i, jp1, Nx, kinks)
            )
        MxW =
            half * (
                _face_mass_x(U, H, tmask, ocn, umask, im1, j, Nx, kinks) +
                _face_mass_x(U, H, tmask, ocn, umask, im1, jp1, Nx, kinks)
            )
        eM = _kink_width(kinks, :uH)
        FE = _supwind(kinks, MxE, v, _donor(v, V[ip1, j], vmask[ip1, j]), eM)
        FW = _supwind(kinks, MxW, _donor(v, V[im1, j], vmask[im1, j]), v, eM)
        FN = _supwind(kinks, MyN, v, _donor(v, V[i, jp1], vmask[i, jp1]), eM)
        FS = _supwind(kinks, MyS, _donor(v, V[i, jm1], vmask[i, jm1]), v, eM)
        out[i, j] = -((FE - FW) / dx + (FN - FS) / dy)
    end
end

# Dispatch on `Params.momentum_advection`; the centred path is v1.x's.
momentum_advection_U(m) = _advect_U(m, m.momentum_advection)
momentum_advection_V(m) = _advect_V(m, m.momentum_advection)

function _advect_U(m, ::UpstreamMomentumAdvection)
    nx, ny = size(m.U.present)
    launch_interior!(
        _upstream_advection_U_kernel!,
        m.adv,
        m.H.present,
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
        m.params.kinks,
    )
    return m.adv
end

function _advect_V(m, ::UpstreamMomentumAdvection)
    nx, ny = size(m.V.present)
    launch_interior!(
        _upstream_advection_V_kernel!,
        m.adv,
        m.H.present,
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
        m.params.kinks,
    )
    return m.adv
end

function _advect_U(m, ::CentredMomentumAdvection)
    launch_interior!(
        _upwind_advection_U_kernel!,
        m.adv,
        m.H.present,
        m.tmask,
        m.ocn,
        m.V.present,
        m.vmask,
        m.U.present,
        m.umask,
        _u_walls(m, _advection_slips(m))...,
        m.dx,
        m.dy,
        m.params.kinks,
    )
    return m.adv
end
function _advect_V(m, ::CentredMomentumAdvection)
    launch_interior!(
        _upwind_advection_V_kernel!,
        m.adv,
        m.H.present,
        m.tmask,
        m.ocn,
        m.V.present,
        m.vmask,
        m.U.present,
        m.umask,
        _v_walls(m, _advection_slips(m))...,
        m.dx,
        m.dy,
        m.params.kinks,
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
        _u_walls(m, _wall_slips(m))...,
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
        _v_walls(m, _wall_slips(m))...,
        m.A_h,
        m.dx^2,
        m.dy^2,
    )
    return m.lap
end

# NonlinearLateralViscosity: the shear-scaled coefficient is per-face, so it must
# be applied inside the kernel rather than as a post-hoc scalar multiply; wall
# drag keeps using the plain, unscaled m.A_h (see _nonlinear_laplace_U_kernel!).
#
# The kernel reads the cross-velocity collocated onto the U points (`Hq`) and H on
# the U points (`adv`), filled here in three passes over the grid.  `adv` is free
# because `step_u_momentum!` forms this term before the advection one; `Hq` is only
# used by the tracer steps.
function laplace_U(m, lv::NonlinearLateralViscosity)
    nx, ny = size(m.V.past)
    H0 = laplacian_thickness(m)
    other, Hface = m.Hq, m.adv
    # V collocated onto the U points, so the kernel can form |Δu| across a face:
    # the same 4-point average the drag term uses for the speed magnitude, formed
    # in two halves (y into `adv`, then x into `Hq`).
    launch!(_v_half_kernel!, m.adv, m.V.past, ny)
    launch!(_other_x_kernel!, other, m.adv, nx)
    launch!(_H_on_u_kernel!, Hface, H0, m.tmask, nx)
    launch_interior!(
        _nonlinear_laplace_U_kernel!,
        m.lap,
        m.U.past,
        other,
        Hface,
        H0,
        m.tmask,
        m.ocn,
        _u_walls(m, _wall_slips(m))...,
        m.A_h,
        lv.C_visc * m.dx / 100,
        lv.C_visc * m.dy / 100,
        m.dx^2,
        m.dy^2,
        m.params.kinks,
    )
    return m.lap
end
function laplace_V(m, lv::NonlinearLateralViscosity)
    nx, ny = size(m.U.past)
    H0 = laplacian_thickness(m)
    other, Hface = m.Hq, m.adv
    # U collocated onto the V points (x half, then y); mirrors laplace_U above.
    launch!(_u_half_kernel!, m.adv, m.U.past, nx)
    launch!(_other_y_kernel!, other, m.adv, ny)
    launch!(_H_on_v_kernel!, Hface, H0, m.tmask, ny)
    launch_interior!(
        _nonlinear_laplace_V_kernel!,
        m.lap,
        m.V.past,
        other,
        Hface,
        H0,
        m.tmask,
        m.ocn,
        _v_walls(m, _wall_slips(m))...,
        m.A_h,
        lv.C_visc * m.dx / 100,
        lv.C_visc * m.dy / 100,
        m.dx^2,
        m.dy^2,
        m.params.kinks,
    )
    return m.lap
end

# 4-point collocation of the cross-velocity component, as `ip_half(jm_half(V))`
# and `jp_half(im_half(U))` without the intermediate arrays, in two passes that
# each average along one axis: in one pass the read would be wrapped along both
# axes at once, which Reactant raises to a gather.
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
@kernel function _other_x_kernel!(out, @Const(half), Nx)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = _other_x(half, i, j, Nx)
end
@kernel function _other_y_kernel!(out, @Const(half), Ny)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = _other_y(half, i, j, Ny)
end

# H on the u- (v-) points, masked by the cell the point belongs to.  Zero on the
# far border ring, where the wrapped neighbour is masked out by `tmask`.
@kernel function _H_on_u_kernel!(out, @Const(H0), @Const(tmask), Nx)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = _face_avg(H0, tmask, i, j, _xp1(i, Nx), j) * tmask[i, j]
end
@kernel function _H_on_v_kernel!(out, @Const(H0), @Const(tmask), Ny)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = _face_avg(H0, tmask, i, j, i, _yp1(j, Ny)) * tmask[i, j]
end
