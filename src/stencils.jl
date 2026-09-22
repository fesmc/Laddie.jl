
# Periodic neighbour indices along each axis.  Arrays are [ix, iy]: `_xp1`/`_xm1`
# step the first index, `_yp1`/`_ym1` the second.  These are index directions, not
# compass directions — the grid axes of a projected domain need not align with
# east/north (see the note on the shift primitives in utils.jl).  The time-step
# stencils are launched over the interior (`launch_interior!`) and use plain
# `i ± 1`; the wrap remains only where a stencil must also cover the border ring:
# the collocated cross-velocities of `NonlinearLateralViscosity` (a u-point at
# index 1 is a real face), the second-neighbour reads of the upstream momentum
# advection, and the host-side diagnostics and output averages.
@inline _xp1(i, N) = ifelse(i == N, 1, i + 1)
@inline _xm1(i, N) = ifelse(i == 1, N, i - 1)
@inline _yp1(j, N) = ifelse(j == N, 1, j + 1)
@inline _ym1(j, N) = ifelse(j == 1, N, j - 1)
@inline _safe_div(a, b) = iszero(b) ? zero(a) : a / b

@kernel function _lapT_kernel!(
    out,
    @Const(var),
    @Const(D0jp),
    @Const(D0jm),
    @Const(D0ip),
    @Const(D0im),
    @Const(tmask),
    dy2,
    dx2,
)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        jp1 = j + 1
        jm1 = j - 1
        ip1 = i + 1
        im1 = i - 1
        flux_N = D0jp[i, j] * (var[i, jp1] - var[i, j]) * tmask[i, jp1] / dy2
        flux_S = D0jm[i, j] * (var[i, jm1] - var[i, j]) * tmask[i, jm1] / dy2
        flux_E = D0ip[i, j] * (var[ip1, j] - var[i, j]) * tmask[ip1, j] / dx2
        flux_W = D0im[i, j] * (var[im1, j] - var[i, j]) * tmask[im1, j] / dx2
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

# Momentum advection of U.  D is masked on the fly (`Dt(i, j) = D·tmask`); the
# staggered velocity averages come from `precompute_advection_stencils!`.
@kernel function _upwind_advection_U_kernel!(
    out,
    @Const(D),
    @Const(tmask),
    @Const(ocn),
    @Const(Vip),
    @Const(Ujp),
    @Const(Ujm),
    @Const(Uip),
    @Const(Uim),
    @Const(U),
    @Const(glNu),
    @Const(glSu),
    @Const(lndNu),
    @Const(lndSu),
    slip_gl,
    slip_land,
    dx,
    dy,
)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        jp1 = j + 1
        jm1 = j - 1
        ip1 = i + 1
        FT = typeof(slip_gl)
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
        # Per-face slip factor, zero off the walls: gl?? and lnd?? partition the wall
        # faces (the grounding line takes precedence at mixed corners), so exactly
        # one of the two factors applies on a wall face.
        wallN = slip_gl * glNu[i, j] + slip_land * lndNu[i, j]
        wallS = slip_gl * glSu[i, j] + slip_land * lndSu[i, j]
        flux_N = -D_N * Vip[i, j] * (Ujp[i, j] - wallN * u) / dy
        flux_S = D_S * Vip[i, jm1] * (Ujm[i, j] - wallS * u) / dy
        flux_E = -D_E * Uip[i, j] * (Uip[i, j] - (one(FT) - sU) * u * ocn_e) / dx
        flux_W = D_W * Uim[i, j] * (Uim[i, j] - sU * u * ocn[i, j]) / dx
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

# Momentum advection of V; the mirror image of `_upwind_advection_U_kernel!`.
@kernel function _upwind_advection_V_kernel!(
    out,
    @Const(D),
    @Const(tmask),
    @Const(ocn),
    @Const(Vjp),
    @Const(Vjm),
    @Const(Vip),
    @Const(Vim),
    @Const(Ujp),
    @Const(V),
    @Const(glEv),
    @Const(glWv),
    @Const(lndEv),
    @Const(lndWv),
    slip_gl,
    slip_land,
    dx,
    dy,
)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        jp1 = j + 1
        ip1 = i + 1
        im1 = i - 1
        FT = typeof(slip_gl)
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
        flux_N = -D_N * Vjp[i, j] * (Vjp[i, j] - (one(FT) - sV) * v * ocn_n) / dy
        flux_S = D_S * Vjm[i, j] * (Vjm[i, j] - sV * v * ocn[i, j]) / dy
        # Per-face slip factor: see _upwind_advection_U_kernel! for the composition.
        wallE = slip_gl * glEv[i, j] + slip_land * lndEv[i, j]
        wallW = slip_gl * glWv[i, j] + slip_land * lndWv[i, j]
        flux_E = -D_E * Ujp[i, j] * (Vip[i, j] - wallE * v) / dx
        flux_W = D_W * Ujp[im1, j] * (Vim[i, j] - wallW * v) / dx
        out[i, j] = flux_N + flux_S + flux_E + flux_W
    end
end

@kernel function _laplace_U_kernel!(
    out,
    @Const(var),
    @Const(D0),
    @Const(D_on_ugrid),
    @Const(tmask),
    @Const(ocn),
    @Const(glNu),
    @Const(glSu),
    @Const(lndNu),
    @Const(lndSu),
    slip_gl,
    slip_land,
    A_h,
    dx2,
    dy2,
)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    FT = typeof(slip_gl)
    @inbounds begin
        jp1 = j + 1
        jm1 = j - 1
        ip1 = i + 1
        im1 = i - 1
        o = one(FT)
        v = var[i, j]
        # Per-face wall drag, zero off the walls (see _upwind_advection_U_kernel!
        # for how the grounding-line and land factors compose).
        dragN =
            (slip_gl * glNu[i, j] + slip_land * lndNu[i, j]) * D_on_ugrid[i, j] * v / dy2
        dragS =
            (slip_gl * glSu[i, j] + slip_land * lndSu[i, j]) * D_on_ugrid[i, j] * v / dy2
        jpD = _safe_div(D_on_ugrid[i, j] + D_on_ugrid[i, jp1], tmask[i, j] + tmask[i, jp1])
        jmD = _safe_div(D_on_ugrid[i, j] + D_on_ugrid[i, jm1], tmask[i, j] + tmask[i, jm1])
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
    @Const(D_on_vgrid),
    @Const(tmask),
    @Const(ocn),
    @Const(glEv),
    @Const(glWv),
    @Const(lndEv),
    @Const(lndWv),
    slip_gl,
    slip_land,
    A_h,
    dx2,
    dy2,
)
    FT = typeof(slip_gl)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        jp1 = j + 1
        jm1 = j - 1
        ip1 = i + 1
        im1 = i - 1
        o = one(FT)
        v = var[i, j]
        # See _laplace_U_kernel! for the slip-factor composition.
        dragE =
            (slip_gl * glEv[i, j] + slip_land * lndEv[i, j]) * D_on_vgrid[i, j] * v / dx2
        dragW =
            (slip_gl * glWv[i, j] + slip_land * lndWv[i, j]) * D_on_vgrid[i, j] * v / dx2
        ipD = _safe_div(D_on_vgrid[i, j] + D_on_vgrid[ip1, j], tmask[i, j] + tmask[ip1, j])
        imD = _safe_div(D_on_vgrid[i, j] + D_on_vgrid[im1, j], tmask[i, j] + tmask[im1, j])
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
# the cross-component collocated onto this component's points.  Where
# visc_x/visc_y = C_visc * dx/100 and C_visc * dy/100 carry the reference's
# dUabs * triCw / 100 scaling (laddie_velocity.f90:260).  The grounding-line/land
# wall-drag terms keep the plain, unscaled A_h_wall — mirroring the reference,
# which never scales its border term by dUabs (laddie_velocity.f90:249-254).
@kernel function _nonlinear_laplace_U_kernel!(
    out,
    @Const(var),
    @Const(other),
    @Const(D0),
    @Const(D_on_ugrid),
    @Const(tmask),
    @Const(ocn),
    @Const(glNu),
    @Const(glSu),
    @Const(lndNu),
    @Const(lndSu),
    slip_gl,
    slip_land,
    A_h_wall,
    visc_x,
    visc_y,
    dx2,
    dy2,
)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    FT = typeof(slip_gl)
    @inbounds begin
        jp1 = j + 1
        jm1 = j - 1
        ip1 = i + 1
        im1 = i - 1
        o = one(FT)
        v = var[i, j]
        dragN =
            A_h_wall *
            (slip_gl * glNu[i, j] + slip_land * lndNu[i, j]) *
            D_on_ugrid[i, j] *
            v / dy2
        dragS =
            A_h_wall *
            (slip_gl * glSu[i, j] + slip_land * lndSu[i, j]) *
            D_on_ugrid[i, j] *
            v / dy2
        jpD = _safe_div(D_on_ugrid[i, j] + D_on_ugrid[i, jp1], tmask[i, j] + tmask[i, jp1])
        jmD = _safe_div(D_on_ugrid[i, j] + D_on_ugrid[i, jm1], tmask[i, j] + tmask[i, jm1])
        ov = other[i, j]
        dN = var[i, jp1] - v
        dS = var[i, jm1] - v
        dE = var[ip1, j] - v
        dW = var[im1, j] - v
        # |Δu| across each face: this component's difference combined with the
        # cross-component's difference over the same displacement.
        aN = sqrt(dN * dN + (other[i, jp1] - ov)^2)
        aS = sqrt(dS * dS + (other[i, jm1] - ov)^2)
        aE = sqrt(dE * dE + (other[ip1, j] - ov)^2)
        aW = sqrt(dW * dW + (other[im1, j] - ov)^2)
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
    @Const(other),
    @Const(D0),
    @Const(D_on_vgrid),
    @Const(tmask),
    @Const(ocn),
    @Const(glEv),
    @Const(glWv),
    @Const(lndEv),
    @Const(lndWv),
    slip_gl,
    slip_land,
    A_h_wall,
    visc_x,
    visc_y,
    dx2,
    dy2,
)
    FT = typeof(slip_gl)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        jp1 = j + 1
        jm1 = j - 1
        ip1 = i + 1
        im1 = i - 1
        o = one(FT)
        v = var[i, j]
        dragE =
            A_h_wall *
            (slip_gl * glEv[i, j] + slip_land * lndEv[i, j]) *
            D_on_vgrid[i, j] *
            v / dx2
        dragW =
            A_h_wall *
            (slip_gl * glWv[i, j] + slip_land * lndWv[i, j]) *
            D_on_vgrid[i, j] *
            v / dx2
        ipD = _safe_div(D_on_vgrid[i, j] + D_on_vgrid[ip1, j], tmask[i, j] + tmask[ip1, j])
        imD = _safe_div(D_on_vgrid[i, j] + D_on_vgrid[im1, j], tmask[i, j] + tmask[im1, j])
        ov = other[i, j]
        dN = var[i, jp1] - v
        dS = var[i, jm1] - v
        dE = var[ip1, j] - v
        dW = var[im1, j] - v
        # See _nonlinear_laplace_U_kernel! for the |Δu| composition.
        aN = sqrt(dN * dN + (other[i, jp1] - ov)^2)
        aS = sqrt(dS * dS + (other[i, jm1] - ov)^2)
        aE = sqrt(dE * dE + (other[ip1, j] - ov)^2)
        aW = sqrt(dW * dW + (other[im1, j] - ov)^2)
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
# T(i+1, j).  Zero-gradient thickness when the donor side is open ocean.
@inline function _face_mass_x(U, D, tmask, ocn, umask, i, j, Nx)
    @inbounds begin
        ip1 = _xp1(i, Nx)
        u = U[i, j]
        Dd =
            u > zero(u) ? D[i, j] * tmask[i, j] + D[ip1, j] * ocn[i, j] :
            D[ip1, j] * tmask[ip1, j] + D[i, j] * ocn[ip1, j]
        return umask[i, j] * u * Dd
    end
end

# Mass flux through the y-face at v-point (i, j), between T(i, j) and T(i, j+1).
@inline function _face_mass_y(V, D, tmask, ocn, vmask, i, j, Ny)
    @inbounds begin
        jp1 = _yp1(j, Ny)
        v = V[i, j]
        Dd =
            v > zero(v) ? D[i, j] * tmask[i, j] + D[i, jp1] * ocn[i, j] :
            D[i, jp1] * tmask[i, jp1] + D[i, j] * ocn[i, jp1]
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
    slip_gl, slip_land = _advection_slips(m)
    launch_interior!(
        _upwind_advection_U_kernel!,
        m.adv,
        m.adv,
        m.D.present,
        m.tmask,
        m.ocn,
        m.Vip,
        m.Ujp,
        m.Ujm,
        m.Uip,
        m.Uim,
        m.U.present,
        m.glNu,
        m.glSu,
        m.lndNu,
        m.lndSu,
        slip_gl,
        slip_land,
        m.dx,
        m.dy,
    )
    return m.adv
end
function _advect_V(m, ::CentredMomentumAdvection)
    slip_gl, slip_land = _advection_slips(m)
    launch_interior!(
        _upwind_advection_V_kernel!,
        m.adv,
        m.adv,
        m.D.present,
        m.tmask,
        m.ocn,
        m.Vjp,
        m.Vjm,
        m.Vip,
        m.Vim,
        m.Ujp,
        m.V.present,
        m.glEv,
        m.glWv,
        m.lndEv,
        m.lndWv,
        slip_gl,
        slip_land,
        m.dx,
        m.dy,
    )
    return m.adv
end
function laplace_T(out, m, var)
    launch_interior!(
        _lapT_kernel!,
        out,
        out,
        var,
        m.D0jp,
        m.D0jm,
        m.D0ip,
        m.D0im,
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
    slip_gl, slip_land = _wall_slips(m)
    launch_interior!(
        _laplace_U_kernel!,
        m.lap,
        m.lap,
        m.U.past,
        laplacian_thickness(m),
        m.D_on_ugrid,
        m.tmask,
        m.ocn,
        m.glNu,
        m.glSu,
        m.lndNu,
        m.lndSu,
        slip_gl,
        slip_land,
        m.A_h,
        m.dx^2,
        m.dy^2,
    )
    return m.lap
end
function laplace_V(m, ::PrescribedLateralViscosity)
    slip_gl, slip_land = _wall_slips(m)
    launch_interior!(
        _laplace_V_kernel!,
        m.lap,
        m.lap,
        m.V.past,
        laplacian_thickness(m),
        m.D_on_vgrid,
        m.tmask,
        m.ocn,
        m.glEv,
        m.glWv,
        m.lndEv,
        m.lndWv,
        slip_gl,
        slip_land,
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
    slip_gl, slip_land = _wall_slips(m)
    nx, ny = size(m.V.past)
    # V collocated onto the U points, so the kernel can form |Δu| across a face.
    # Same 4-point average the drag term uses for the speed magnitude.
    launch!(_v_at_u_kernel!, m.VatU, m.VatU, m.V.past, nx, ny)
    launch_interior!(
        _nonlinear_laplace_U_kernel!,
        m.lap,
        m.lap,
        m.U.past,
        m.VatU,
        laplacian_thickness(m),
        m.D_on_ugrid,
        m.tmask,
        m.ocn,
        m.glNu,
        m.glSu,
        m.lndNu,
        m.lndSu,
        slip_gl,
        slip_land,
        m.A_h,
        lv.C_visc * m.dx / 100,
        lv.C_visc * m.dy / 100,
        m.dx^2,
        m.dy^2,
    )
    return m.lap
end
function laplace_V(m, lv::NonlinearLateralViscosity)
    slip_gl, slip_land = _wall_slips(m)
    nx, ny = size(m.U.past)
    # U collocated onto the V points; mirrors laplace_U above.
    launch!(_u_at_v_kernel!, m.UatV, m.UatV, m.U.past, nx, ny)
    launch_interior!(
        _nonlinear_laplace_V_kernel!,
        m.lap,
        m.lap,
        m.V.past,
        m.UatV,
        laplacian_thickness(m),
        m.D_on_vgrid,
        m.tmask,
        m.ocn,
        m.glEv,
        m.glWv,
        m.lndEv,
        m.lndWv,
        slip_gl,
        slip_land,
        m.A_h,
        lv.C_visc * m.dx / 100,
        lv.C_visc * m.dy / 100,
        m.dx^2,
        m.dy^2,
    )
    return m.lap
end

# 4-point collocation of the cross-velocity component, as `ip_half(jm_half(V))`
# and `jp_half(im_half(U))` without the intermediate arrays.
@kernel function _v_at_u_kernel!(out, @Const(V), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ip1 = _xp1(i, Nx)
        jm1 = _ym1(j, Ny)
        out[i, j] = ((V[i, j] + V[i, jm1]) / 2 + (V[ip1, j] + V[ip1, jm1]) / 2) / 2
    end
end
@kernel function _u_at_v_kernel!(out, @Const(U), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        im1 = _xm1(i, Nx)
        jp1 = _yp1(j, Ny)
        out[i, j] = ((U[i, j] + U[im1, j]) / 2 + (U[i, jp1] + U[im1, jp1]) / 2) / 2
    end
end

# Staggered velocity averages shared by the momentum-advection and -step kernels.
function precompute_advection_stencils!(m)
    launch_interior!(
        _precompute_staggered_kernel!,
        m.Vip,
        m.Vip,
        m.Vim,
        m.Vjp,
        m.Vjm,
        m.Uip,
        m.Uim,
        m.Ujp,
        m.Ujm,
        m.V.present,
        m.U.present,
        m.vmask,
        m.umask,
    )
    return
end

function precompute_laplacian_stencils!(m)
    launch_interior!(
        _precompute_laplacian_kernel!,
        m.D0ip,
        m.D0ip,
        m.D0im,
        m.D0jp,
        m.D0jm,
        m.D_on_ugrid,
        m.D_on_vgrid,
        laplacian_thickness(m),
        m.tmask,
    )
    return
end
