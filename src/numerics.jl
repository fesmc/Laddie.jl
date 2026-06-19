# Robert-Asselin time filter applied to one leapfrog level.
@kernel function _robert_asselin_kernel!(
    present,
    @Const(past),
    @Const(future),
    @Const(mask),
    nu,
)
    i, j = @index(Global, NTuple)
    @inbounds present[i, j] +=
        nu / 2 *
        (past[i, j] + future[i, j] - 2 * present[i, j]) *
        mask[i, j]
end

@kernel function _precompute_D_shifts_kernel!(
    Dym1,
    Dyp1,
    Dxm1,
    Dxp1,
    Dxm1ym1,
    Dxp1ym1,
    Dxm1yp1,
    @Const(D),
    @Const(tmask),
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        n = _north(i, Ny)
        s = _south(i, Ny)
        e = _east(j, Nx)
        w = _west(j, Nx)
        Dn = D[n, j] * tmask[n, j]
        Ds = D[s, j] * tmask[s, j]
        De = D[i, e] * tmask[i, e]
        Dw = D[i, w] * tmask[i, w]
        Dym1[i, j] = Dn
        Dyp1[i, j] = Ds
        Dxm1[i, j] = De
        Dxp1[i, j] = Dw
        Dxm1ym1[i, j] = D[n, e] * tmask[n, e]
        Dxp1ym1[i, j] = D[n, w] * tmask[n, w]
        Dxm1yp1[i, j] = D[s, e] * tmask[s, e]
    end
end

@kernel function _precompute_staggered_kernel!(
    Vip,
    Vim,
    Vjp,
    Vjm,
    Uip,
    Uim,
    Ujp,
    Ujm,
    signU,
    signV,
    Vyp1,
    Uxp1,
    @Const(V),
    @Const(U),
    @Const(vmask_ip),
    @Const(vmask_im),
    @Const(vmask_jp),
    @Const(vmask_jm),
    @Const(umask_ip),
    @Const(umask_im),
    @Const(umask_jp),
    @Const(umask_jm),
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        n = _north(i, Ny)
        s = _south(i, Ny)
        e = _east(j, Nx)
        w = _west(j, Nx)
        Vij = V[i, j]
        Uij = U[i, j]
        Vip[i, j] = _safe_div(Vij + V[i, e], vmask_ip[i, j])
        Vim[i, j] = _safe_div(Vij + V[i, w], vmask_im[i, j])
        Vjp[i, j] = _safe_div(Vij + V[n, j], vmask_jp[i, j])
        Vjm[i, j] = _safe_div(Vij + V[s, j], vmask_jm[i, j])
        Uip[i, j] = _safe_div(Uij + U[i, e], umask_ip[i, j])
        Uim[i, j] = _safe_div(Uij + U[i, w], umask_im[i, j])
        Ujp[i, j] = _safe_div(Uij + U[n, j], umask_jp[i, j])
        Ujm[i, j] = _safe_div(Uij + U[s, j], umask_jm[i, j])
        signU[i, j] = sign(Uij)
        signV[i, j] = sign(Vij)
        Vyp1[i, j] = V[s, j]
        Uxp1[i, j] = U[i, w]
    end
end

@kernel function _precompute_laplacian_kernel!(
    D0ip,
    D0im,
    D0jp,
    D0jm,
    @Const(D),
    @Const(tmask_ip),
    @Const(tmask_im),
    @Const(tmask_jp),
    @Const(tmask_jm),
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        n = _north(i, Ny)
        s = _south(i, Ny)
        e = _east(j, Nx)
        w = _west(j, Nx)
        Dij = D[i, j]
        D0ip[i, j] = _safe_div(Dij + D[i, e], tmask_ip[i, j])
        D0im[i, j] = _safe_div(Dij + D[i, w], tmask_im[i, j])
        D0jp[i, j] = _safe_div(Dij + D[n, j], tmask_jp[i, j])
        D0jm[i, j] = _safe_div(Dij + D[s, j], tmask_jm[i, j])
    end
end

@kernel function _clamp_kernel!(a, lo, hi)
    i, j = @index(Global, NTuple)
    @inbounds a[i, j] = clamp(a[i, j], lo, hi)
end

# Infer backend from array `A`, launch `kernel!` over the full array extent.
_workgroup(::CPU) = (8, 8)
_workgroup(::Any) = (32, 8)   # GPU: 256 threads, warp-aligned x-dimension

function launch!(kernel!, A, args...)
    backend = KA.get_backend(A)
    kernel!(backend, _workgroup(backend))(args...; ndrange = size(A))
    return nothing
end


# ==================================================================
# Time integration (Lambert et al. 2023)
# ==================================================================

_update_conv2!(::Any, ::ClampDensity) = nothing
_update_conv2!(::Any, ::ResetToAmbient) = nothing
function _update_conv2!(m, c_p::RelaxToAmbient)
    @. m.conv2 = (m.drho < 0) * m.D.present / c_p.convection_time
end

function precompute_integration_terms!(m)
    @. m.dDdt = (m.D.future - m.D.past) / (m.dt + m.dt)
    @. m.Ddrho = m.D.present * m.drho
    _update_conv2!(m, m.convection_scheme)
    return
end

@kernel function _step_thickness_kernel!(
    out,
    @Const(D0),
    @Const(convD),
    @Const(melt),
    @Const(nentr),
    @Const(tmask),
    dt,
)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = ifelse(
        iszero(tmask[i, j]),
        D0[i, j],
        D0[i, j] + (convD[i, j] + melt[i, j] + nentr[i, j]) * dt,
        # D0[i, j] + (-convD[i, j] + melt[i, j] + nentr[i, j]) * dt,
        # D0[i, j],
    )
end

@kernel function _step_u_momentum_kernel!(
    out,
    @Const(Up),
    @Const(U1),
    @Const(dDdt),
    @Const(Ddrho),
    @Const(Dxm1),
    @Const(D1),
    @Const(drho),
    @Const(dzdx),
    @Const(Vjm),
    @Const(V1),
    @Const(detr),
    @Const(cU),
    @Const(lU),
    @Const(tmask_ip),
    @Const(umask),
    g,
    f,
    C_d,
    A_h,
    dx,
    dt,
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(g)
        half = FT(1/2)
        e = _east(j, Nx)
        s = _south(i, Ny)
        tmip = tmask_ip[i, j]
        ip_dDdt = _safe_div(dDdt[i, j] + dDdt[i, e], tmip)
        ip_D_drho = _safe_div(Ddrho[i, j] + Ddrho[i, e], tmip)
        ip_D_dzdx = _safe_div(Ddrho[i, j] * dzdx[i, j] + Ddrho[i, e] * dzdx[i, e], tmip)
        ip_D = _safe_div(D1[i, j] + D1[i, e], tmip)
        ip_D_Vjm = _safe_div(D1[i, j] * Vjm[i, j] + D1[i, e] * Vjm[i, e], tmip)
        ipjmV = half * (half * (V1[i, j] + V1[s, j]) + half * (V1[i, e] + V1[s, e]))
        rhs =
            -U1[i, j] * ip_dDdt +                                      # thickness-tendency correction
            cU[i, j] +                                                  # horizontal advection
            -g * ip_D_drho * (Dxm1[i, j] - D1[i, j]) / dx +           # pressure: D gradient
            g * ip_D_dzdx +                                             # pressure: ice-shelf slope
            -half * g * ip_D^2 * (drho[i, e] - drho[i, j]) / dx +     # pressure: density gradient
            f * ip_D_Vjm +                                              # Coriolis
            -C_d * U1[i, j] * sqrt(U1[i, j]^2 + ipjmV^2) +             # quadratic drag
            A_h * lU[i, j] +                                             # horizontal viscosity
            -detr[i, j] * U1[i, j]                                     # momentum loss by detrainment
        out[i, j] = Up[i, j] + _safe_div(rhs, ip_D) * umask[i, j] * dt
    end
end

@kernel function _step_v_momentum_kernel!(
    out,
    @Const(Vp),
    @Const(V1),
    @Const(dDdt),
    @Const(Ddrho),
    @Const(Dym1),
    @Const(D1),
    @Const(drho),
    @Const(dzdy),
    @Const(Uim),
    @Const(U1),
    @Const(detr),
    @Const(cV),
    @Const(lV),
    @Const(tmask_jp),
    @Const(vmask),
    g,
    f,
    C_d,
    A_h,
    dy,
    dt,
    Ny,
    Nx,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(g)
        half = FT(0.5)
        n = _north(i, Ny)
        w = _west(j, Nx)
        tmjp = tmask_jp[i, j]
        jp_dDdt = _safe_div(dDdt[i, j] + dDdt[n, j], tmjp)
        jp_D_drho = _safe_div(Ddrho[i, j] + Ddrho[n, j], tmjp)
        jp_D_dzdy = _safe_div(Ddrho[i, j] * dzdy[i, j] + Ddrho[n, j] * dzdy[n, j], tmjp)
        jp_D = _safe_div(D1[i, j] + D1[n, j], tmjp)
        jp_D_Uim = _safe_div(D1[i, j] * Uim[i, j] + D1[n, j] * Uim[n, j], tmjp)
        jpimU = half * (half * (U1[i, j] + U1[i, w]) + half * (U1[n, j] + U1[n, w]))
        rhs =
            -V1[i, j] * jp_dDdt +                                      # thickness-tendency correction
            cV[i, j] +                                                  # horizontal advection
            -g * jp_D_drho * (Dym1[i, j] - D1[i, j]) / dy +           # pressure: D gradient
            g * jp_D_dzdy +                                             # pressure: ice-shelf slope
            -half * g * jp_D^2 * (drho[n, j] - drho[i, j]) / dy +     # pressure: density gradient
            -f * jp_D_Uim +                                             # Coriolis
            -C_d * V1[i, j] * sqrt(V1[i, j]^2 + jpimU^2) +             # quadratic drag
            A_h * lV[i, j] +                                             # horizontal viscosity
            -detr[i, j] * V1[i, j]                                     # momentum loss by detrainment
        out[i, j] = Vp[i, j] + _safe_div(rhs, jp_D) * vmask[i, j] * dt
    end
end

@kernel function _step_temperature_kernel!(
    out,
    @Const(T_past),
    @Const(T_present),
    @Const(dDdt),
    @Const(cT),
    @Const(nentr),
    @Const(Ta),
    @Const(melt),
    @Const(Tb),
    @Const(lT),
    @Const(D1),
    @Const(tmask),
    gamT,
    K_h,
    conv2,
    dt,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        rhs =
            -T_present[i, j] * dDdt[i, j] +          # thickness-tendency correction
            cT[i, j] +                                # horizontal advection
            nentr[i, j] * Ta[i, j] +                 # entrainment of ambient water at Ta
            melt[i, j] * Tb[i, j] +                  # meltwater input at freezing point
            -gamT * (T_present[i, j] - Tb[i, j]) +           # turbulent ice-ocean heat exchange
            K_h * lT[i, j] +                                   # horizontal diffusion
            -(T_past[i, j] - Ta[i, j]) * conv2                # convective restoring to ambient
        out[i, j] = T_past[i, j] + _safe_div(rhs, D1[i, j]) * tmask[i, j] * dt
    end
end

@kernel function _step_temperature_mat_gamT_kernel!(
    out,
    @Const(T_past),
    @Const(T_present),
    @Const(dDdt),
    @Const(cT),
    @Const(nentr),
    @Const(Ta),
    @Const(melt),
    @Const(Tb),
    @Const(lT),
    @Const(D1),
    @Const(tmask),
    @Const(gamT),
    K_h,
    conv2,
    dt,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        rhs =
            -T_present[i, j] * dDdt[i, j] +          # thickness-tendency correction
            cT[i, j] +                                # horizontal advection
            nentr[i, j] * Ta[i, j] +                 # entrainment of ambient water at Ta
            melt[i, j] * Tb[i, j] +                  # meltwater input at freezing point
            -gamT[i, j] * (T_present[i, j] - Tb[i, j]) +     # turbulent ice-ocean heat exchange
            K_h * lT[i, j] +                                   # horizontal diffusion
            -(T_past[i, j] - Ta[i, j]) * conv2                # convective restoring to ambient
        out[i, j] = T_past[i, j] + _safe_div(rhs, D1[i, j]) * tmask[i, j] * dt
    end
end

@kernel function _step_temperature_mat_conv2_kernel!(
    out,
    @Const(T_past),
    @Const(T_present),
    @Const(dDdt),
    @Const(cT),
    @Const(nentr),
    @Const(Ta),
    @Const(melt),
    @Const(Tb),
    @Const(lT),
    @Const(D1),
    @Const(tmask),
    gamT,
    K_h,
    @Const(conv2),
    dt,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        rhs =
            -T_present[i, j] * dDdt[i, j] +          # thickness-tendency correction
            cT[i, j] +                                # horizontal advection
            nentr[i, j] * Ta[i, j] +                 # entrainment of ambient water at Ta
            melt[i, j] * Tb[i, j] +                  # meltwater input at freezing point
            -gamT * (T_present[i, j] - Tb[i, j]) +           # turbulent ice-ocean heat exchange
            K_h * lT[i, j] +                                   # horizontal diffusion
            -(T_past[i, j] - Ta[i, j]) * conv2[i, j]          # convective restoring to ambient
        out[i, j] = T_past[i, j] + _safe_div(rhs, D1[i, j]) * tmask[i, j] * dt
    end
end

@kernel function _step_temperature_mat_both_kernel!(
    out,
    @Const(T_past),
    @Const(T_present),
    @Const(dDdt),
    @Const(cT),
    @Const(nentr),
    @Const(Ta),
    @Const(melt),
    @Const(Tb),
    @Const(lT),
    @Const(D1),
    @Const(tmask),
    @Const(gamT),
    K_h,
    @Const(conv2),
    dt,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        rhs =
            -T_present[i, j] * dDdt[i, j] +          # thickness-tendency correction
            cT[i, j] +                                # horizontal advection
            nentr[i, j] * Ta[i, j] +                 # entrainment of ambient water at Ta
            melt[i, j] * Tb[i, j] +                  # meltwater input at freezing point
            -gamT[i, j] * (T_present[i, j] - Tb[i, j]) +     # turbulent ice-ocean heat exchange
            K_h * lT[i, j] +                                   # horizontal diffusion
            -(T_past[i, j] - Ta[i, j]) * conv2[i, j]          # convective restoring to ambient
        out[i, j] = T_past[i, j] + _safe_div(rhs, D1[i, j]) * tmask[i, j] * dt
    end
end

@kernel function _step_salinity_kernel!(
    out,
    @Const(S_past),
    @Const(S_present),
    @Const(dDdt),
    @Const(cS),
    @Const(nentr),
    @Const(Sa),
    @Const(lS),
    @Const(D1),
    @Const(tmask),
    K_h,
    conv2,
    dt,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        rhs =
            -S_present[i, j] * dDdt[i, j] +              # thickness-tendency correction
            cS[i, j] +                                    # horizontal advection
            nentr[i, j] * Sa[i, j] +                     # entrainment of ambient water at Sa
            K_h * lS[i, j] +                              # horizontal diffusion
            -(S_past[i, j] - Sa[i, j]) * conv2           # convective restoring to ambient
        out[i, j] = S_past[i, j] + _safe_div(rhs, D1[i, j]) * tmask[i, j] * dt
    end
end

@kernel function _step_salinity_mat_conv2_kernel!(
    out,
    @Const(S_past),
    @Const(S_present),
    @Const(dDdt),
    @Const(cS),
    @Const(nentr),
    @Const(Sa),
    @Const(lS),
    @Const(D1),
    @Const(tmask),
    K_h,
    @Const(conv2),
    dt,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        rhs =
            -S_present[i, j] * dDdt[i, j] +              # thickness-tendency correction
            cS[i, j] +                                    # horizontal advection
            nentr[i, j] * Sa[i, j] +                     # entrainment of ambient water at Sa
            K_h * lS[i, j] +                              # horizontal diffusion
            -(S_past[i, j] - Sa[i, j]) * conv2[i, j]    # convective restoring to ambient
        out[i, j] = S_past[i, j] + _safe_div(rhs, D1[i, j]) * tmask[i, j] * dt
    end
end

function step_thickness(m, dt)
    launch!(
        _step_thickness_kernel!,
        m.D.future,
        m.D.future,
        m.D.past,
        m.convD,
        m.melt,
        m.nentr,
        m.tmask,
        dt,
    )
    return
end
function step_u_momentum(m, dt)
    upwind_advection_U(m)
    laplace_U(m)
    ny, nx = size(m.U.future)
    launch!(
        _step_u_momentum_kernel!,
        m.U.future,
        m.U.future,
        m.U.past,
        m.U.present,
        m.dDdt,
        m.Ddrho,
        m.Dxm1,
        m.D.present,
        m.drho,
        m.dzdx,
        m.Vjm,
        m.V.present,
        m.detr,
        m.cU,
        m.lU,
        m.tmask_ip,
        m.umask,
        m.g,
        m.f,
        m.C_d,
        m.A_h,
        m.dx,
        dt,
        ny,
        nx,
    )
    return
end
function step_v_momentum(m, dt)
    upwind_advection_V(m)
    laplace_V(m)
    ny, nx = size(m.V.future)
    launch!(
        _step_v_momentum_kernel!,
        m.V.future,
        m.V.future,
        m.V.past,
        m.V.present,
        m.dDdt,
        m.Ddrho,
        m.Dym1,
        m.D.present,
        m.drho,
        m.dzdy,
        m.Uim,
        m.U.present,
        m.detr,
        m.cV,
        m.lV,
        m.tmask_jp,
        m.vmask,
        m.g,
        m.f,
        m.C_d,
        m.A_h,
        m.dy,
        dt,
        ny,
        nx,
    )
    return
end
# Kernel selection dispatches on the concrete types of gamT and conv2, which are
# fixed at model-construction time by the MP and CS type parameters.
_launch_T!(args, gamT::Number, K_h, conv2::Number, dt) =
    launch!(_step_temperature_kernel!, args..., gamT, K_h, conv2, dt)
_launch_T!(args, gamT::Number, K_h, conv2::AbstractArray, dt) =
    launch!(_step_temperature_mat_conv2_kernel!, args..., gamT, K_h, conv2, dt)
_launch_T!(args, gamT::AbstractArray, K_h, conv2::Number, dt) =
    launch!(_step_temperature_mat_gamT_kernel!, args..., gamT, K_h, conv2, dt)
_launch_T!(args, gamT::AbstractArray, K_h, conv2::AbstractArray, dt) =
    launch!(_step_temperature_mat_both_kernel!, args..., gamT, K_h, conv2, dt)

_launch_S!(args, K_h, conv2::Number, dt) =
    launch!(_step_salinity_kernel!, args..., K_h, conv2, dt)
_launch_S!(args, K_h, conv2::AbstractArray, dt) =
    launch!(_step_salinity_mat_conv2_kernel!, args..., K_h, conv2, dt)

function step_temperature(m, dt)
    @. m.DT = m.D.present * m.T.present
    upwind_advection_T(m.cT, m, m.DT)
    laplace_T(m.lT, m, m.T.past)
    args = (
        m.T.future,
        m.T.future,
        m.T.past,
        m.T.present,
        m.dDdt,
        m.cT,
        m.nentr,
        m.Ta,
        m.melt,
        m.Tb,
        m.lT,
        m.D.present,
        m.tmask,
    )
    _launch_T!(args, m.gamT, m.K_h, m.conv2, dt)
    return
end
function step_salinity(m, dt)
    @. m.DS = m.D.present * m.S.present
    upwind_advection_T(m.cS, m, m.DS)
    laplace_T(m.lS, m, m.S.past)
    args = (
        m.S.future,
        m.S.future,
        m.S.past,
        m.S.present,
        m.dDdt,
        m.cS,
        m.nentr,
        m.Sa,
        m.lS,
        m.D.present,
        m.tmask,
    )
    _launch_S!(args, m.K_h, m.conv2, dt)
    return
end


function _clamp_thickness!(m)
    max_layer_thickness!(m, m.params.max_layer_thickness)
    @. m.D.future = max(m.D.future, m.D_min)
    return
end

function _check_nans_shelf!(m, varname, arr)
    any(isnan.(arr) .& (m.tmask .> 0)) &&
        error("NaN in $varname at t = $(round(m.t / m.seconds_per_day, digits=4)) days (step $(m.count))")
end

function leapfrog_step!(m, nsteps)
    dt = nsteps * m.dt
    dbg = m.config.dbg
    step_thickness(m, dt)
    _clamp_thickness!(m)
    precompute_integration_terms!(m)
    dbg.check_nans && _check_nans_shelf!(m, "D", m.D.future)
    
    step_u_momentum(m, dt)
    @. m.U.future = clamp(m.U.future, -m.v_cut, m.v_cut)
    dbg.check_nans && _check_nans_shelf!(m, "U", m.U.future)

    step_v_momentum(m, dt)
    @. m.V.future = clamp(m.V.future, -m.v_cut, m.v_cut)
    dbg.check_nans && _check_nans_shelf!(m, "V", m.V.future)

    step_temperature(m, dt)
    @. m.T.future = clamp(m.T.future, -5, 5)
    dbg.check_nans && _check_nans_shelf!(m, "T", m.T.future)

    step_salinity(m, dt)
    @. m.S.future = clamp(m.S.future, 32, 36)
    dbg.check_nans && _check_nans_shelf!(m, "S", m.S.future)
    return
end

# Re-initialise the leapfrog after a dt change: collapse the `past` level onto
# `present` so the two are co-located in time, refresh secondary fields, then
# take one first-order step at the new dt.  Structurally identical to the
# bootstrap that ends `_initialize_prognostics!`/`init_from_restart!`, so the
# next `advance_leapfrog!` rotation leaves a past/present pair separated by the
# new dt and the following centred `leapfrog_step!(m, 2)` is consistent.  The
# anchor is the Robert–Asselin-filtered `present`, exactly as at startup.
function _rebootstrap_leapfrog!(m)
    for var in (m.D, m.U, m.V, m.T, m.S)
        var.past .= var.present
    end
    update_secondary_fields!(m)
    leapfrog_step!(m, 1)
    return
end

# ============================================================================
# Time-stepping orchestration
# ============================================================================

function apply_robert_asselin_filter!(m)
    for (var, mask) in
        ((m.D, m.tmask), (m.U, m.umask), (m.V, m.vmask), (m.T, m.tmask), (m.S, m.tmask))
        launch!(
            _robert_asselin_kernel!,
            var.present,
            var.present,
            var.past,
            var.future,
            mask,
            m.nu,
        )
    end
    update_density!(m)
    update_convection!(m)
    return
end

function advance_leapfrog!(m)
    for var in (m.D, m.U, m.V, m.T, m.S)
        
        rotate!(var)
    end
    update_secondary_fields!(m)
    return
end

function clamp_velocities!(m)
    launch!(_clamp_kernel!, m.U.future, m.U.future, -m.v_cut, m.v_cut)
    launch!(_clamp_kernel!, m.V.future, m.V.future, -m.v_cut, m.v_cut)
    return
end
