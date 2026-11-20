# Robert-Asselin time filter applied to one leapfrog level.
@kernel function _robert_asselin_kernel!(
    present,
    @Const(past),
    @Const(future),
    @Const(mask),
    nu,
)
    nu = _val(nu)
    i, j = @index(Global, NTuple)
    @inbounds present[i, j] +=
        nu / 2 * (past[i, j] + future[i, j] - 2 * present[i, j]) * mask[i, j]
end

# ==================================================================
# Time integration (Lambert et al. 2023)
# ==================================================================

"""
    convective_relaxation!(m, convection)

Set the relaxation rate `m.conv2 = [δρ < 0] H/convection_time` on ice-covered cells under
[`RelaxToAmbientConvection`](@ref); nothing for the other schemes.
"""
function convective_relaxation! end

convective_relaxation!(::Any, ::AbstractConvection) = nothing
function convective_relaxation!(m, cs::RelaxToAmbientConvection)
    # `imask`, not `tmask`: gap cells are never relaxed towards ambient.  See
    # `convection!(m, ::RelaxToAmbientConvection)`.
    launch!(
        _relax_conv2_kernel!,
        m.conv2,
        m.drho,
        m.imask,
        m.H.present,
        cs.convection_time,
    )
end

@kernel function _relax_conv2_kernel!(
    conv2,
    @Const(drho),
    @Const(imask),
    @Const(H),
    convection_time,
)
    convection_time = _val(convection_time)
    i, j = @index(Global, NTuple)
    @inbounds conv2[i, j] = (drho[i, j] < 0) * imask[i, j] * H[i, j] / convection_time
end

# The per-step elementwise updates below are kernels rather than broadcasts: on
# the CPU a broadcast runs on the main thread only, both costing its own time
# serially and pulling the arrays the threaded kernels just wrote across cores.
@kernel function _integration_terms_kernel!(
    dHdt,
    Hdrho,
    @Const(H_future),
    @Const(H_past),
    @Const(H),
    @Const(drho),
    span,
)
    span = _val(span)
    i, j = @index(Global, NTuple)
    @inbounds begin
        dHdt[i, j] = (H_future[i, j] - H_past[i, j]) / span
        Hdrho[i, j] = H[i, j] * drho[i, j]
    end
end

# `H` is the thickness of the pressure and Coriolis terms (`H.present` in the leapfrog,
# the blend `Hfb` in FB-RK3), and dH/dt is taken over `span`.  The leapfrog passes
# 2·dt, as LADDIE v1 (`prepare_integrate`), also in the bootstrap step, whose levels are
# only dt apart (devnotes/open-questions/restart-step-sizing.md).
function precompute_integration_terms!(m, H, span)
    launch!(
        _integration_terms_kernel!,
        m.dHdt,
        m.Hdrho,
        m.H.future,
        m.H.past,
        H,
        m.drho,
        span,
    )
    convective_relaxation!(m, m.convection)
    return
end

# `LeakyGapsBC`: the drag coefficient of a face, scaled by the mean mélange cover of its
# two cells (`gap_drag`: α in gap cells, 1 elsewhere).  Under any other gap condition
# `gap_drag` is a scalar and C_d passes through; off gaps `C_d·2/2` is C_d exactly.
@inline _face_drag(C_d, ::Number, i, j, i2, j2) = C_d
@inline _face_drag(C_d, f::AbstractArray, i, j, i2, j2) = @inbounds C_d * (f[i, j] + f[i2, j2]) / 2

@kernel function _step_thickness_kernel!(
    out,
    @Const(H0),
    @Const(convH),
    @Const(melt),
    @Const(nentr),
    @Const(detr_cap),
    @Const(tmask),
    frazil,
    dt,
)
    dt, frazil = _val(dt), _val(frazil)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = ifelse(
        iszero(tmask[i, j]),
        H0[i, j],
        _unclamped_thickness(H0, convH, melt, nentr, detr_cap, frazil, i, j, dt),
    )
end

# The thickness the thickness step forms before `_clamp_thickness!`, the `H*` of the
# flux forms (see `FluxTracerForm`, `FluxMomentumForm`).
@inline _unclamped_thickness(H_past, convH, melt, nentr, detr_cap, frazil, i, j, dt) =
    @inbounds H_past[i, j] + (convH[i, j] + melt[i, j] + nentr[i, j] - detr_cap[i, j] - _at(frazil, i, j)) * dt

"""
    momentum_thickness_tendency(form, U, dHdt)

The `-U ∂H/∂t` term of [`AdvectiveMomentumForm`](@ref); zero under [`FluxMomentumForm`](@ref).
"""
function momentum_thickness_tendency end

# The two `momentum_form`s (see `FluxMomentumForm`).  The flux form steps the
# transport H̄U: `-U ∂H/∂t` drops out, and the sinks at the layer's own U (detrainment,
# cap detrainment, frazil, drag) move from the rhs to the denominator, at the new level.
@inline momentum_thickness_tendency(::AdvectiveMomentumForm, U, dHdt) = -U * dHdt
@inline momentum_thickness_tendency(::FluxMomentumForm, U, dHdt) = zero(U)

"""
    explicit_sink(form, x)

The part of a momentum sink `x` kept in the explicit right-hand side: all of it under
[`AdvectiveMomentumForm`](@ref), none under [`FluxMomentumForm`](@ref), which takes it at the
new level.
"""
function explicit_sink end

@inline explicit_sink(::AdvectiveMomentumForm, x) = x
@inline explicit_sink(::FluxMomentumForm, x) = zero(x)

"""
    momentum_flux_terms(form, H_past, H_future, convH, melt, nentr, detr, detr_cap, frazil, tmask, i, j, i2, j2, tm, dt) -> (H_old, H_new, sink)

What [`momentum_update`](@ref) needs at the velocity point between cells `(i, j)` and
`(i2, j2)`: the face thickness of the past and of the new level, and the sink rate (m s⁻¹) taken
at the new level. Zeros under [`AdvectiveMomentumForm`](@ref).
"""
function momentum_flux_terms end

@inline function momentum_flux_terms(::AdvectiveMomentumForm, H_past, H_future, convH, melt, nentr, detr,
                                      detr_cap, frazil, tmask, i, j, i2, j2, tm, dt)
    z = zero(dt)
    return z, z, z
end
@inline function momentum_flux_terms(::FluxMomentumForm, H_past, H_future, convH, melt, nentr, detr,
                                      detr_cap, frazil, tmask, i, j, i2, j2, tm, dt)
    @inbounds begin
        H_old = _safe_div(H_past[i, j] + H_past[i2, j2], tm)
        H_new = _safe_div(
            _new_thickness(H_past, H_future, convH, melt, nentr, detr_cap, frazil, tmask, i, j, dt) +
            _new_thickness(H_past, H_future, convH, melt, nentr, detr_cap, frazil, tmask, i2, j2, dt),
            tm,
        )
        sink = _safe_div(
            detr[i, j] + detr[i2, j2] + detr_cap[i, j] + detr_cap[i2, j2] + _at(frazil, i, j) + _at(frazil, i2, j2),
            tm,
        )
        return H_old, H_new, sink
    end
end
@inline _new_thickness(H_past, H_future, convH, melt, nentr, detr_cap, frazil, tmask, i, j, dt) =
    @inbounds max(H_future[i, j], _unclamped_thickness(H_past, convH, melt, nentr, detr_cap, frazil, i, j, dt)) *
              tmask[i, j]

"""
    momentum_update(form, U_past, U_present, rhs, H1, H_old, H_new, sink, outflow, mask, dt)

The new velocity at a velocity point under the [`AbstractMomentumForm`](@ref) `form`:
`U⁻ + dt · rhs/H̄` for the advective form, `(H̄⁻U⁻ + dt · rhs)/(H̄⁺ + dt · s)` for the flux
form (see [`FluxMomentumForm`](@ref)).
"""
function momentum_update end

# `sink` is the rate (m s⁻¹) at which the flux form removes momentum at the new level.
# `outflow` is the advective part of it, the rate at which the point's own velocity
# leaves through the faces of its control volume (upstream advection only, else
# zero): `rhs` carries it at the present level, `- outflow·U_present`, so it is added
# back and moved into the denominator, as `tracer_update` does with the outflow.
@inline momentum_update(::AdvectiveMomentumForm, U_past, U_present, rhs, H1, H_old, H_new, sink, outflow, mask, dt) =
    U_past + _safe_div(rhs, H1) * mask * dt
@inline momentum_update(::FluxMomentumForm, U_past, U_present, rhs, H1, H_old, H_new, sink, outflow, mask, dt) =
    ifelse(mask > 0, _safe_div(H_old * U_past + (rhs + outflow * U_present) * dt, H_new + (sink + outflow) * dt), U_past)

"""
    momentum_outflow(m, form, advection)

The buffer the momentum advection writes its outflow rate into (`m.Hq`, free once the
Laplacian has used it), or `nothing`: only [`FluxMomentumForm`](@ref) with
[`UpstreamMomentumAdvection`](@ref) takes the outflow at the new level.
"""
function momentum_outflow end

momentum_outflow(m) = momentum_outflow(m, m.params.momentum_form, m.momentum_advection)
momentum_outflow(m, ::FluxMomentumForm, ::UpstreamMomentumAdvection) = m.Hq
momentum_outflow(m, ::AbstractMomentumForm, ::AbstractMomentumAdvection) = nothing
_or_zero(x, FT) = x
_or_zero(::Nothing, FT) = zero(FT)

@kernel function _step_u_momentum_kernel!(
    out,
    @Const(Up),
    @Const(U1),
    @Const(dHdt),
    @Const(Hdrho),
    @Const(H1),
    @Const(drho),
    @Const(dzdx),
    @Const(V1),
    @Const(detr),
    @Const(detr_cap),
    @Const(cU),
    @Const(lU),
    @Const(tmask),
    @Const(umask),
    @Const(vmask),
    @Const(fu),
    @Const(Ud),
    @Const(Vd),
    @Const(H_past),
    @Const(H_future),
    @Const(convH),
    @Const(melt),
    @Const(nentr),
    outflow,
    frazil,
    gap_drag,
    g,
    C_d,
    pgf_w,
    drho_w,
    dx,
    dt,
    kinks,
    momentum_form,
)
    g, C_d, pgf_w, drho_w, dx, dt = _val(g), _val(C_d), _val(pgf_w), _val(drho_w), _val(dx), _val(dt)
    frazil = _val(frazil)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        FT = typeof(g)
        half = FT(0.5)
        ip1 = i + 1
        jm1 = j - 1
        tmip = tmask[i, j] + tmask[ip1, j]
        ip_dHdt = _safe_div(dHdt[i, j] + dHdt[ip1, j], tmip)
        ip_detr_cap = _safe_div(detr_cap[i, j] + detr_cap[ip1, j], tmip)
        ip_frazil = _safe_div(_at(frazil, i, j) + _at(frazil, ip1, j), tmip)
        ip_H_drho = _safe_div(Hdrho[i, j] + Hdrho[ip1, j], tmip)
        ip_H_dzdx = _safe_div(Hdrho[i, j] * dzdx[i, j] + Hdrho[ip1, j] * dzdx[ip1, j], tmip)
        ip_H = _safe_div(H1[i, j] + H1[ip1, j], tmip)
        Vjm = _face_avg(V1, vmask, i, j, i, jm1)
        Vjm_ip = _face_avg_ring0(V1, vmask, ip1, j, 0, -1)
        ip_H_Vjm = _safe_div(H1[i, j] * Vjm + H1[ip1, j] * Vjm_ip, tmip)
        # The drag speed, at the level of `Params.closure_level` (Ud, Vd).
        ipjmV = half * (half * (Vd[i, j] + Vd[i, jm1]) + half * (Vd[ip1, j] + Vd[ip1, jm1]))
        speed = _ssqrt(kinks, Ud[i, j]^2 + ipjmV^2, kink_width(kinks, :u))
        C_dU = _face_drag(C_d, gap_drag, i, j, ip1, j)   # mélange cover in gaps (LeakyGapsBC)
        # tmip is 1 at a one-sided face (ice front, gap-sink edge), 2 elsewhere.  The
        # weights pgf_w and drho_w of the `AbstractFrontPressure` drop the H-gradient and
        # density-gradient terms there (weight 1) or keep them (0, an exact multiply by 1).
        pgf_gate = one(FT) + pgf_w * (tmip - FT(2))
        drho_gate = one(FT) + drho_w * (tmip - FT(2))
        H_next = H1[ip1, j] * tmask[ip1, j]
        f = momentum_form
        rhs =
            momentum_thickness_tendency(f, U1[i, j], ip_dHdt) +       # thickness-tendency correction
            cU[i, j] +                                                  # horizontal advection
            -g * ip_H_drho * (H_next - H1[i, j]) / dx * pgf_gate +     # pressure: H gradient
            g * ip_H_dzdx +                                             # pressure: ice-shelf slope
            -half * g * ip_H^2 * (drho[ip1, j] - drho[i, j]) / dx * drho_gate + # pressure: density gradient
            fu[i, j] * ip_H_Vjm +                                              # Coriolis
            explicit_sink(f, -C_dU * U1[i, j] * speed) +              # quadratic drag
            lU[i, j] +                                                   # horizontal viscosity
            explicit_sink(f, -detr[i, j] * U1[i, j]) +                # momentum loss by detrainment
            explicit_sink(f, -(ip_detr_cap + ip_frazil) * U1[i, j])   # cap detrainment and frazil, cancel their share of dHdt
        H_old, H_new, sink = momentum_flux_terms(
            f, H_past, H_future, convH, melt, nentr, detr, detr_cap, frazil, tmask, i, j, ip1, j, tmip, dt,
        )
        U_new = momentum_update(
            f, Up[i, j], U1[i, j], rhs, ip_H, H_old, H_new, sink + C_dU * speed, _at(outflow, i, j), umask[i, j], dt,
        )
        out[i, j] = U_new
    end
end

@kernel function _step_v_momentum_kernel!(
    out,
    @Const(Vp),
    @Const(V1),
    @Const(dHdt),
    @Const(Hdrho),
    @Const(H1),
    @Const(drho),
    @Const(dzdy),
    @Const(U1),
    @Const(detr),
    @Const(detr_cap),
    @Const(cV),
    @Const(lV),
    @Const(tmask),
    @Const(vmask),
    @Const(umask),
    @Const(fv),
    @Const(Vd),
    @Const(Ud),
    @Const(H_past),
    @Const(H_future),
    @Const(convH),
    @Const(melt),
    @Const(nentr),
    outflow,
    frazil,
    gap_drag,
    g,
    C_d,
    pgf_w,
    drho_w,
    dy,
    dt,
    kinks,
    momentum_form,
)
    g, C_d, pgf_w, drho_w, dy, dt = _val(g), _val(C_d), _val(pgf_w), _val(drho_w), _val(dy), _val(dt)
    frazil = _val(frazil)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        FT = typeof(g)
        half = FT(0.5)
        jp1 = j + 1
        im1 = i - 1
        tmjp = tmask[i, j] + tmask[i, jp1]
        jp_dHdt = _safe_div(dHdt[i, j] + dHdt[i, jp1], tmjp)
        jp_detr_cap = _safe_div(detr_cap[i, j] + detr_cap[i, jp1], tmjp)
        jp_frazil = _safe_div(_at(frazil, i, j) + _at(frazil, i, jp1), tmjp)
        jp_H_drho = _safe_div(Hdrho[i, j] + Hdrho[i, jp1], tmjp)
        jp_H_dzdy = _safe_div(Hdrho[i, j] * dzdy[i, j] + Hdrho[i, jp1] * dzdy[i, jp1], tmjp)
        jp_H = _safe_div(H1[i, j] + H1[i, jp1], tmjp)
        Uim = _face_avg(U1, umask, i, j, im1, j)
        Uim_jp = _face_avg_ring0(U1, umask, i, jp1, -1, 0)
        jp_H_Uim = _safe_div(H1[i, j] * Uim + H1[i, jp1] * Uim_jp, tmjp)
        # The drag speed, at the level of `Params.closure_level` (Vd, Ud).
        jpimU = half * (half * (Ud[i, j] + Ud[im1, j]) + half * (Ud[i, jp1] + Ud[im1, jp1]))
        speed = _ssqrt(kinks, Vd[i, j]^2 + jpimU^2, kink_width(kinks, :u))
        C_dV = _face_drag(C_d, gap_drag, i, j, i, jp1)   # mélange cover in gaps (LeakyGapsBC)
        # See _step_u_momentum_kernel! for the ice-front gates.
        pgf_gate = one(FT) + pgf_w * (tmjp - FT(2))
        drho_gate = one(FT) + drho_w * (tmjp - FT(2))
        H_next = H1[i, jp1] * tmask[i, jp1]
        f = momentum_form
        rhs =
            momentum_thickness_tendency(f, V1[i, j], jp_dHdt) +       # thickness-tendency correction
            cV[i, j] +                                                  # horizontal advection
            -g * jp_H_drho * (H_next - H1[i, j]) / dy * pgf_gate +     # pressure: H gradient
            g * jp_H_dzdy +                                             # pressure: ice-shelf slope
            -half * g * jp_H^2 * (drho[i, jp1] - drho[i, j]) / dy * drho_gate + # pressure: density gradient
            -fv[i, j] * jp_H_Uim +                                             # Coriolis
            explicit_sink(f, -C_dV * V1[i, j] * speed) +              # quadratic drag
            lV[i, j] +                                                   # horizontal viscosity
            explicit_sink(f, -detr[i, j] * V1[i, j]) +                # momentum loss by detrainment
            explicit_sink(f, -(jp_detr_cap + jp_frazil) * V1[i, j])   # cap detrainment and frazil, cancel their share of dHdt
        H_old, H_new, sink = momentum_flux_terms(
            f, H_past, H_future, convH, melt, nentr, detr, detr_cap, frazil, tmask, i, j, i, jp1, tmjp, dt,
        )
        V_new = momentum_update(
            f, Vp[i, j], V1[i, j], rhs, jp_H, H_old, H_new, sink + C_dV * speed, _at(outflow, i, j), vmask[i, j], dt,
        )
        out[i, j] = V_new
    end
end

# A kernel argument that is either a per-cell field or one value for the whole
# domain (gamT and conv2 are scalars or matrices depending on the melt and
# convection schemes); dispatch picks the right read at compile time.
@inline _at(x::Number, i, j) = x
@inline _at(x::AbstractArray, i, j) = @inbounds x[i, j]

"""
    thickness_tendency(form, q, dHdt)

The `-q ∂H/∂t` term of [`AdvectiveTracerForm`](@ref); zero under [`FluxTracerForm`](@ref).
"""
function thickness_tendency end

# The two `tracer_form`s (see `FluxTracerForm`, devnotes/improvements/flux-form-tracers.md).
# Masked cells keep their past value under both.  The flux form takes the terms that
# remove layer water at its own q (face outflow, cap detrainment, for T frazil), the
# rate `sink` (m s⁻¹), at the new level: rhs carries them as `-sink·q_present`, so they
# are added back to the content and moved into the denominator.  q⁺ is then a weighted
# mean of q⁻ and the waters coming in, with nonnegative weights at any step.
@inline thickness_tendency(::AdvectiveTracerForm, q, dHdt) = -q * dHdt
@inline thickness_tendency(::FluxTracerForm, q, dHdt) = zero(q)

"""
    tracer_update(form, q_past, q_present, rhs, sink, H1, H_past, H_future, H_star, qa, tmask, dt)

The new tracer value of a cell under the [`AbstractTracerForm`](@ref) `form`:
`q⁻ + dt · rhs/H` for the advective form; for the flux form the content `H⁻q⁻ + dt · rhs`, with
the `sink` at the new level and the water added by the `H_min` floor at `qa`, over the new
thickness (see [`FluxTracerForm`](@ref)).
"""
function tracer_update end

@inline tracer_update(::AdvectiveTracerForm, q_past, q_present, rhs, sink, H1, H_past, H_future, H_star, qa, tmask, dt) =
    q_past + _safe_div(rhs, H1) * tmask * dt
@inline function tracer_update(::FluxTracerForm, q_past, q_present, rhs, sink, H1, H_past, H_future, H_star, qa, tmask, dt)
    content = H_past * q_past + (rhs + sink * q_present) * dt
    added = max(H_future - H_star, zero(H_star))   # by the floor, at qa
    return ifelse(
        tmask > 0,
        _safe_div(content + added * qa, max(H_future, H_star) + sink * dt),
        q_past,
    )
end

"""
    outflow_rate!(m, form)

Write the outflow rate of each cell (m s⁻¹), the volume leaving through its faces per unit
area, into `m.Hq` for [`FluxTracerForm`](@ref) (see [`tracer_update`](@ref)); nothing under
[`AdvectiveTracerForm`](@ref).
"""
function outflow_rate! end

outflow_rate!(m, ::AdvectiveTracerForm) = nothing
function outflow_rate!(m, ::FluxTracerForm)
    launch_interior!(
        _outflow_rate_kernel!,
        m.Hq,
        m.H.present,
        m.U.present,
        m.V.present,
        m.umask,
        m.vmask,
        m.dx,
        m.dy,
        m.params.kinks,
    )
    return
end

@kernel function _outflow_rate_kernel!(out, @Const(H), @Const(U), @Const(V), @Const(umask), @Const(vmask), dx, dy, kinks)
    dx, dy = _val(dx), _val(dy)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        eu = kink_width(kinks, :u)
        out[i, j] =
            H[i, j] * (
                (_smax0(kinks, V[i, j], eu) * vmask[i, j] - _sneg(kinks, V[i, j-1], eu) * vmask[i, j-1]) / dy +
                (_smax0(kinks, U[i, j], eu) * umask[i, j] - _sneg(kinks, U[i-1, j], eu) * umask[i-1, j]) / dx
            )
    end
end

@kernel function _step_temperature_kernel!(
    out,
    @Const(T_past),
    @Const(T_present),
    @Const(dHdt),
    @Const(cT),
    @Const(nentr),
    @Const(detr_cap),
    @Const(Ta),
    @Const(melt),
    @Const(Tb),
    @Const(lT),
    @Const(H1),
    @Const(H_past),
    @Const(H_future),
    @Const(convH),
    @Const(outflow),
    @Const(tmask),
    tracer_form,
    gamT,
    K_h,
    conv2,
    frazil,
    L,
    c_p,
    dt,
)
    gamT, K_h, conv2, dt = _val(gamT), _val(K_h), _val(conv2), _val(dt)
    frazil, L, c_p = _val(frazil), _val(L), _val(c_p)
    i, j = @index(Global, NTuple)
    @inbounds begin
        rhs =
            thickness_tendency(tracer_form, T_present[i, j], dHdt[i, j]) + # thickness-tendency correction
            cT[i, j] +                                      # horizontal advection
            nentr[i, j] * Ta[i, j] +                        # entrainment of ambient water at Ta
            -detr_cap[i, j] * T_present[i, j] +             # cap detrainment of layer water at T
            melt[i, j] * Tb[i, j] +                         # meltwater input at freezing point
            -_at(gamT, i, j) * (T_present[i, j] - Tb[i, j]) + # turbulent ice-ocean heat exchange
            K_h * lT[i, j] +                                # horizontal diffusion
            -(T_past[i, j] - Ta[i, j]) * _at(conv2, i, j) + # convective restoring to ambient
            _at(frazil, i, j) * (L / c_p - T_present[i, j]) # frazil: latent heat in, water out at T
        # The unclamped thickness, exactly as `_step_thickness_kernel!` forms it.
        H_star = _unclamped_thickness(H_past, convH, melt, nentr, detr_cap, frazil, i, j, dt)
        # Frazil removes layer water at T, so it is part of the sink.
        T_new = tracer_update(
            tracer_form, T_past[i, j], T_present[i, j], rhs,
            outflow[i, j] + detr_cap[i, j] + _at(frazil, i, j),
            H1[i, j], H_past[i, j], H_future[i, j], H_star, Ta[i, j], tmask[i, j], dt,
        )
        out[i, j] = T_new
    end
end

@kernel function _step_salinity_kernel!(
    out,
    @Const(S_past),
    @Const(S_present),
    @Const(dHdt),
    @Const(cS),
    @Const(nentr),
    @Const(detr_cap),
    @Const(Sa),
    @Const(lS),
    @Const(H1),
    @Const(H_past),
    @Const(H_future),
    @Const(convH),
    @Const(melt),
    @Const(outflow),
    @Const(tmask),
    tracer_form,
    K_h,
    conv2,
    frazil,
    dt,
)
    K_h, conv2, frazil, dt = _val(K_h), _val(conv2), _val(frazil), _val(dt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        rhs =
            thickness_tendency(tracer_form, S_present[i, j], dHdt[i, j]) + # thickness-tendency correction
            cS[i, j] +                                      # horizontal advection
            nentr[i, j] * Sa[i, j] +                        # entrainment of ambient water at Sa
            -detr_cap[i, j] * S_present[i, j] +             # cap detrainment of layer water at S
            K_h * lS[i, j] +                                # horizontal diffusion
            -(S_past[i, j] - Sa[i, j]) * _at(conv2, i, j)   # convective restoring to ambient
        # No frazil term, and no frazil in the sink: frazil leaves its salt behind,
        # which the thinner layer then holds (brine rejection).
        H_star = _unclamped_thickness(H_past, convH, melt, nentr, detr_cap, frazil, i, j, dt)
        out[i, j] = tracer_update(
            tracer_form, S_past[i, j], S_present[i, j], rhs, outflow[i, j] + detr_cap[i, j],
            H1[i, j], H_past[i, j], H_future[i, j], H_star, Sa[i, j], tmask[i, j], dt,
        )
    end
end

function step_thickness!(m, dt)
    launch!(
        _step_thickness_kernel!,
        m.H.future,
        m.H.past,
        m.convH,
        m.melt,
        m.nentr,
        m.detr_cap,
        m.tmask,
        m.frazil,
        dt,
    )
    return
end
# `H` is the thickness of the pressure and Coriolis terms (see `precompute_integration_terms!`).
step_u_momentum!(m, dt, H = m.H.present) = _step_momentum!(m, m.U, dt, H)
step_v_momentum!(m, dt, H = m.H.present) = _step_momentum!(m, m.V, dt, H)

# The momentum step of the component `q`, U or V (`_uv`).
function _step_momentum!(m, q, dt, H)
    c = _uv(m, q)
    # Laplacian first: NonlinearLateralViscosity uses `adv` as scratch.
    c.laplace(m)
    outflow = momentum_outflow(m)
    c.advect(m, outflow)
    level = m.params.closure_level
    launch_interior!(c.step, q.future, q.past, q.present, m.dHdt, m.Hdrho, H, m.drho, c.slope,
                     c.partner.present, m.detr, m.detr_cap, m.adv, m.lap, m.tmask, c.mask, c.partner_mask, c.f,
                     closure_field(q, level), closure_field(c.partner, level), m.H.past, m.H.future, m.convH,
                     m.melt, m.nentr, _or_zero(outflow, m.FT), m.frazil, m.gap_drag, m.g, m.C_d,
                     front_pgf_weight(m.front_pressure, m.g), front_drho_weight(m.front_pressure, m.g), c.d, dt,
                     m.params.kinks, m.params.momentum_form)
    return
end

# The transport terms of a tracer `q`: the advection of its content H·q into `adv`, the
# Laplacian of its past level into `lap`, and the outflow rate into `Hq`.
function _tracer_transport!(m, q)
    launch!(_product_kernel!, m.Hq, m.H.present, q.present)
    upwind_advection_T(m.adv, m, m.Hq)
    laplace_T(m.lap, m, q.past)
    outflow_rate!(m, m.params.tracer_form)
    return
end

function step_temperature!(m, dt)
    _tracer_transport!(m, m.T)
    H, T = m.H, m.T
    gamT = first(_exchange_velocities(m))
    launch!(_step_temperature_kernel!, T.future, T.past, T.present, m.dHdt, m.adv, m.nentr, m.detr_cap, m.Ta,
            m.melt, m.Tb, m.lap, H.present, H.past, H.future, m.convH, m.Hq, m.tmask, m.params.tracer_form,
            gamT, m.K_h, m.conv2, m.frazil, m.L, m.c_p, dt)
    return
end

function step_salinity!(m, dt)
    _tracer_transport!(m, m.S)
    H, S = m.H, m.S
    launch!(_step_salinity_kernel!, S.future, S.past, S.present, m.dHdt, m.adv, m.nentr, m.detr_cap, m.Sa,
            m.lap, H.present, H.past, H.future, m.convH, m.melt, m.Hq, m.tmask, m.params.tracer_form,
            m.K_h, m.conv2, m.frazil, dt)
    return
end

# The upper bound of `max_layer_thickness` (see `cap_layer_thickness`), then the
# H_min floor, in one pass.  Both are enforced before the step, by `detr_cap` and
# `ent2` (`step_entrainment!`), so this is a safety net: it binds only where the
# first guess there misses the stepped H — in the bootstrap step, which spans dt
# rather than the 2·dt both are computed for, and by rounding.  Under
# `FluxTracerForm` the tracer step accounts for what it moves (`tracer_update`).
function _clamp_thickness!(m)
    launch!(
        _clamp_thickness_kernel!,
        m.H.future,
        m.z_draft,
        m.z_bed,
        m.tmask,
        m.params.max_layer_thickness,
        m.H_min,
    )
    return
end

@kernel function _clamp_thickness_kernel!(
    H,
    @Const(z_draft),
    @Const(z_bed),
    @Const(tmask),
    max_layer_thickness,
    H_min,
)
    H_min = _val(H_min)
    i, j = @index(Global, NTuple)
    @inbounds begin
        H[i, j] = cap_layer_thickness(
            max_layer_thickness,
            ExactKinks(),   # the stepped H sits on its bounds; see `SmoothKinks`
            H[i, j],
            z_draft[i, j],
            z_bed[i, j],
            tmask[i, j],
        )
        # The H_min floor must respect the domain mask: inactive cells hold H = 0.
        # A non-zero H outside the domain would leak into the interior, because
        # the face averages of the Laplacian stencils (`_face_avg`) divide the
        # *sum* over a cell pair by the number of active cells in it, biasing the
        # diffusion of every boundary cell.
        H[i, j] = max(H[i, j], H_min) * tmask[i, j]
    end
end

# Tracer bounds inside the domain only; see `leapfrog_step!`.
# `quantity` is `Val(:T)` or `Val(:S)`, the ε of `SmoothKinks` it takes.
@kernel function _nan_on_shelf_kernel!(flag, @Const(arr), @Const(tmask))
    i, j = @index(Global, NTuple)
    @inbounds flag[i, j] = isnan(arr[i, j]) & (tmask[i, j] > 0)
end

function _check_nans_shelf!(m, varname, arr, clock)
    launch!(_nan_on_shelf_kernel!, m.diag, arr, m.tmask)
    any(>(0), m.diag) && error(
        "NaN in $varname at t = $(round(_t_days(clock, m), digits=4)) days " *
        "(iteration $(clock.iteration))",
    )
    return nothing
end

# One leapfrog integration of `m` over `nsteps × dt`: `nsteps = 2` is the centred
# step, `nsteps = 1` the first-order bootstrap.  dH/dt keeps the base `dt` (see
# `precompute_integration_terms!`).  Keywords as `_step_fields!`.
leapfrog_step!(m, dt, nsteps; kwargs...) = _step_fields!(m, nsteps * dt, dt + dt, nothing; kwargs...)

# The thickness of the pressure and Coriolis terms: `H.present`, or the blend
# a·H_future + (1 − a − b)·H_present + b·H_past of FB-RK3 written into `m.Hfb`.
_momentum_thickness!(m, ::Nothing) = m.H.present
function _momentum_thickness!(m, (a, b)::Tuple)
    launch!(_blend_kernel!, m.Hfb, m.H.future, m.H.present, m.H.past, a, b)
    return m.Hfb
end

@kernel function _blend_kernel!(out, @Const(H_future), @Const(H_present), @Const(H_past), a, b)
    a, b = _val(a), _val(b)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = a * H_future[i, j] + (one(a) - a - b) * H_present[i, j] + b * H_past[i, j]
end

# Step every prognostic of `m` from `past` over `span`, with its tendencies at `present`
# (the core shared by a leapfrog step and an FB-RK3 stage).  dH/dt is taken over
# `dHdt_span`; `blend` is `nothing` or the FB-RK3 weights `(a, b)` of the momentum
# thickness (`_momentum_thickness!`).  The Laplacians act on the `past` level.  With
# `check_nans`, each prognostic is checked for NaNs on the shelf after its step, and an
# error reports the time of `clock`.
function _step_fields!(m, span, dHdt_span, blend; check_nans = false, clock = nothing)
    step_thickness!(m, span)
    _clamp_thickness!(m)
    H = _momentum_thickness!(m, blend)
    precompute_integration_terms!(m, H, dHdt_span)
    check_nans && _check_nans_shelf!(m, "H", m.H.future, clock)

    # Both momentum components are stepped before the limiter, because it caps the
    # speed and so needs U and V together (see `clamp_velocities!`).
    step_u_momentum!(m, span, H)
    step_v_momentum!(m, span, H)
    clip_front_inflow!(m, m.boundary.open_ocean)
    clamp_velocities!(m)
    check_nans && _check_nans_shelf!(m, "U", m.U.future, clock)
    check_nans && _check_nans_shelf!(m, "V", m.V.future, clock)

    # Tracer bounds (`Params.tracer_bounds`, see `AbstractTracerBounds`).
    step_temperature!(m, span)
    apply_tracer_bounds!(m, m.T.future, Val(:T))
    check_nans && _check_nans_shelf!(m, "T", m.T.future, clock)

    step_salinity!(m, span)
    _gap_surface!(m, m.gap_heat, span)
    apply_tracer_bounds!(m, m.S.future, Val(:S))
    check_nans && _check_nans_shelf!(m, "S", m.S.future, clock)
    return
end

# ============================================================================
# Time-stepping orchestration, by `Params.integrator`
# ============================================================================

# Prepare a model to be stepped at `dt`: run when a Simulation is constructed on a
# freshly initialised model (whose three time levels are identical), and by
# `init_from_restart!` after loading the saved levels.  Keywords as `leapfrog_step!`.
_bootstrap!(m::Model, dt; kwargs...) = _bootstrap!(m, m.params.integrator, dt; kwargs...)

# Re-initialise the time levels after a change of dt.
_rebootstrap!(sim::Simulation) = _rebootstrap!(sim.exec, sim)
_rebootstrap!(::NativeExecution, sim::Simulation) = _rebootstrap!(sim.model, sim.clock.dt; _nan_check(sim)...)
_rebootstrap!(m::Model, dt; kwargs...) = _rebootstrap!(m, m.params.integrator, dt; kwargs...)

# One time step of `m` with the time step `dt`: `time_step!` without the clock.
# Keywords as `leapfrog_step!`.
_step_model!(m::Model, dt; kwargs...) = _step_model!(m, m.params.integrator, dt; kwargs...)

# Leapfrog: refresh secondary fields for the centred step over 2·dt (as every step), then
# take one first-order step.
function _bootstrap!(m, ::LeapfrogIntegrator, dt; kwargs...)
    step_secondary_fields!(m, dt + dt)
    leapfrog_step!(m, dt, 1; kwargs...)
    return
end

# Leapfrog: collapse the `past` level onto `present` so the two are co-located in time,
# then bootstrap at the new dt.  The next `advance_leapfrog!` rotation leaves a
# past/present pair separated by the new dt, so the following centred
# `leapfrog_step!(m, dt, 2)` is consistent.  The anchor is the Robert–Asselin-filtered
# `present`, exactly as at startup.
function _rebootstrap!(m, integrator::LeapfrogIntegrator, dt; kwargs...)
    for var in _prognostics(m)
        var.past .= var.present
    end
    _bootstrap!(m, integrator, dt; kwargs...)
    return
end

# Leapfrog: rotate the levels, take the centred step over 2·dt, filter (Robert–Asselin).
function _step_model!(m, integrator::LeapfrogIntegrator, dt; kwargs...)
    advance_leapfrog!(m, dt)
    leapfrog_step!(m, dt, 2; kwargs...)
    apply_robert_asselin_filter!(m, integrator.nu)
    return
end

function apply_robert_asselin_filter!(m, nu)
    for (var, mask) in
        ((m.H, m.tmask), (m.U, m.umask), (m.V, m.vmask), (m.T, m.tmask), (m.S, m.tmask))
        launch!(
            _robert_asselin_kernel!,
            var.present,
            var.past,
            var.future,
            mask,
            nu,
        )
    end
    # Refresh density and convection on the filtered level, as LADDIE v1
    # does.  The next `advance_leapfrog!` recomputes both on the rotated level, but
    # this call is not redundant: under ResetToAmbientConvection it resets the filtered T/S,
    # which become `past`, and for every scheme it is the `drho` and `convective_instability`
    # that the output, the diagnostics and the CFL number read.
    density!(m)
    convection!(m)
    return
end

function advance_leapfrog!(m, dt)
    for var in _prognostics(m)
        rotate!(var)
    end
    step_secondary_fields!(m, dt + dt)
    return
end

# Per-point scale factors for the speed limiter.  Reads U and V, writes neither,
# so the factors are all computed from the unlimited field and the result cannot
# depend on the order the components are written.
#
# On the C-grid U and V are not co-located, so the partner component is averaged
# onto the point being limited — the same four-point stencil the bottom-drag
# terms use (`u_bottom_drag` / `v_bottom_drag` in test/equation_terms.jl).
# min(1, v_cut / spd).  The divisor of the unused branch is swapped too: reverse-mode
# AD differentiates it, and at rest (spd = 0) its zero adjoint times 1/spd is NaN.
# Under `SmoothKinks`, v_cut / max(spd, v_cut) with a smooth max, which is never
# below v_cut.
@inline _cap_factor(::ExactKinks, spd, v_cut, ε) =
    ifelse(spd > v_cut, v_cut / ifelse(spd > v_cut, spd, one(spd)), one(spd))
@inline _cap_factor(k::SmoothKinks, spd, v_cut, ε) = v_cut / smooth_max(k, spd, v_cut, ε)

@kernel function _speed_scale_kernel!(sU, sV, @Const(U), @Const(V), v_cut, kinks)
    v_cut = _val(v_cut)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    FT = typeof(v_cut)
    @inbounds begin
        jp1 = j + 1
        jm1 = j - 1
        ip1 = i + 1
        im1 = i - 1
        Vbar = (V[i, j] + V[i, jm1] + V[ip1, j] + V[ip1, jm1]) / FT(4)   # V at the U-point
        Ubar = (U[i, j] + U[im1, j] + U[i, jp1] + U[im1, jp1]) / FT(4)   # U at the V-point
        eu = kink_width(kinks, :u)
        spdU = _ssqrt(kinks, U[i, j] * U[i, j] + Vbar * Vbar, eu)
        spdV = _ssqrt(kinks, V[i, j] * V[i, j] + Ubar * Ubar, eu)
        sU[i, j] = _cap_factor(kinks, spdU, v_cut, eu)
        sV[i, j] = _cap_factor(kinks, spdV, v_cut, eu)
    end
end

@kernel function _apply_scale_kernel!(U, V, @Const(sU), @Const(sV))
    i, j = @index(Global, NTuple)
    @inbounds begin
        U[i, j] = U[i, j] * sU[i, j]
        V[i, j] = V[i, j] * sV[i, j]
    end
end

"""
$(TYPEDSIGNATURES)

Cap the flow **speed** at `Params.v_cut` by scaling both velocity components with
one factor, `(U, V) *= min(1, v_cut / |u|)`, so the flow direction is preserved.

As LADDIE v2 (`laddie_velocity.f90`); clamping each component instead would rotate the flow
and admit `√2 · v_cut` on the diagonal. Each component is scaled with the speed at its own
point (4-point average of the other component), from the unlimited field.
"""
function clamp_velocities!(m)
    # The scale factors live in the `adv`/`lap` work buffers: both momentum steps
    # have consumed them, and the tracer steps overwrite them.
    scaleU, scaleV = m.adv, m.lap
    launch_interior!(
        _speed_scale_kernel!,
        scaleU,
        scaleV,
        m.U.future,
        m.V.future,
        m.v_cut,
        m.params.kinks,
    )
    launch!(_apply_scale_kernel!, m.U.future, m.V.future, scaleU, scaleV)
    return
end
