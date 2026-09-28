# Point-wise equation-of-state and freezing-point pieces shared by several kernels.
# Each spells out one formula in one operation order, so every caller rounds alike.

# Buoyancy contrast (ρ_a − ρ)/ρ₀ of water (S, T) against (Sa, Ta), linear equation
# of state.
@inline _drho(Sa, S, Ta, T, beta, alpha) = beta * (Sa - S) - alpha * (Ta - T)

# Liquidus: freezing temperature at salinity `S` and depth `z`, and its inverse, the
# salinity whose freezing temperature at `z` is `Tf`.  `_three_eq_melt_kernel!`
# groups `l2 + l3*z` separately and so keeps its own copy.
@inline _freezing_point(S, z, l1, l2, l3) = l1 * S + l2 + l3 * z
@inline _freezing_salinity(Tf, z, l1, l2, l3) = (Tf - l2 - l3 * z) / l1

# γ_T / γ_S, the fixed ratio of the heat and salt exchange velocities.
const _GAMMA_T_OVER_S = 35

@kernel function _density_kernel!(
    drho,
    @Const(Sa),
    @Const(S),
    @Const(Ta),
    @Const(T),
    @Const(tmask),
    beta,
    alpha,
)
    beta, alpha = _val(beta), _val(alpha)
    i, j = @index(Global, NTuple)
    @inbounds drho[i, j] = _drho(Sa[i, j], S[i, j], Ta[i, j], T[i, j], beta, alpha) * tmask[i, j]
end

# Three-equation melt parameterisation (Jenkins 1991) + ice-base temperature.
# `gamT`/`gamS` are scalars (FixedGamTMelting) or per-cell fields
# (TurbulentGamTMelting); `_at` reads either.
@kernel function _three_eq_melt_kernel!(
    melt,
    Tb,
    @Const(T),
    @Const(S),
    @Const(z_draft),
    @Const(tmask),
    @Const(imask),
    @Const(T_ice_base),
    gamT,
    gamS,
    c_p,
    c_i,
    L,
    l1,
    l2,
    l3,
)
    gamT, gamS, c_p, c_i, L, l1, l2, l3 =
        _val(gamT), _val(gamS), _val(c_p), _val(c_i), _val(L), _val(l1), _val(l2), _val(l3)
    i, j = @index(Global, NTuple)
    FT = typeof(c_p)
    @inbounds begin
        # Effective latent heat, per cell: L_eff = L - c_i*T_i.  Colder ice soaks up
        # more heat per unit melt, and the ice base need not be equally cold across
        # the domain, so this is read from the ice forcing rather than a constant.
        cp_over_Leff = c_p / (L - c_i * T_ice_base[i, j])
        ci_over_cp = c_i / c_p
        gT = _at(gamT, i, j)
        gS = _at(gamS, i, j)
        Tf_depth = l2 + l3 * z_draft[i, j]
        quad_b =
            cp_over_Leff * gT * (Tf_depth - T[i, j]) +
            gS * (one(FT) + cp_over_Leff * ci_over_cp * (Tf_depth + l1 * S[i, j]))
        quad_c = cp_over_Leff * gT * gS * (Tf_depth - T[i, j] + l1 * S[i, j])
        disc = quad_b * quad_b - FT(4) * quad_c
        disc = ifelse(disc < zero(FT), zero(FT), disc)
        melt_rate = (-quad_b + _safe_sqrt(disc)) / FT(2)
        # Only ice-covered cells melt.  Gap cells (in tmask, not in imask) are ice-free,
        # so they add no meltwater volume and no buoyancy — and their Tb is set to T,
        # which makes the ice-ocean heat exchange -gamT*(T - Tb) in the temperature
        # equation vanish identically.  That is exactly what the reference gets from
        # its melt = min(melt, Hi/dt) limiter: with melt = 0 the three-equation
        # solution for Tb collapses to Tb = T.
        melt[i, j] = iszero(imask[i, j]) ? zero(FT) : melt_rate
        Tb_denom = cp_over_Leff * gT + cp_over_Leff * ci_over_cp * melt_rate
        # `ifelse` + `_safe_div`: reverse-mode AD differentiates the unused branches
        # too, and γ_T is zero outside the domain under the u★-dependent schemes.
        Tb[i, j] = ifelse(
            iszero(tmask[i, j]),
            zero(FT),
            ifelse(iszero(imask[i, j]), T[i, j],
                   _safe_div(cp_over_Leff * gT * T[i, j] - melt_rate, Tb_denom)),
        )
    end
end

@kernel function _ambient_interp_kernel!(
    Ta,
    Sa,
    @Const(z_draft),
    @Const(D),
    @Const(Tz),
    @Const(Sz),
    z0,
    dz,
    nz,
)
    z0, dz = _val(z0), _val(dz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(z0)
        depth_idx = (z_draft[i, j] - D[i, j] - z0) / dz
        # Guard against non-finite or out-of-range depth_idx before the integer
        # conversion: trunc(Int, x) throws InexactError on CPU and is UB on GPU
        # when x overflows Int64 (~9.2e18).  Clamp to [0, nz-1] in float first
        # so trunc always receives a representable value.
        depth_idx = ifelse(isfinite(depth_idx), depth_idx, zero(FT))
        depth_idx = clamp(depth_idx, zero(FT), FT(nz - 1))
        # unsafe_trunc: the value is in range after the clamp, and the checked
        # `trunc` leaves an InexactError branch (a trap) that Reactant cannot raise.
        # The weight subtracts `floor` (= trunc, depth_idx ≥ 0), not the index
        # converted back: Enzyme on raised kernels (Reactant 0.2.286) passes the
        # derivative through the int round trip, so `depth_idx - FT(idx_lo)` gets
        # a zero tangent.
        lo = floor(_primal(depth_idx))
        idx_lo = unsafe_trunc(Int, lo)
        idx_hi = clamp(idx_lo + 1, 0, nz - 1)
        weight = depth_idx - lo
        Ta[i, j] = weight * Tz[idx_hi+1] + (one(FT) - weight) * Tz[idx_lo+1]
        Sa[i, j] = weight * Sz[idx_hi+1] + (one(FT) - weight) * Sz[idx_lo+1]
    end
end


"""
$(TYPEDSIGNATURES)

Vertically interpolate the ambient T/S profiles to the depth of each grid cell's
plume base (z_draft − D), writing results into `m.Ta` and `m.Sa`. This sampling has
no numbered equation in Lambert et al. (2023); see `docs/src/equations.md`.
"""
function update_ambient_fields!(m)
    nz = length(m.z)
    launch!(
        _ambient_interp_kernel!,
        m.Ta,
        m.Sa,
        m.z_draft,
        m.D.present,
        m.Tz,
        m.Sz,
        m.z0,
        m.dz,
        nz,
    )
    return
end

"Reduced (dimensionless) density ``\\delta\\rho = \\Delta\\rho_a/\\rho_0 = \\beta(S_a - S) - \\alpha(T_a - T)``  (Lambert et al. 2023, Eqs. 6–7)."
update_density!(m) = launch!(
    _density_kernel!,
    m.drho,
    m.Sa,
    m.S.present,
    m.Ta,
    m.T.present,
    m.tmask,
    m.beta,
    m.alpha,
)

"""
$(TYPEDSIGNATURES)

Flag convectively unstable cells and clamp ``\\delta\\rho`` to a minimum positive value
so the plume remains denser than ambient.

Applies in gap cells too (`tmask` but not `imask`): the buoyancy floor is the one
convection treatment LADDIE v2 also has, and it applies there over its whole active
domain, gaps included.
"""
function update_convection!(m, cs::ClampDensity)
    thr = cs.d_rho_min / m.rho0_seawater
    launch!(_clamp_density_kernel!, m.convection, m.drho, thr)
end

# The convection schemes are pointwise, and run as kernels rather than broadcasts so
# that they are threaded on the CPU: they run twice per step (see
# `apply_robert_asselin_filter!`).  Each reads the `drho` of the current T/S.
@kernel function _clamp_density_kernel!(convection, drho, thr)
    thr = _val(thr)
    i, j = @index(Global, NTuple)
    FT = typeof(thr)
    @inbounds begin
        d = drho[i, j]
        convection[i, j] = ifelse(d < 0, one(FT), zero(FT))
        drho[i, j] = max(d, thr)
    end
end

"""
$(TYPEDSIGNATURES)

Flag convectively unstable cells, then instantly reset their T/S to ambient
values so the density remains stable.

Restricted to ice-covered cells (`imask`).  In a gap the ambient profile is sampled at
the sea surface, where it is cold and fresh, so `drho < 0` is close to unconditional
there; resetting would overwrite the T/S anomaly the layer is carrying across the gap
and rebuild the meltwater sink that [`ConnectedGapsBC`](@ref) exists to remove.  LADDIE
v2 has no reset scheme to copy here, so this is a Laddie.jl-only decision.
"""
function update_convection!(m, cs::ResetToAmbient)
    thr = cs.d_rho_min / m.rho0_seawater
    S_adj = cs.d_rho_min / (m.rho0_seawater * m.beta)
    launch!(
        _reset_to_ambient_kernel!,
        m.convection,
        m.T.present,
        m.S.present,
        m.drho,
        m.Ta,
        m.Sa,
        m.tmask,
        m.imask,
        thr,
        S_adj,
        m.beta,
        m.alpha,
    )
end

# Reset T/S of unstable ice-covered cells to ambient, and refresh `drho` there with
# the expression of `_density_kernel!` (elsewhere T/S are unchanged, so is `drho`).
@kernel function _reset_to_ambient_kernel!(
    convection,
    T,
    S,
    drho,
    @Const(Ta),
    @Const(Sa),
    @Const(tmask),
    @Const(imask),
    thr,
    S_adj,
    beta,
    alpha,
)
    thr, S_adj, beta, alpha = _val(thr), _val(S_adj), _val(beta), _val(alpha)
    i, j = @index(Global, NTuple)
    FT = typeof(thr)
    @inbounds begin
        d = drho[i, j]
        ice = imask[i, j] > 0
        convection[i, j] = ifelse((d < 0) & ice, one(FT), zero(FT))
        if (d < thr) & ice
            T[i, j] = Ta[i, j]
            S[i, j] = Sa[i, j] - S_adj
            drho[i, j] = _drho(Sa[i, j], S[i, j], Ta[i, j], T[i, j], beta, alpha) * tmask[i, j]
        end
    end
end

"""
$(TYPEDSIGNATURES)

Flag convectively unstable cells; relaxation is applied implicitly during the
tracer time step via `conv2`.

Restricted to ice-covered cells (`imask`) for the same reason as
[`ResetToAmbient`](@ref) — relaxing a gap cell towards surface ambient is a slower
version of the same sink.
"""
function update_convection!(m, ::RelaxToAmbient)
    launch!(_flag_unstable_ice_kernel!, m.convection, m.drho, m.imask)
end

@kernel function _flag_unstable_ice_kernel!(convection, @Const(drho), @Const(imask))
    i, j = @index(Global, NTuple)
    FT = eltype(convection)
    @inbounds convection[i, j] =
        ifelse((drho[i, j] < 0) & (imask[i, j] > 0), one(FT), zero(FT))
end

update_convection!(m) = update_convection!(m, m.convection_scheme)

# Log-layer transfer coefficients (Lambert et al. 2023, Eqs. 11–12).  The log term
# is floored at 0 (u★D/ν₀ ≥ 1): without the floor the denominator crosses zero on a
# thin, slow layer.  The floor is a no-op wherever u★D/ν₀ ≥ 1, so the result is
# unchanged there; the constructor guarantees the offsets are positive.
@kernel function _turbulent_gamma_kernel!(
    gamT,
    gamS,
    @Const(ustar),
    @Const(D),
    @Const(tmask),
    PrCorr,
    ScCorr,
    nu0,
)
    PrCorr, ScCorr, nu0 = _val(PrCorr), _val(ScCorr), _val(nu0)
    i, j = @index(Global, NTuple)
    FT = typeof(nu0)
    @inbounds begin
        us = ustar[i, j]
        logterm = FT(2.12) * max(log(us * D[i, j] / nu0 + FT(1e-12)), zero(FT))
        active = tmask[i, j] > 0
        gamT[i, j] = ifelse(active, us / (logterm + PrCorr), zero(FT))
        gamS[i, j] = ifelse(active, us / (logterm + ScCorr), zero(FT))
    end
end

function _compute_turbulent_transfer_coefficients!(m, mp::TurbulentGamTMelting)
    launch!(
        _turbulent_gamma_kernel!,
        m.gamT,
        m.gamS,
        m.ustar,
        m.D.present,
        m.tmask,
        _log_layer_offset(mp.Pr),
        _log_layer_offset(mp.Sc),
        mp.nu0,
    )
end

"""
$(TYPEDSIGNATURES)

Three-equation ice-ocean melt parameterisation with a fixed heat transfer
coefficient ``\\gamma_T`` (Jenkins 1991; Lambert et al. 2023, Eqs. 8–10 and 13).
Sets `m.ustar`, `m.melt`, `m.Tb`; `m.gamT` and `m.gamS` hold the constant
exchange velocities from the start.
"""
function update_melt!(m, ::FixedGamTMelting)
    update_ustar!(m)
    _launch_three_eq_melt!(m)
end

# γ_T and γ_S as the kernels take them.  FixedGamTMelting's constants come straight
# from the parameter, so that a parameter traced by Reactant reaches the kernels
# (the cache holds a copy for reporting, set at build: a traced time step cannot
# reassign a scalar field).  The other schemes compute fields into the cache.
_exchange_velocities(m) = _exchange_velocities(m, m.params.melting)
_exchange_velocities(m, mp::FixedGamTMelting) = _fixed_exchange_velocities(mp, m.FT)
_exchange_velocities(m, ::AbstractMelting) = (m.gamT, m.gamS)
_fixed_exchange_velocities(mp, FT) = (mp.gamTfix, mp.gamTfix / FT(_GAMMA_T_OVER_S))

function _launch_three_eq_melt!(m)
    gamT, gamS = _exchange_velocities(m)
    launch!(
        _three_eq_melt_kernel!,
        m.melt,
        m.Tb,
        m.T.present,
        m.S.present,
        m.z_draft,
        m.tmask,
        m.imask,
        m.T_ice_base,
        gamT,
        gamS,
        m.c_p,
        m.c_i,
        m.L,
        m.l1,
        m.l2,
        m.l3,
    )
end

"""
$(TYPEDSIGNATURES)

Prescribed melt rate (see [`PrescribedMelting`](@ref)): sets `m.melt` from the
prescribed field, `m.Tb` to the local freezing point, and `m.gamT` to the transfer
coefficient that makes the ice–ocean heat flux match the prescribed melt.  Also
sets `m.ustar`, which entrainment needs.
"""
function update_melt!(m, ::PrescribedMelting)
    update_ustar!(m)
    launch!(
        _prescribed_melt_kernel!,
        m.melt,
        m.Tb,
        m.gamT,
        m.T.present,
        m.S.present,
        m.z_draft,
        m.tmask,
        m.imask,
        m.T_ice_base,
        m.melt_prescribed,
        m.c_p,
        m.c_i,
        m.L,
        m.l1,
        m.l2,
        m.l3,
    )
end

# Prescribed melt with a consistent heat sink.  The interface sits at the local
# freezing point, and the layer gives up the heat the prescribed melt requires,
# c_p·γT·(T − Tb) = ṁ·(L − c_i·(T_i − Tb)) — the heat balance of the three-equation
# model with ṁ given.  The temperature equation carries that flux as −γT·(T − Tb),
# so γT is reported as the flux divided by (T − Tb); where T == Tb exactly the flux
# has no representation in that form and is dropped.  Gap cells follow the
# three-equation kernels: no melt, and Tb = T so the exchange term vanishes.
@kernel function _prescribed_melt_kernel!(
    melt,
    Tb,
    gamT,
    @Const(T),
    @Const(S),
    @Const(z_draft),
    @Const(tmask),
    @Const(imask),
    @Const(T_ice_base),
    @Const(melt_prescribed),
    c_p,
    c_i,
    L,
    l1,
    l2,
    l3,
)
    c_p, c_i, L, l1, l2, l3 = _val(c_p), _val(c_i), _val(L), _val(l1), _val(l2), _val(l3)
    i, j = @index(Global, NTuple)
    FT = typeof(c_p)
    @inbounds begin
        z = zero(FT)
        if iszero(tmask[i, j])
            melt[i, j] = z
            Tb[i, j] = z
            gamT[i, j] = z
        elseif iszero(imask[i, j])
            melt[i, j] = z
            Tb[i, j] = T[i, j]
            gamT[i, j] = z
        else
            mdot = melt_prescribed[i, j]
            tb = _freezing_point(S[i, j], z_draft[i, j], l1, l2, l3)
            heat = mdot * (L - c_i * (T_ice_base[i, j] - tb)) / c_p
            melt[i, j] = mdot
            Tb[i, j] = tb
            gamT[i, j] = _safe_div(heat, T[i, j] - tb)
        end
    end
end

"""
$(TYPEDSIGNATURES)

Three-equation ice-ocean melt parameterisation with turbulence-dependent
transfer coefficients ``\\gamma_T``, ``\\gamma_S`` via the log-layer formulation
(Holland & Jenkins 1999; Lambert et al. 2023, Eqs. 8–13).
Sets `m.ustar`, `m.gamT`, `m.gamS`, `m.melt`, `m.Tb`.
"""
function update_melt!(m, mp::TurbulentGamTMelting)
    update_ustar!(m)
    _compute_turbulent_transfer_coefficients!(m, mp)
    _launch_three_eq_melt!(m)
end

"""
$(TYPEDSIGNATURES)

Three-equation melt with transfer coefficients proportional to the friction velocity,
``\\gamma_T = \\Gamma_T u_\\star`` and ``\\gamma_S = \\gamma_T/35``.
Sets `m.ustar`, `m.gamT`, `m.gamS`, `m.melt`, `m.Tb`.
"""
function update_melt!(m, mp::UStarGamTMelting)
    update_ustar!(m)
    launch!(_ustar_gamma_kernel!, m.gamT, m.gamS, m.ustar, m.tmask, mp.Gamma_T)
    _launch_three_eq_melt!(m)
end

@kernel function _ustar_gamma_kernel!(gamT, gamS, @Const(ustar), @Const(tmask), Gamma_T)
    Gamma_T = _val(Gamma_T)
    i, j = @index(Global, NTuple)
    FT = typeof(Gamma_T)
    @inbounds begin
        g = Gamma_T * ustar[i, j] * tmask[i, j]
        gamT[i, j] = g
        gamS[i, j] = g / FT(_GAMMA_T_OVER_S)
    end
end

update_melt!(m) = update_melt!(m, m.melting)

"""
$(TYPEDSIGNATURES)

Holland–Jenkins shear entrainment (see [`HollandEntrainment`](@ref)), an
alternative to the buoyancy-flux form of Lambert et al. (2023, Eq. 14).
"""
function _compute_entrainment!(m, ep::HollandEntrainment)
    coeff = ep.cl * m.K_h / m.A_h^2
    drho_coeff = m.g * m.K_h / m.A_h
    launch_interior!(
        _holland_entrainment_kernel!,
        m.entr,
        m.detr,
        m.U.present,
        m.V.present,
        m.drho,
        m.D.present,
        m.tmask,
        coeff,
        drho_coeff,
    )
end

"""
$(TYPEDSIGNATURES)

Reference-LADDIE mechanical-energy entrainment: ``e = 2\\mu u_\\star^3 / (g D \\delta\\rho)`` minus a melt
detrainment correction. This is the form verified against the Python reference
(Lambert et al. 2023; see `docs/src/equations.md`). Contrast
`_compute_entrainment!(m, ::GasparEntrainment)`, the literal Eq. 14.
"""
function _compute_entrainment!(m, ep::LambertEntrainment)
    _launch_buoyancy_entrainment!(m, (ep.mu + ep.mu) / m.g, false)
end

"""
$(TYPEDSIGNATURES)

Literal Eq. 14 entrainment: ``e = \\mu u_\\star^3 / (g D^2 \\delta\\rho)`` minus the same melt detrainment
correction (Lambert et al. 2023, Eq. 14; see `docs/src/equations.md`). Differs
from the reference [`LambertEntrainment`](@ref) by the ``D^2`` denominator and the
``\\mu`` (not ``2\\mu``) prefactor.
"""
function _compute_entrainment!(m, ep::GasparEntrainment)
    _launch_buoyancy_entrainment!(m, ep.mu / m.g, true)
end

# Shared by Lambert and Gaspar: they differ only in the production prefactor and
# in whether the production term divides by D or by D².
function _launch_buoyancy_entrainment!(m, prefactor, D_squared)
    launch!(
        _buoyancy_entrainment_kernel!,
        m.entr,
        m.detr,
        m.T.present,
        m.S.present,
        m.Tb,
        m.z_draft,
        m.ustar,
        m.D.present,
        m.drho,
        m.melt,
        m.tmask,
        prefactor,
        D_squared,
        m.max_detrainment,
        m.drho_floor,
        m.alpha,
        m.beta,
        m.l1,
        m.l2,
        m.l3,
    )
end

"""
$(TYPEDSIGNATURES)

Compute entrainment/detrainment rates and assemble the net entrainment
`m.nentr = entr + ent2 - detr` that enters the thickness and tracer equations.
`ent2` is the minimum additional entrainment rate needed to prevent D falling
below `D_min` after the upcoming thickness step; it is computed here (before
stepping) from `D.past`, matching the reference Python LADDIE implementation.
"""
function update_entrainment!(m, dt)
    _compute_entrainment!(m, m.entrainment)
    upwind_advection_T(m.convD, m, m.D.present)
    FT = m.FT
    launch!(
        _net_entrainment_kernel!,
        m.ent2,
        m.nentr,
        m.D.past,
        m.convD,
        m.melt,
        m.entr,
        m.detr,
        m.tmask,
        m.D_min,
        FT(2) * dt,
    )
    return
end

@kernel function _net_entrainment_kernel!(
    ent2,
    nentr,
    @Const(D_past),
    @Const(convD),
    @Const(melt),
    @Const(entr),
    @Const(detr),
    @Const(tmask),
    D_min,
    dt2,
)
    D_min, dt2 = _val(D_min), _val(dt2)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ent2[i, j] =
            max(
                zero(D_min),
                (D_min - D_past[i, j]) / dt2 -
                (convD[i, j] + melt[i, j] + entr[i, j] - detr[i, j]),
            ) * tmask[i, j]
        nentr[i, j] = entr[i, j] + ent2[i, j] - detr[i, j]
    end
end

# Friction velocity at the T-point: u★ = √(C_d_top · (im_half(U)² + jm_half(V)² + u_tide²))
# im_half(U)[i,j] = (U[i,j] + U[i−1,j]) / 2,  jm_half(V)[i,j] = (V[i,j] + V[i,j−1]) / 2
function update_ustar!(m)
    launch_interior!(
        _ustar_kernel!,
        m.ustar,
        m.U.present,
        m.V.present,
        m.tmask,
        m.C_d_top,
        m.u_tide,
    )
end

@kernel function _ustar_kernel!(ustar, @Const(U), @Const(V), @Const(tmask), C_d_top, u_tide)
    C_d_top, u_tide = _val(C_d_top), _val(u_tide)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        FT = typeof(C_d_top)
        half = FT(0.5)
        im1 = i - 1
        jm1 = j - 1
        u_im = (U[i, j] + U[im1, j]) * half
        v_jm = (V[i, j] + V[i, jm1]) * half
        ustar[i, j] =
            sqrt(C_d_top * (u_im * u_im + v_jm * v_jm + u_tide * u_tide)) * tmask[i, j]
    end
end

# Buoyancy-flux entrainment, fusing S_b, δρ_b, drho_pos, the signed rate, entr and
# detr into one pass; only entr and detr are stored.  Production term = prefactor·u★³/(D·δρ) for the reference form
# (LambertEntrainment, prefactor = 2μ/g) and prefactor·u★³/(D²·δρ) for the literal
# Eq. 14 (GasparEntrainment, prefactor = μ/g, `D_squared = true`).
# drho_pos = max(drho_floor, drho) is never zero, so the δρ_b/drho_pos division is
# safe; D can be zero outside the domain, hence _safe_div for the production term.
@kernel function _buoyancy_entrainment_kernel!(
    entr,
    detr,
    @Const(T),
    @Const(S),
    @Const(Tb),
    @Const(z_draft),
    @Const(ustar),
    @Const(D),
    @Const(drho),
    @Const(melt),
    @Const(tmask),
    prefactor,
    D_squared,
    max_detrainment,
    drho_floor,
    alpha,
    beta,
    l1,
    l2,
    l3,
)
    prefactor, max_detrainment, drho_floor, alpha, beta, l1, l2, l3 =
        _val(prefactor), _val(max_detrainment), _val(drho_floor), _val(alpha), _val(beta), _val(l1), _val(l2), _val(l3)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(prefactor)
        drho_pos = max(drho_floor, drho[i, j])
        sb = _freezing_salinity(Tb[i, j], z_draft[i, j], l1, l2, l3)
        db_ij = _drho(S[i, j], sb, T[i, j], Tb[i, j], beta, alpha) * tmask[i, j]
        us3 = ustar[i, j]^3
        Dij = D[i, j]
        Dpow = D_squared ? Dij * Dij : Dij
        e_ij =
            prefactor * _safe_div(us3, Dpow * drho_pos) -
            db_ij / drho_pos * melt[i, j] * tmask[i, j]
        z = zero(FT)
        entr[i, j] = max(e_ij, z)
        detr[i, j] = min(max_detrainment, max(-e_ij, z))
    end
end

# Holland–Jenkins entrainment fused with im/jm: avoids two circshift allocations.
@kernel function _holland_entrainment_kernel!(
    entr,
    detr,
    @Const(U),
    @Const(V),
    @Const(drho),
    @Const(D),
    @Const(tmask),
    coeff,
    drho_coeff,
)
    coeff, drho_coeff = _val(coeff), _val(drho_coeff)
    i0, j0 = @index(Global, NTuple)
    i, j = i0 + 1, j0 + 1   # interior launch (`launch_interior!`)
    @inbounds begin
        FT = typeof(coeff)
        half = FT(0.5)
        im1 = i - 1
        jm1 = j - 1
        u_im = (U[i, j] + U[im1, j]) * half
        v_jm = (V[i, j] + V[i, jm1]) * half
        speed_sq =
            max(zero(FT), u_im * u_im + v_jm * v_jm - drho_coeff * drho[i, j] * D[i, j])
        entr[i, j] = coeff * _safe_sqrt(speed_sq) * tmask[i, j]
        detr[i, j] = zero(FT)
    end
end

# `dt` is the base time step; only the `ent2` top-up in `update_entrainment!` needs it.
function update_secondary_fields!(m, dt)
    update_ambient_fields!(m)
    update_density!(m)
    update_convection!(m)
    update_melt!(m)
    update_entrainment!(m, dt)
    return
end
