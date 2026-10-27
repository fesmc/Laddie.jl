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

# Conduction into the ice at one cell (see `AbstractIceConduction`): the effective
# `c_i` of the melt-proportional term, and the fixed flux `q` (J kg⁻¹ m s⁻¹, per unit
# mass of freshwater like the melt) that the heat balance carries on top of it.  `Tf`
# is the freezing point at the layer salinity, `T - Tf` having the sign of the melt,
# `Ts` the ice temperature of the forcing.  The ice thickness is the hydrostatic one
# from the draft; `_safe_div` keeps cells without ice at zero.
@inline function _ice_conduction(cond, c_i, T, Tf, Ts, z_draft)
    ci_eff = c_i * ifelse(T > Tf, cond.ci_melt, cond.ci_freeze)
    Hi = -z_draft * cond.thick_ratio
    q = _safe_div(cond.k_ice * (Tf - Ts), Hi * cond.rho_fw)
    return ci_eff, q
end

# The scalars `_ice_conduction` reads, for the kernel launches.
function _conduction_args(m)
    ci_melt, ci_freeze, k_ice = _conduction_coefficients(m.params.ice_conduction, m.FT)
    return (; ci_melt, ci_freeze, k_ice, thick_ratio = m.rho0_seawater / m.rho_ice, rho_fw = m.rho_freshwater)
end

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
    cond,
)
    gamT, gamS, c_p, c_i, L, l1, l2, l3, cond =
        _val(gamT), _val(gamS), _val(c_p), _val(c_i), _val(L), _val(l1), _val(l2), _val(l3), _val(cond)
    i, j = @index(Global, NTuple)
    FT = typeof(c_p)
    @inbounds begin
        # Effective latent heat, per cell: L_eff = L - c_i*T_i.  Colder ice soaks up
        # more heat per unit melt, and the ice base need not be equally cold across
        # the domain, so this is read from the ice forcing rather than a constant.
        Tf_depth = l2 + l3 * z_draft[i, j]
        # The closure for the conduction into the ice sets the c_i of this cell and a
        # fixed flux q.  The flux enters the heat balance as c_p·γT·(T − Tb) − q, so it
        # is the temperature T − q/(c_p·γT) that the equations below see.
        ci_eff, q_cond =
            _ice_conduction(cond, c_i, T[i, j], Tf_depth + l1 * S[i, j], T_ice_base[i, j], z_draft[i, j])
        cp_over_Leff = c_p / (L - ci_eff * T_ice_base[i, j])
        ci_over_cp = ci_eff / c_p
        gT = _at(gamT, i, j)
        gS = _at(gamS, i, j)
        T_bal = T[i, j] - _safe_div(q_cond, c_p * gT)
        quad_b =
            cp_over_Leff * gT * (Tf_depth - T_bal) +
            gS * (one(FT) + cp_over_Leff * ci_over_cp * (Tf_depth + l1 * S[i, j]))
        quad_c = cp_over_Leff * gT * gS * (Tf_depth - T_bal + l1 * S[i, j])
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
                   _safe_div(cp_over_Leff * gT * T_bal - melt_rate, Tb_denom)),
        )
    end
end

@kernel function _ambient_interp_kernel!(
    Ta,
    Sa,
    @Const(z_draft),
    @Const(H),
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
        depth_idx = (z_draft[i, j] - H[i, j] - z0) / dz
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
        Ta[i, j] = weight * _level(Tz, i, j, idx_hi + 1) + (one(FT) - weight) * _level(Tz, i, j, idx_lo + 1)
        Sa[i, j] = weight * _level(Sz, i, j, idx_hi + 1) + (one(FT) - weight) * _level(Sz, i, j, idx_lo + 1)
    end
end

# Level `k` of the ambient profile seen by cell (i, j): the one profile of an
# `OceanForcing1D`, or the cell's own column of an `OceanForcing2D`.
@inline _level(P::AbstractVector, i, j, k) = @inbounds P[k]
@inline _level(P::AbstractArray{<:Any,3}, i, j, k) = @inbounds P[i, j, k]

# The ambient sampling of a `PerBasinOceanForcing1D`: as `_ambient_interp_kernel!`,
# reading the profile of the cell's basin, row `basin[i, j]` of the `(n_basins, nz)`
# tables. A kernel of its own, so the shared one (and its Reactant/Enzyme
# workarounds) stays as it is; the depth index is computed the same way.
@kernel function _ambient_interp_basin_kernel!(
    Ta,
    Sa,
    @Const(z_draft),
    @Const(H),
    @Const(Tz),
    @Const(Sz),
    @Const(basin),
    z0,
    dz,
    nz,
)
    z0, dz = _val(z0), _val(dz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(z0)
        depth_idx = (z_draft[i, j] - H[i, j] - z0) / dz
        depth_idx = ifelse(isfinite(depth_idx), depth_idx, zero(FT))
        depth_idx = clamp(depth_idx, zero(FT), FT(nz - 1))
        lo = floor(_primal(depth_idx))
        idx_lo = unsafe_trunc(Int, lo)
        idx_hi = clamp(idx_lo + 1, 0, nz - 1)
        weight = depth_idx - lo
        b = basin[i, j]
        Ta[i, j] = weight * Tz[b, idx_hi+1] + (one(FT) - weight) * Tz[b, idx_lo+1]
        Sa[i, j] = weight * Sz[b, idx_hi+1] + (one(FT) - weight) * Sz[b, idx_lo+1]
    end
end


"""
$(TYPEDSIGNATURES)

Vertically interpolate the ambient T/S profiles to the depth of each grid cell's
plume base (z_draft − H), writing results into `m.Ta` and `m.Sa`. This sampling has
no numbered equation in Lambert et al. (2023); see `docs/src/equations.md`.
"""
update_ambient_fields!(m) = _update_ambient_fields!(m, m.forcing.ocean)

function _update_ambient_fields!(m, ::PerBasinOceanForcing1D)
    launch!(
        _ambient_interp_basin_kernel!,
        m.Ta,
        m.Sa,
        m.z_draft,
        m.H.present,
        m.Tz,
        m.Sz,
        m.basin,
        m.z0,
        m.dz,
        length(m.z),
    )
    return
end

function _update_ambient_fields!(m, ::AbstractOceanForcing)
    nz = length(m.z)
    launch!(
        _ambient_interp_kernel!,
        m.Ta,
        m.Sa,
        m.z_draft,
        m.H.present,
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
# is floored at 0 (u★H/ν₀ ≥ 1): without the floor the denominator crosses zero on a
# thin, slow layer.  The floor is a no-op wherever u★H/ν₀ ≥ 1, so the result is
# unchanged there; the constructor guarantees the offsets are positive.
@kernel function _turbulent_gamma_kernel!(
    gamT,
    gamS,
    @Const(ustar),
    @Const(H),
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
        logterm = FT(2.12) * max(log(us * H[i, j] / nu0 + FT(1e-12)), zero(FT))
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
        m.H.present,
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
        _conduction_args(m),
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
        _conduction_args(m),
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
    cond,
)
    c_p, c_i, L, l1, l2, l3, cond =
        _val(c_p), _val(c_i), _val(L), _val(l1), _val(l2), _val(l3), _val(cond)
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
            ci_eff, q_cond = _ice_conduction(cond, c_i, T[i, j], tb, T_ice_base[i, j], z_draft[i, j])
            heat = (mdot * (L - ci_eff * (T_ice_base[i, j] - tb)) + q_cond) / c_p
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
    launch!(_ustar_gamma_kernel!, m.gamT, m.gamS, m.ustar, m.tmask, mp.Gamma_T, mp.ratio)
    _launch_three_eq_melt!(m)
end

@kernel function _ustar_gamma_kernel!(gamT, gamS, @Const(ustar), @Const(tmask), Gamma_T, ratio)
    Gamma_T, ratio = _val(Gamma_T), _val(ratio)
    i, j = @index(Global, NTuple)
    @inbounds begin
        g = Gamma_T * ustar[i, j] * tmask[i, j]
        gamT[i, j] = g
        gamS[i, j] = g / ratio
    end
end

"""
$(TYPEDSIGNATURES)

Two-equation ice–ocean melt parameterisation (see [`TwoEquationMelting`](@ref)).
Sets `m.ustar`, `m.gamT`, `m.melt`, `m.Tb` (the freezing point at the layer salinity);
`m.gamS` is not used.
"""
function update_melt!(m, mp::TwoEquationMelting)
    update_ustar!(m)
    launch!(_ustar_gamT_kernel!, m.gamT, m.ustar, m.tmask, mp.Gamma_TS)
    launch!(
        _two_eq_melt_kernel!,
        m.melt,
        m.Tb,
        m.T.present,
        m.S.present,
        m.z_draft,
        m.tmask,
        m.imask,
        m.T_ice_base,
        m.gamT,
        m.c_p,
        m.c_i,
        m.L,
        m.l1,
        m.l2,
        m.l3,
        _conduction_args(m),
    )
end

@kernel function _ustar_gamT_kernel!(gamT, @Const(ustar), @Const(tmask), Gamma)
    Gamma = _val(Gamma)
    i, j = @index(Global, NTuple)
    @inbounds gamT[i, j] = Gamma * ustar[i, j] * tmask[i, j]
end

# c_p·γT·(T − Tb) = ṁ·(L − c_i,eff·(T_i − Tb)) + q at Tb = Tf(S): linear in ṁ.  Gap
# cells follow the three-equation kernel: no melt, and Tb = T so that the exchange term
# of the temperature equation vanishes.
@kernel function _two_eq_melt_kernel!(
    melt,
    Tb,
    @Const(T),
    @Const(S),
    @Const(z_draft),
    @Const(tmask),
    @Const(imask),
    @Const(T_ice_base),
    @Const(gamT),
    c_p,
    c_i,
    L,
    l1,
    l2,
    l3,
    cond,
)
    c_p, c_i, L, l1, l2, l3, cond =
        _val(c_p), _val(c_i), _val(L), _val(l1), _val(l2), _val(l3), _val(cond)
    i, j = @index(Global, NTuple)
    FT = typeof(c_p)
    @inbounds begin
        Tf = _freezing_point(S[i, j], z_draft[i, j], l1, l2, l3)
        ci_eff, q_cond = _ice_conduction(cond, c_i, T[i, j], Tf, T_ice_base[i, j], z_draft[i, j])
        melt_rate = (c_p * gamT[i, j] * (T[i, j] - Tf) - q_cond) / (L - ci_eff * (T_ice_base[i, j] - Tf))
        melt[i, j] = iszero(imask[i, j]) ? zero(FT) : melt_rate
        Tb[i, j] = ifelse(iszero(tmask[i, j]), zero(FT), ifelse(iszero(imask[i, j]), T[i, j], Tf))
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
        m.H.present,
        m.tmask,
        coeff,
        drho_coeff,
    )
end

"""
$(TYPEDSIGNATURES)

Gaspar (1988) mechanical-energy entrainment (see [`GasparEntrainment`](@ref)):
``e = 2\\mu u_\\star^3 / (g H \\delta\\rho)`` minus a melt detrainment correction
(Lambert et al. 2023, Eq. 14; see `docs/src/equations.md`).
"""
function _compute_entrainment!(m, ep::GasparEntrainment)
    launch!(
        _buoyancy_entrainment_kernel!,
        m.entr,
        m.detr,
        m.T.present,
        m.S.present,
        m.Tb,
        m.z_draft,
        m.ustar,
        m.H.present,
        m.drho,
        m.melt,
        m.tmask,
        (ep.mu + ep.mu) / m.g,
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
`ent2` is the minimum additional entrainment rate needed to prevent H falling
below `H_min` after the upcoming thickness step; it is computed here (before
stepping) from `H.past`, matching the reference Python LADDIE implementation.

`detr_cap` is the mirror image for the upper bound `Params.max_layer_thickness`:
the detrainment rate that brings the same first guess of `H` down to the cap, as
LADDIE v2 does (`laddie_thickness.f90`).  It removes layer water at the layer's own
T, S and velocity, so it thins the layer without changing its tracers or momentum
(see the step kernels in `numerics.jl`); it is zero wherever the cap does not bind.
"""
function update_entrainment!(m, dt)
    _compute_entrainment!(m, m.entrainment)
    upwind_advection_T(m.convH, m, m.H.present)
    FT = m.FT
    launch!(
        _net_entrainment_kernel!,
        m.ent2,
        m.nentr,
        m.detr_cap,
        m.H.past,
        m.convH,
        m.melt,
        m.entr,
        m.detr,
        m.z_draft,
        m.z_bed,
        m.tmask,
        m.frazil,
        m.params.max_layer_thickness,
        m.H_min,
        FT(2) * dt,
    )
    return
end

@kernel function _net_entrainment_kernel!(
    ent2,
    nentr,
    detr_cap,
    @Const(H_past),
    @Const(convH),
    @Const(melt),
    @Const(entr),
    @Const(detr),
    @Const(z_draft),
    @Const(z_bed),
    @Const(tmask),
    frazil,
    max_layer_thickness,
    H_min,
    dt2,
)
    H_min, dt2, frazil = _val(H_min), _val(dt2), _val(frazil)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(H_min)
        # Frazil formation removes layer water like a sink, so the floor tops it up.
        ent2[i, j] =
            max(
                zero(H_min),
                (H_min - H_past[i, j]) / dt2 -
                (convH[i, j] + melt[i, j] - _at(frazil, i, j) + entr[i, j] - detr[i, j]),
            ) * tmask[i, j]
        nentr[i, j] = entr[i, j] + ent2[i, j] - detr[i, j]
        # First guess of the stepped H, then the excess over the cap.  The floor
        # wins where the cap is below H_min, as in `_clamp_thickness_kernel!`.
        # Masked cells may sit under a bed of -Inf, hence the `ifelse`.
        H_star = H_past[i, j] + (convH[i, j] + melt[i, j] - _at(frazil, i, j) + nentr[i, j]) * dt2
        H_top = max(
            _max_layer_thickness(max_layer_thickness, H_star, z_draft[i, j], z_bed[i, j], one(FT)),
            H_min,
        )
        detr_cap[i, j] =
            ifelse(tmask[i, j] > 0, max(zero(FT), (H_star - H_top) / dt2), zero(FT))
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
# detr into one pass; only entr and detr are stored.  Production term =
# prefactor·u★³/(H·δρ) with prefactor = 2μ/g (GasparEntrainment).
# drho_pos = max(drho_floor, drho) is never zero, so the δρ_b/drho_pos division is
# safe; H can be zero outside the domain, hence _safe_div for the production term.
@kernel function _buoyancy_entrainment_kernel!(
    entr,
    detr,
    @Const(T),
    @Const(S),
    @Const(Tb),
    @Const(z_draft),
    @Const(ustar),
    @Const(H),
    @Const(drho),
    @Const(melt),
    @Const(tmask),
    prefactor,
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
        e_ij =
            prefactor * _safe_div(us3, H[i, j] * drho_pos) -
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
    @Const(H),
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
            max(zero(FT), u_im * u_im + v_jm * v_jm - drho_coeff * drho[i, j] * H[i, j])
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
    update_frazil!(m)
    update_entrainment!(m, dt)
    return
end

"""
$(TYPEDSIGNATURES)

Frazil-ice formation rate `m.frazil` (m s⁻¹, freshwater equivalent) under
[`RelaxedFrazilIce`](@ref): the rate whose latent heat relaxes the supercooling of
the past level, ``\\max(T_f - T, 0)``, over `relaxation_time`.  Ice-covered cells
only (`imask`): a gap cell is open water, and the ice it would form is sea ice, not
marine ice.  Nothing to do under [`NoFrazilIce`](@ref), where `m.frazil` is a scalar
zero.  Computed before `update_entrainment!`, whose preventive bounds on H include
it.
"""
update_frazil!(m) = update_frazil!(m, m.frazil_ice)
update_frazil!(m, ::NoFrazilIce) = nothing
function update_frazil!(m, fz::RelaxedFrazilIce)
    launch!(
        _frazil_kernel!,
        m.frazil,
        m.T.past,
        m.S.past,
        m.H.present,
        m.z_draft,
        m.imask,
        fz.relaxation_time,
        m.c_p,
        m.L,
        m.l1,
        m.l2,
        m.l3,
    )
end

@kernel function _frazil_kernel!(
    frazil,
    @Const(T),
    @Const(S),
    @Const(H),
    @Const(z_draft),
    @Const(imask),
    relaxation_time,
    c_p,
    L,
    l1,
    l2,
    l3,
)
    relaxation_time, c_p, L, l1, l2, l3 =
        _val(relaxation_time), _val(c_p), _val(L), _val(l1), _val(l2), _val(l3)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(c_p)
        supercooling = max(zero(FT), l1 * S[i, j] + l2 + l3 * z_draft[i, j] - T[i, j])
        frazil[i, j] = imask[i, j] * c_p / L * H[i, j] * supercooling / relaxation_time
    end
end
