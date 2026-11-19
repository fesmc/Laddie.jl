"""
Abstract supertype for the ice–ocean melt parameterisation.  Pass a concrete
instance as `Params(; melting = ...)`.

Available subtypes:
 - [`PrescribedMelting`](@ref)
 - [`FixedGamTMelting`](@ref)
 - [`TurbulentGamTMelting`](@ref)
 - [`UStarGamTMelting`](@ref)
 - [`TwoEquationMelting`](@ref)
"""
abstract type AbstractMelting end

# Warning for an experimental option, shared by the parameterization families.  One
# warning per session under the console logger, through `maxlog`, since rebuilding a
# parameter set calls the constructors again.
_warn_experimental(name, kind, default) = @warn(
    "$name is an experimental $kind; the default is $default.  " *
    "See the documentation for its assumptions.",
    maxlog = 1,
    _id = Symbol(name),
)

"""
$(TYPEDEF)

Prescribed melt rate, `ṁ = ṁ₀`, applied on ice-covered cells only
(gap cells never melt).

`melt` is in m yr⁻¹ of freshwater — the same units as the model's `melt` output — and
may be a scalar or a full-domain matrix the same size as the `mask` passed to
[`Grid`](@ref); `Model` crops a matrix alongside the mask.  Negative values freeze.

The melt still acts on the layer as in the three-equation model, so a prescribed
melt rate cools and freshens the plume consistently:

- the interface temperature is the local freezing point,
  `T_b = λ₁ S + λ₂ + λ₃ z_b`;
- the layer loses the heat that melting requires,
  `γ_T (T − T_b) = ṁ (L − c_i (T_i − T_b)) / c_p`, with the
  equivalent `γ_T` reported as `m.gamT`;
- the friction velocity `u⋆` is computed as usual, since entrainment needs it.

# Example

```jldoctest
julia> Params(; melting = PrescribedMelting(10.0)).melting   # uniform 10 m/yr
PrescribedMelting(melt = 10.0 m/yr)
```

A matrix the size of `mask`, `PrescribedMelting(melt_matrix)`, prescribes a 2D melt rate.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct PrescribedMelting{M} <: AbstractMelting
    "prescribed melt rate (m yr⁻¹ freshwater), a scalar or a matrix the size of `mask` (default `0`)"
    melt::M = 0.0
end

Base.show(io::IO, mp::PrescribedMelting) =
    print(io, "PrescribedMelting(melt = $(_range_str(mp.melt, 3)) m/yr)")

"""
$(TYPEDEF)

Three-equation ice–ocean melt parameterisation ([Jenkins 1991](@cite jenkins_onedimensional_1991); the equations are
given under [`TurbulentGamTMelting`](@ref)) with a constant turbulent heat transfer
coefficient `γ_T` and `γ_S = γ_T / 35` (the default).

# Example

```jldoctest
julia> Params(; melting = FixedGamTMelting(0.00018)).melting
FixedGamTMelting{Float64}(0.00018)
```

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct FixedGamTMelting{FT} <: AbstractMelting
    "heat transfer coefficient, dimensionless in the reference formulation (ISOMIP+ default `1.8e-4`)"
    gamTfix::FT = 0.00018
end

"""
$(TYPEDEF)

Three-equation ice–ocean melt parameterisation (the equations are given under
[`TurbulentGamTMelting`](@ref)) with transfer coefficients proportional to the friction
velocity ([Jenkins et al. 2010](@cite jenkins_observation_2010)):

```
γ_T = Γ_T u⋆,    γ_S = γ_T/r,
```

with the ratio ``r`` of the heat and salt transfer coefficients 35 by default.

This is LADDIE v2's `'uniform'` option, which is also what the ISOMIP+ protocol tunes
([Asay-Davis et al. 2016](@cite asay-davis_experimental_2016)).  [Jesse et al. (2026)](@cite jesse_ice-sheet_2026) use `Γ_T = 3 × 10⁻²`.

[Jenkins et al. (2010)](@cite jenkins_observation_2010) fit `C_d^(1/2) Γ_T = 1.1 × 10⁻³` and
`C_d^(1/2) Γ_S = 3.1 × 10⁻⁵` (`r` = 35) to observations beneath Ronne Ice
Shelf, with `C_d` = 0.0097.  Since `Γ` multiplies the `u⋆` of this model, which uses
`Params.C_d_top`, the equivalent is `Γ_T = 1.1 × 10⁻³/√(C_d^top)`, about
`3.3 × 10⁻²` for the default `C_d_top = 1.1e-3`.  The default
`Γ_T = 3 × 10⁻²` is already close.

A ratio other than 35 is experimental and warns when constructed.

# Example

```jldoctest
julia> Params(; melting = UStarGamTMelting(3.0e-2)).melting
UStarGamTMelting{Float64}(0.03, 35.0)
```

`UStarGamTMelting(3.0e-2, 70.0)` sets `Γ_T/Γ_S = 70`, an experimental ratio (with a warning).

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct UStarGamTMelting{FT} <: AbstractMelting
    "dimensionless heat transfer coefficient (default `3.0e-2`)"
    Gamma_T::FT = 3.0e-2
    "ratio of the heat to the salt transfer coefficient, `γ_T/γ_S` (default `35`)"
    ratio::FT = 35.0
    UStarGamTMelting{FT}(Gamma_T, ratio) where {FT} = new{FT}(Gamma_T, ratio)
end

function UStarGamTMelting(Gamma_T::FT, ratio::FT) where {FT<:Real}
    ratio > 0 ||
        throw(ArgumentError("UStarGamTMelting: ratio must be positive, got $ratio"))
    ratio == 35 || _warn_experimental("UStarGamTMelting(ratio = $ratio)", "heat-to-salt transfer ratio", "35")
    return UStarGamTMelting{FT}(Gamma_T, ratio)
end
UStarGamTMelting(Gamma_T::Real, ratio::Real) = UStarGamTMelting(promote(Gamma_T, ratio)...)
UStarGamTMelting(Gamma_T::FT) where {FT<:AbstractFloat} = UStarGamTMelting(Gamma_T, FT(35))

# Rebuilt unchecked: a zero AD tangent (`Enzyme.make_zero`) must pass.
_rebuild(::Type{<:UStarGamTMelting}, Gamma_T, ratio) =
    UStarGamTMelting{typeof(Gamma_T)}(Gamma_T, ratio)

"""
$(TYPEDEF)

Two-equation ice–ocean melt parameterisation ([McPhee 1992](@cite mcphee_turbulent_1992); [Jenkins et al. 2010](@cite jenkins_observation_2010), Eqs. 14–15).
The interface is at the freezing point of the layer salinity, ``T_b = T_f(S, z_b)``, so the salt
balance drops out, and the heat transfer coefficient `γ_TS = Γ_TS u⋆`
absorbs the effect of salt diffusion:

```
c_p γ_TS (T − T_b) = ṁ L + ṁ c_i (T_b − T_i) + q_cond
```

(`q_cond` and the `c_i` term from [`AbstractIceConduction`](@ref), the
Holland & Jenkins form by default.)  Unlike the three-equation form it is linear in `ṁ`,

```
ṁ = (c_p γ_TS (T − T_f) − q_cond)/(L − c_i (T_i − T_f)).
```

It is as good as the three-equation form at the one site where [Jenkins et al.](@cite jenkins_observation_2010) tested both,
and expected to be less so under strong thermal forcing, where the interface salinity departs
from the layer's: `Γ_T > Γ_TS` always ([Jenkins et al. 2010](@cite jenkins_observation_2010), Eq. 8).  The
buoyancy-flux entrainment reads the interface salinity ``S_b = S`` of this scheme, so the
buoyancy of the meltwater enters only through the thickness, not through the entrainment's
melt correction.  `gamS` is not used and stays zero.

[Jenkins et al. (2010)](@cite jenkins_observation_2010) fit `C_d^(1/2) Γ_TS = 5.9 × 10⁻⁴` (`Γ_TS` = 0.006
with `C_d` = 0.0097; [McPhee 1992](@cite mcphee_turbulent_1992)).  Since `Γ_TS` multiplies the `u⋆` of this model,
which uses `Params.C_d_top`, the equivalent is `5.9 × 10⁻⁴/√(C_d^top)`, which is
the default `0.0178` for `C_d_top = 1.1e-3`.  Change it along with `C_d_top`.

Select via `Params(; melting = TwoEquationMelting())`; warns when constructed.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct TwoEquationMelting{FT} <: AbstractMelting
    "dimensionless heat transfer coefficient (default `0.0178`)"
    Gamma_TS::FT = 0.0178
    TwoEquationMelting{FT}(Gamma_TS) where {FT} = new{FT}(Gamma_TS)
end

function TwoEquationMelting(Gamma_TS::FT) where {FT}
    Gamma_TS > 0 ||
        throw(ArgumentError("TwoEquationMelting: Gamma_TS must be positive, got $Gamma_TS"))
    _warn_experimental("TwoEquationMelting", "melt parameterisation", "FixedGamTMelting(); see the three-equation schemes")
    return TwoEquationMelting{FT}(Gamma_TS)
end

# Rebuilt unchecked: a zero AD tangent (`Enzyme.make_zero`) must pass.
_rebuild(::Type{<:TwoEquationMelting}, Gamma_TS) = TwoEquationMelting{typeof(Gamma_TS)}(Gamma_TS)

"""
$(TYPEDEF)

Three-equation melt parameterisation with turbulence-dependent transfer
coefficients `γ_T` and `γ_S` via the log-layer formulation
([Holland & Jenkins 1999](@cite holland_modeling_1999); [Lambert et al. 2023](@cite lambert_modelling_2023), Eqs. 11–12):

```
c_p γ_T (T − T_b) = ṁ L + ṁ c_i (T_b − T_i)
γ_S (S − S_b) = ṁ S_b
T_b = λ₁ S_b + λ₂ + λ₃ z_b
```

The turbulent heat exchange coefficients are determined by:

```
γ_T = u⋆/(2.12 log(u⋆ H/ν₀) + 12.5 Pr^(2/3) − 8.68)
γ_S = u⋆/(2.12 log(u⋆ H/ν₀) + 12.5 Sc^(2/3) − 8.68)
```

This results in a quadratic equation for the melt rate `ṁ`.

The log term is floored at zero, i.e. `u⋆ H/ν₀` is taken as at least 1.
Below that the layer is thinner than its own viscous sublayer, where the log law
does not hold, and the unbounded log would drive the denominator through zero and
give a negative or huge `γ`.  The floor never binds at the default
parameters: `Params.u_tide` and `Params.H_min` keep `u⋆ H/ν₀` above
about 170.  With the floor, both denominators stay positive when
`12.5 Pr^(2/3) > 8.68` (and likewise for `Sc`), which the
constructor checks.

# Example

```jldoctest
julia> Params(; melting = TurbulentGamTMelting(13.8, 2432.0, 1.95e-6)).melting
TurbulentGamTMelting{Float64}(13.8, 2432.0, 1.95e-6)
```

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct TurbulentGamTMelting{FT} <: AbstractMelting
    "Prandtl number (default `13.8`); must exceed `(8.68/12.5)^1.5 ≈ 0.58`"
    Pr::FT = 13.8
    "Schmidt number (default `2432.0`); same bound as `Pr`"
    Sc::FT = 2432.0
    "molecular kinematic viscosity (m² s⁻¹, default `1.95e-6`); must be positive"
    nu0::FT = 1.95e-6
    TurbulentGamTMelting{FT}(Pr, Sc, nu0) where {FT} = new{FT}(Pr, Sc, nu0)
end

function TurbulentGamTMelting(Pr::FT, Sc::FT, nu0::FT) where {FT}
    for (name, x) in (("Pr", Pr), ("Sc", Sc))
        _log_layer_offset(x) > 0 || throw(
            ArgumentError(
                "TurbulentGamTMelting: $name = $x makes the transfer-coefficient " *
                "denominator non-positive; it must exceed (8.68/12.5)^1.5 ≈ 0.58",
            ),
        )
    end
    nu0 > 0 ||
        throw(ArgumentError("TurbulentGamTMelting: nu0 must be positive, got $nu0"))
    return TurbulentGamTMelting{FT}(Pr, Sc, nu0)
end

# Rebuilt unchecked: a zero AD tangent (`Enzyme.make_zero`) must pass.
_rebuild(::Type{<:TurbulentGamTMelting}, Pr, Sc, nu0) =
    TurbulentGamTMelting{typeof(Pr)}(Pr, Sc, nu0)

# Constant part of the γ_T/γ_S denominator, 12.5·x^(2/3) − 8.68 for x = Pr or Sc.
function _log_layer_offset(x)
    FT = typeof(x)
    return FT(12.5) * x^(FT(2) / FT(3)) - FT(8.68)
end

# ============================================================================
# Melt rate: `update_melt!` and its kernels
# ============================================================================

# γ_T / γ_S, the fixed ratio of the heat and salt exchange velocities.
const _GAMMA_T_OVER_S = 35

# Three-equation melt parameterisation (Jenkins 1991) + interface temperature Tb.
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
    @Const(T_ice_farfield),
    gamT,
    gamS,
    c_p,
    c_i,
    L,
    l1,
    l2,
    l3,
    conduction,
)
    gamT, gamS, c_p, c_i, L, l1, l2, l3, conduction =
        _val(gamT), _val(gamS), _val(c_p), _val(c_i), _val(L), _val(l1), _val(l2), _val(l3), _val(conduction)
    i, j = @index(Global, NTuple)
    FT = typeof(c_p)
    @inbounds begin
        # Effective latent heat, per cell: L_eff = L - c_i*T_i.  Colder ice soaks up
        # more heat per unit melt, and the ice base need not be equally cold across
        # the domain, so this is read from the ice forcing rather than a constant.
        Tf_depth = l2 + l3 * z_draft[i, j]
        # The parameterisation for the conduction into the ice sets the c_i of this cell and a
        # fixed flux q.  The flux enters the heat balance as c_p·γT·(T − Tb) − q, so it
        # is the temperature T − q/(c_p·γT) that the equations below see.
        ci_eff, q_cond =
            _ice_conduction(conduction, c_i, T[i, j], Tf_depth + l1 * S[i, j], T_ice_farfield[i, j], z_draft[i, j])
        cp_over_Leff = c_p / (L - ci_eff * T_ice_farfield[i, j])
        ci_over_cp = ci_eff / c_p
        gT = _at(gamT, i, j)
        gS = _at(gamS, i, j)
        T_bal = T[i, j] - _safe_div(q_cond, c_p * gT)
        quad_b =
            cp_over_Leff * gT * (Tf_depth - T_bal) +
            gS * (one(FT) + cp_over_Leff * ci_over_cp * (Tf_depth + l1 * S[i, j]))
        quad_c = cp_over_Leff * gT * gS * (Tf_depth - T_bal + l1 * S[i, j])
        disc = quad_b * quad_b - FT(4) * quad_c
        melt_rate = (-quad_b + _ssqrt(conduction.kinks, disc, kink_width(conduction.kinks, :w))) / FT(2)
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
    kinks,
)
    PrCorr, ScCorr, nu0 = _val(PrCorr), _val(ScCorr), _val(nu0)
    i, j = @index(Global, NTuple)
    FT = typeof(nu0)
    @inbounds begin
        us = ustar[i, j]
        logterm = FT(2.12) * _smax0(kinks, log(us * H[i, j] / nu0 + FT(1e-12)), kink_width(kinks, :log))
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
        m.params.kinks,
    )
end

"""
$(TYPEDSIGNATURES)

Three-equation ice-ocean melt parameterisation with a fixed heat transfer
coefficient `γ_T` ([Jenkins 1991](@cite jenkins_onedimensional_1991); [Lambert et al. 2023](@cite lambert_modelling_2023), Eqs. 8–10 and 13).
Sets `m.ustar`, `m.melt`, `m.Tb`; `m.gamT` and `m.gamS` hold the constant
exchange velocities from the start.
"""
function update_melt!(m, ::FixedGamTMelting)
    update_ustar!(m)
    _launch_three_eq_melt!(m)
    return nothing
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
        m.T_ice_farfield,
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
    return nothing
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
        m.T_ice_farfield,
        m.melt_prescribed,
        m.c_p,
        m.c_i,
        m.L,
        m.l1,
        m.l2,
        m.l3,
        _conduction_args(m),
    )
    return nothing
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
    @Const(T_ice_farfield),
    @Const(melt_prescribed),
    c_p,
    c_i,
    L,
    l1,
    l2,
    l3,
    conduction,
)
    c_p, c_i, L, l1, l2, l3, conduction =
        _val(c_p), _val(c_i), _val(L), _val(l1), _val(l2), _val(l3), _val(conduction)
    i, j = @index(Global, NTuple)
    FT = typeof(c_p)
    @inbounds begin
        z = zero(FT)
        # Outside the domain everything is zero; in a gap cell (ocean, no ice) there is no
        # melt and Tb = T; under ice the prescribed rate sets Tb and γT.
        ocean = !iszero(tmask[i, j])
        ice = ocean & !iszero(imask[i, j])
        mdot = melt_prescribed[i, j]
        tb = _freezing_point(S[i, j], z_draft[i, j], l1, l2, l3)
        ci_eff, q_cond = _ice_conduction(conduction, c_i, T[i, j], tb, T_ice_farfield[i, j], z_draft[i, j])
        heat = (mdot * (L - ci_eff * (T_ice_farfield[i, j] - tb)) + q_cond) / c_p
        melt[i, j] = ifelse(ice, mdot, z)
        Tb[i, j] = ifelse(ice, tb, ifelse(ocean, T[i, j], z))
        gamT[i, j] = ifelse(ice, _safe_div(heat, T[i, j] - tb), z)
    end
end

"""
$(TYPEDSIGNATURES)

Three-equation ice-ocean melt parameterisation with turbulence-dependent
transfer coefficients `γ_T`, `γ_S` via the log-layer formulation
([Holland & Jenkins 1999](@cite holland_modeling_1999); [Lambert et al. 2023](@cite lambert_modelling_2023), Eqs. 8–13).
Sets `m.ustar`, `m.gamT`, `m.gamS`, `m.melt`, `m.Tb`.
"""
function update_melt!(m, mp::TurbulentGamTMelting)
    update_ustar!(m)
    _compute_turbulent_transfer_coefficients!(m, mp)
    _launch_three_eq_melt!(m)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Three-equation melt with transfer coefficients proportional to the friction velocity,
`γ_T = Γ_T u⋆` and `γ_S = γ_T/r`, with `r = mp.ratio` (35 by default).
Sets `m.ustar`, `m.gamT`, `m.gamS`, `m.melt`, `m.Tb`.
"""
function update_melt!(m, mp::UStarGamTMelting)
    update_ustar!(m)
    launch!(_ustar_gamma_kernel!, m.gamT, m.gamS, m.ustar, m.tmask, mp.Gamma_T, mp.ratio)
    _launch_three_eq_melt!(m)
    return nothing
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
        m.T_ice_farfield,
        m.gamT,
        m.c_p,
        m.c_i,
        m.L,
        m.l1,
        m.l2,
        m.l3,
        _conduction_args(m),
    )
    return nothing
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
    @Const(T_ice_farfield),
    @Const(gamT),
    c_p,
    c_i,
    L,
    l1,
    l2,
    l3,
    conduction,
)
    c_p, c_i, L, l1, l2, l3, conduction =
        _val(c_p), _val(c_i), _val(L), _val(l1), _val(l2), _val(l3), _val(conduction)
    i, j = @index(Global, NTuple)
    FT = typeof(c_p)
    @inbounds begin
        Tf = _freezing_point(S[i, j], z_draft[i, j], l1, l2, l3)
        ci_eff, q_cond = _ice_conduction(conduction, c_i, T[i, j], Tf, T_ice_farfield[i, j], z_draft[i, j])
        melt_rate = (c_p * gamT[i, j] * (T[i, j] - Tf) - q_cond) / (L - ci_eff * (T_ice_farfield[i, j] - Tf))
        melt[i, j] = iszero(imask[i, j]) ? zero(FT) : melt_rate
        Tb[i, j] = ifelse(iszero(tmask[i, j]), zero(FT), ifelse(iszero(imask[i, j]), T[i, j], Tf))
    end
end

update_melt!(m) = update_melt!(m, m.melting)
