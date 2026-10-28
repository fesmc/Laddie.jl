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

Prescribed melt rate, ``\\dot{m} = \\dot{m}_0``, applied on ice-covered cells only
(gap cells never melt).

`melt` is in m yr⁻¹ of freshwater — the same units as the model's `melt` output — and
may be a scalar or a full-domain matrix the same size as the `mask` passed to
[`Grid`](@ref); `Model` crops a matrix alongside the mask.  Negative values freeze.

The melt still acts on the layer as in the three-equation model, so a prescribed
melt rate cools and freshens the plume consistently:

- the interface temperature is the local freezing point,
  ``T_b = \\lambda_1 S + \\lambda_2 + \\lambda_3 z_b``;
- the layer loses the heat that melting requires,
  ``\\gamma_T (T - T_b) = \\dot{m}\\,(L - c_i (T_i - T_b)) / c_p``, with the
  equivalent ``\\gamma_T`` reported as `m.gamT`;
- the friction velocity ``u_\\star`` is computed as usual, since entrainment needs it.

# Example

```julia
Params(; melting = PrescribedMelting(10.0))          # uniform 10 m/yr
Params(; melting = PrescribedMelting(melt_matrix))   # 2D, same size as `mask`
```

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
coefficient ``\\gamma_T`` and ``\\gamma_S = \\gamma_T / 35`` (the default).

# Example

```julia
Params(; melting = FixedGamTMelting(0.00018))
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

```math
\\gamma_T = \\Gamma_T \\, u_\\star, \\qquad \\gamma_S = \\gamma_T / r ,
```

with the ratio ``r`` of the heat and salt transfer coefficients 35 by default.

This is LADDIE v2's `'uniform'` option, which is also what the ISOMIP+ protocol tunes
([Asay-Davis et al. 2016](@cite asay-davis_experimental_2016)).  [Jesse et al. (2026)](@cite jesse_ice-sheet_2026) use ``\\Gamma_T = 3 \\times 10^{-2}``.

[Jenkins et al. (2010)](@cite jenkins_observation_2010) fit ``C_d^{1/2}\\Gamma_T = 1.1 \\times 10^{-3}`` and
``C_d^{1/2}\\Gamma_S = 3.1 \\times 10^{-5}`` (``r`` = 35) to observations beneath Ronne Ice
Shelf, with ``C_d`` = 0.0097.  Since ``\\Gamma`` multiplies the ``u_\\star`` of this model, which uses
`Params.C_d_top`, the equivalent is ``\\Gamma_T = 1.1 \\times 10^{-3}/\\sqrt{C_d^\\text{top}}``, about
``3.3 \\times 10^{-2}`` for the default `C_d_top = 1.1e-3`.  The default
``\\Gamma_T = 3 \\times 10^{-2}`` is already close.

A ratio other than 35 is experimental and warns when constructed.

# Example

```julia
Params(; melting = UStarGamTMelting(3.0e-2))
Params(; melting = UStarGamTMelting(3.0e-2, 70.0))   # Γ_T / Γ_S = 70
```

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct UStarGamTMelting{FT} <: AbstractMelting
    "dimensionless heat transfer coefficient (default `3.0e-2`)"
    Gamma_T::FT = 3.0e-2
    "ratio of the heat to the salt transfer coefficient, ``\\gamma_T/\\gamma_S`` (default `35`)"
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
balance drops out, and the heat transfer coefficient ``\\gamma_{TS} = \\Gamma_{TS}\\,u_\\star``
absorbs the effect of salt diffusion:

```math
c_p\\,\\gamma_{TS}\\,(T - T_b) = \\dot m\\,L + \\dot m\\,c_i\\,(T_b - T_i) + q_\\text{cond}
```

(``q_\\text{cond}`` and the ``c_i`` term from [`AbstractIceConduction`](@ref), the
Holland & Jenkins form by default.)  Unlike the three-equation form it is linear in ``\\dot m``,

```math
\\dot m = \\frac{c_p\\,\\gamma_{TS}\\,(T - T_f) - q_\\text{cond}}{L - c_{i}\\,(T_i - T_f)} .
```

It is as good as the three-equation form at the one site where [Jenkins et al.](@cite jenkins_observation_2010) tested both,
and expected to be less so under strong thermal forcing, where the interface salinity departs
from the layer's: ``\\Gamma_T > \\Gamma_{TS}`` always ([Jenkins et al. 2010](@cite jenkins_observation_2010), Eq. 8).  The
buoyancy-flux entrainment reads the interface salinity ``S_b = S`` of this scheme, so the
buoyancy of the meltwater enters only through the thickness, not through the entrainment's
melt correction.  `gamS` is not used and stays zero.

[Jenkins et al. (2010)](@cite jenkins_observation_2010) fit ``C_d^{1/2}\\Gamma_{TS} = 5.9 \\times 10^{-4}`` (``\\Gamma_{TS}`` = 0.006
with ``C_d`` = 0.0097; [McPhee 1992](@cite mcphee_turbulent_1992)).  Since ``\\Gamma_{TS}`` multiplies the ``u_\\star`` of this model,
which uses `Params.C_d_top`, the equivalent is ``5.9 \\times 10^{-4}/\\sqrt{C_d^\\text{top}}``, which is
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
coefficients ``\\gamma_T`` and ``\\gamma_S`` via the log-layer formulation
([Holland & Jenkins 1999](@cite holland_modeling_1999); [Lambert et al. 2023](@cite lambert_modelling_2023), Eqs. 11–12):

```math
\\begin{aligned}
c_p \\, \\gamma_T \\,(T - T_b) &= \\dot{m} \\, L + \\dot{m} \\, c_i \\, (T_b - T_i) \\\\
\\gamma_S \\, (S - S_b) &= \\dot{m} \\, S_b \\\\
T_b &= \\lambda_1 \\, S_b + \\lambda_2 + \\lambda_3 \\, z_b
\\end{aligned}
```

The turbulent heat exchange coefficients are determined by:

```math
\\begin{aligned}
\\gamma_T &= \\frac{u_\\star}{2.12 \\, \\log \\frac{u_\\star H}{\\nu_0} + 12.5 \\, \\mathrm{Pr}^{2/3} - 8.68} \\\\
\\gamma_S &= \\frac{u_\\star}{2.12 \\, \\log \\frac{u_\\star H}{\\nu_0} + 12.5 \\, \\mathrm{Sc}^{2/3} - 8.68}
\\end{aligned}
```

This results in a quadratic equation for the melt rate ``\\dot{m}``.

The log term is floored at zero, i.e. ``u_\\star H/\\nu_0`` is taken as at least 1.
Below that the layer is thinner than its own viscous sublayer, where the log law
does not hold, and the unbounded log would drive the denominator through zero and
give a negative or huge ``\\gamma``.  The floor never binds at the default
parameters: `Params.u_tide` and `Params.H_min` keep ``u_\\star H/\\nu_0`` above
about 170.  With the floor, both denominators stay positive when
``12.5 \\, \\mathrm{Pr}^{2/3} > 8.68`` (and likewise for `Sc`), which the
constructor checks.

# Example

```julia
Params(; melting = TurbulentGamTMelting(13.8, 2432.0, 1.95e-6))
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
