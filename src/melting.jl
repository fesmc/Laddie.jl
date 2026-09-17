"""
Abstract supertype for the ice–ocean melt parameterisation.  Pass a concrete
instance as `Params(; melting = ...)`.

Available subtypes:
 - [`PrescribedMelting`](@ref)
 - [`FixedGamTMelting`](@ref)
 - [`TurbulentGamTMelting`](@ref)
"""
abstract type AbstractMelting end

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
 - `melt` — prescribed melt rate (m yr⁻¹ freshwater), scalar or matrix (default `0`).
"""
@kwdef struct PrescribedMelting{M} <: AbstractMelting
    melt::M = 0.0
end

Base.show(io::IO, mp::PrescribedMelting) =
    print(io, "PrescribedMelting(melt = $(_range_str(mp.melt, 3)) m/yr)")

"""
$(TYPEDEF)

Three-equation ice–ocean melt parameterisation (Jenkins 1991; the equations are
given under [`TurbulentGamTMelting`](@ref)) with a constant turbulent heat transfer
coefficient ``\\gamma_T`` and ``\\gamma_S = \\gamma_T / 35`` (the default).

# Example

```julia
Params(; melting = FixedGamTMelting(0.00018))
```

# Fields
 - `gamTfix`: heat transfer coefficient, dimensionless in the reference's
   formulation (ISOMIP+ default: `1.8e-4`).

"""
@kwdef struct FixedGamTMelting{FT} <: AbstractMelting
    gamTfix::FT = 0.00018
end

"""
$(TYPEDEF)

Three-equation melt parameterisation with turbulence-dependent transfer
coefficients ``\\gamma_T`` and ``\\gamma_S`` via the log-layer formulation
(Holland & Jenkins 1999; Lambert et al. 2023, Eqs. 11–12):

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
\\gamma_T &= \\frac{u_\\star}{2.12 \\, \\log \\frac{u_\\star D}{\\nu_0} + 12.5 \\, \\mathrm{Pr}^{2/3} - 8.68} \\\\
\\gamma_S &= \\frac{u_\\star}{2.12 \\, \\log \\frac{u_\\star D}{\\nu_0} + 12.5 \\, \\mathrm{Sc}^{2/3} - 8.68}
\\end{aligned}
```

This results in a quadratic equation for the melt rate ``\\dot{m}``.

# Example

```julia
Params(; melting = TurbulentGamTMelting(13.8, 2432.0, 1.95e-6))
```

# Fields
 - `Pr`:  Prandtl number (default `13.8`).
 - `Sc`:  Schmidt number (default `2432.0`).
 - `nu0`: molecular kinematic viscosity, m² s⁻¹ (default `1.95e-6`).

"""
@kwdef struct TurbulentGamTMelting{FT} <: AbstractMelting
    Pr::FT = 13.8
    Sc::FT = 2432.0
    nu0::FT = 1.95e-6
end
