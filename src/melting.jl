"""
$(TYPEDSIGNATURES)

An abstract type for the ice–ocean melt parameterisation.

Available subtypes:
 - [`PrescribedMelting`](@ref)
 - [`FixedGamTMelting`](@ref)
 - [`TurbulentGamTMelting`](@ref)
"""
abstract type AbstractMelting end

"""
$(TYPEDSIGNATURES)

Prescribed melt rate:

```math
\\dot{m} = \\dot{m}_0
```

# Example

```julia
Params(; melting = PrescribedMelting(0.1))
```

# Fields
 - `melt` — prescribed melt rate (m yr⁻¹).
"""
@kwdef struct PrescribedMelting{FT} <: AbstractMelting
    melt::FT = zero(FT)
end

"""
$(TYPEDSIGNATURES)

Three-equation ice–ocean melt parameterisation (Jenkins 1991) with a constant
turbulent heat transfer coefficient ``\\gamma_T``:

# Example

```julia
Params(; melting = FixedGamTMelting(0.00018))
```

# Fields
 - `gamTfix`: heat transfer coefficient (ISOMIP+ default: `1.8e-4`).

"""
@kwdef struct FixedGamTMelting{FT} <: AbstractMelting
    gamTfix::FT = 0.00018
end

"""
$(TYPEDSIGNATURES)

Three-equation melt parameterisation with turbulence-dependent transfer
coefficients ``\\gamma_T`` and ``\\gamma_S`` via the log-layer formulation
(Holland & Jenkins 1999; Lambert et al. 2023, Eqs. 11–12):

```math
c_p \\, \\gamma_T \\,(T - T_b) = \\dot{m} \\, L + \\dot{m} \\, c_i \\, (T_b - T_i) \\\
\\gamma_S \\, (S - S_b) = \\dot{m} \\, S_b \\\
T_b = \\lambda_1 \\, S_b + \\lambda_2 + \\lambda_3 \\, z_b
```

The turbulent heat exchange coefficients are determined by:

```math
\\gamma_T = \\frac{u_\\star}{2.12 \\, \\mathrm{log} \\frac{u_\\star D}{\\nu_0} + 12.5 \\, \\mathrm{Pr}^{2/3} - 8.68} \\\
\\gamma_S = \\frac{u_\\star}{2.12 \\, \\mathrm{log} \\frac{u_\\star D}{\\nu_0} + 12.5 \\, \\mathrm{Sc}^{2/3} - 8.68}
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