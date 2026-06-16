
abstract type AbstractMeltParam end

"""
$(TYPEDSIGNATURES)

Three-equation ice–ocean melt parameterisation (Jenkins 1991) with a constant
turbulent heat transfer coefficient γ_T.

- `gamTfix`: heat transfer coefficient (ISOMIP+ default: `1.8e-4`).

Select via `Params(; meltpar = FixedGamT(0.00018))`.
"""
struct FixedGamT{FT} <: AbstractMeltParam
    ;
    gamTfix::FT;
end

"""
$(TYPEDSIGNATURES)

Three-equation melt parameterisation with turbulence-dependent transfer
coefficients γ_T and γ_S via the log-layer formulation
(Holland & Jenkins 1999; Lambert et al. 2023, Eqs. 11–12).

- `Pr`:  Prandtl number (default `13.8`).
- `Sc`:  Schmidt number (default `2432.0`).
- `nu0`: molecular kinematic viscosity, m² s⁻¹ (default `1.95e-6`).

Select via `Params(; meltpar = TurbulentGamT(13.8, 2432.0, 1.95e-6))`.
"""
struct TurbulentGamT{FT} <: AbstractMeltParam
    ;
    Pr::FT;
    Sc::FT;
    nu0::FT;
end