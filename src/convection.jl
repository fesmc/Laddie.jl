"""
Abstract supertype for the treatment of convective instability: cells where the
layer is denser than the ambient water beneath it, i.e. where the reduced density
contrast `δρ` falls below zero.  Pass a concrete instance as
`Params(; convection_scheme = ...)`: [`ResetToAmbient`](@ref) (the default),
[`ClampDensity`](@ref) or [`RelaxToAmbient`](@ref).

The layer is unstable where `δρ = (ρ_a − ρ)/ρ₀ < 0`. No LADDIE paper describes a convection
closure, so the schemes follow the reference codes: `ResetToAmbient` and `RelaxToAmbient` the
LADDIE v1 code of [Lambert et al. (2023)](@cite lambert_modelling_2023), `ClampDensity` the
buoyancy floor of the LADDIE v2 code of [Lambert et al. (2026)](@cite lambert_one-layer_2026).
`RelaxToAmbient` adds the tracer sink `−(q − q_a) conv2`, `conv2 = [δρ < 0] H/convection_time`.
"""
abstract type AbstractConvectionScheme end

"""
$(TYPEDEF)

Handle convective instability (`δρ < 0`) by clamping the density contrast to a
minimum positive value, `d_rho_min / Params.rho0_seawater`, so the plume remains lighter
than ambient.  T and S are left unchanged.  This is the buoyancy floor of LADDIE v2,
`max(drho_amb, buoyancy_min/rho_sw)` with `buoyancy_min = 5e-3`.

Select via `Params(; convection_scheme = ClampDensity(0.005))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct ClampDensity{FT} <: AbstractConvectionScheme
    "minimum density contrast `ρ_a − ρ` (kg m⁻³, default `0.005`); divided by `Params.rho0_seawater` to compare with `δρ`"
    d_rho_min::FT = 0.005
end

"""
$(TYPEDEF)

Handle convective instability by instantly resetting T and S to their ambient values
wherever the contrast falls below `d_rho_min / Params.rho0_seawater`, with S lowered by
`d_rho_min / (rho0_seawater * beta)` so that the reset cell has exactly that contrast.

Select via `Params(; convection_scheme = ResetToAmbient(0.005))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct ResetToAmbient{FT} <: AbstractConvectionScheme
    "threshold density contrast `ρ_a − ρ` that triggers the reset (kg m⁻³, default `0.005`); divided by `Params.rho0_seawater` to compare with `δρ`"
    d_rho_min::FT = 0.005
end

"""
$(TYPEDEF)

Handle convective instability by relaxing T and S of unstable cells toward
ambient values over a prescribed timescale (applied implicitly in the tracer
time step via the `conv2` term).

Select via `Params(; convection_scheme = RelaxToAmbient(10000.0))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct RelaxToAmbient{FT} <: AbstractConvectionScheme
    "relaxation timescale (s, default `10000.0`)"
    convection_time::FT = 10000.0
end
