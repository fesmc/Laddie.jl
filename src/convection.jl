"""
Abstract supertype for the treatment of convective instability: cells where the
layer is denser than the ambient water beneath it, i.e. where the reduced density
contrast ``\\delta\\rho`` falls below zero.  Pass a concrete instance as
`Params(; convection_scheme = ...)`: [`ResetToAmbient`](@ref) (the default),
[`ClampDensity`](@ref) or [`RelaxToAmbient`](@ref).
"""
abstract type AbstractConvectionScheme end

"""
$(TYPEDEF)

Handle convective instability (``\\delta\\rho < 0``) by clamping the density contrast to a
minimum positive value, `d_rho_min / Params.rho0_seawater`, so the plume remains lighter
than ambient.  T and S are left unchanged.  This is the buoyancy floor of LADDIE v2,
`max(drho_amb, buoyancy_min/rho_sw)` with `buoyancy_min = 5e-3`.

Select via `Params(; convection_scheme = ClampDensity(0.005))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct ClampDensity{FT} <: AbstractConvectionScheme
    "minimum density contrast ``\\rho_a - \\rho`` (kg m⁻³, default `0.005`); divided by `Params.rho0_seawater` to compare with ``\\delta\\rho``"
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
    "threshold density contrast ``\\rho_a - \\rho`` that triggers the reset (kg m⁻³, default `0.005`); divided by `Params.rho0_seawater` to compare with ``\\delta\\rho``"
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
