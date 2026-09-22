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
minimum positive value so the plume remains denser than ambient.

Select via `Params(; convection_scheme = ClampDensity(0.005))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct ClampDensity{FT} <: AbstractConvectionScheme
    "minimum density contrast (kg m⁻³, default `0.005`)"
    d_rho_min::FT = 0.005
end

"""
$(TYPEDEF)

Handle convective instability by instantly resetting T and S of unstable cells
to their ambient values, restoring a stable density contrast.

Select via `Params(; convection_scheme = ResetToAmbient(0.005))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct ResetToAmbient{FT} <: AbstractConvectionScheme
    "threshold density contrast that triggers the reset (kg m⁻³, default `0.005`)"
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
