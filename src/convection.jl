
abstract type AbstractConvectionScheme end

"""
$(TYPEDSIGNATURES)

Handle convective instability (``\\delta\\rho < 0``) by clamping the density contrast to a
minimum positive value so the plume remains denser than ambient.

- `d_rho_min`: minimum density contrast, kg m⁻³ (default `0.005`).

Select via `Params(; convection_scheme = ClampDensity(0.005))`.
"""
@kwdef struct ClampDensity{FT} <: AbstractConvectionScheme
    d_rho_min::FT = 0.005
end

"""
$(TYPEDSIGNATURES)

Handle convective instability by instantly resetting T and S of unstable cells
to their ambient values, restoring a stable density contrast.

- `d_rho_min`: threshold density contrast triggering the reset, kg m⁻³ (default `0.005`).

Select via `Params(; convection_scheme = ResetToAmbient(0.005))`.
"""
@kwdef struct ResetToAmbient{FT} <: AbstractConvectionScheme
    d_rho_min::FT = 0.005
end

"""
$(TYPEDSIGNATURES)

Handle convective instability by relaxing T and S of unstable cells toward
ambient values over a prescribed timescale (applied implicitly in the tracer
time step via the `conv2` term).

- `convection_time`: relaxation timescale, s (default `10000.0`).

Select via `Params(; convection_scheme = RelaxToAmbient(10000.0))`.
"""
@kwdef struct RelaxToAmbient{FT} <: AbstractConvectionScheme
    convection_time::FT = 10000.0
end
