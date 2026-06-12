
abstract type AbstractConvectionScheme end

"""
$(TYPEDSIGNATURES)

Handle convective instability (δρ < 0) by clamping the density contrast to a
minimum positive value so the plume remains denser than ambient.

- `mindrho`: minimum density contrast, kg m⁻³ (default `0.005`).

Select via `convop = 0` in `[Options]`; tune via `mindrho` in `[Convection]`.
"""
struct ClampDensity{FT} <: AbstractConvectionScheme
    ;
    mindrho::FT;
end

"""
$(TYPEDSIGNATURES)

Handle convective instability by instantly resetting T and S of unstable cells
to their ambient values, restoring a stable density contrast.

- `mindrho`: threshold density contrast triggering the reset, kg m⁻³ (default `0.005`).

Select via `convop = 1` in `[Options]`.
"""
struct ResetToAmbient{FT} <: AbstractConvectionScheme
    ;
    mindrho::FT;
end

"""
$(TYPEDSIGNATURES)

Handle convective instability by relaxing T and S of unstable cells toward
ambient values over a prescribed timescale (applied implicitly in the tracer
time step via the `conv2` term).

- `convtime`: relaxation timescale, s (default `10000.0`).

Select via `convop = 2` in `[Options]`; tune via `convtime` in `[Convection]`.
"""
struct RelaxToAmbient{FT} <: AbstractConvectionScheme
    ;
    convtime::FT;
end
