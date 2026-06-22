
# ============================================================================
# Forcing types — carry the computed T/S profiles on a uniform z-grid.
# The concrete vector type is a type parameter V (Vector{FT} on CPU,
# CuArray{FT,1} on GPU after to_backend), keeping field access type-stable.
# The ProfileForcing constructor (requires interpolation logic) is in geometry.jl.
# ============================================================================

abstract type AbstractForcing end

struct ISOMIPForcing{FT,V<:AbstractVector{FT}} <: AbstractForcing
    Tz::V
    Sz::V
    z::V
    dz::FT
    z0::FT
    isomipcond::Symbol
end

"""
$(TYPEDSIGNATURES)

Ambient T/S profiles for the ISOMIP+ protocol (Asay-Davis et al. 2016).
Profiles are linear from the surface (T = −1.9 °C, S = 33.8 psu) to 720 m depth.

- `isomipcond = :warm`: T = +1.0 °C, S = 34.7 psu at depth (strong melting).
- `isomipcond = :cold`: T = −1.9 °C, S = 34.55 psu at depth (near-freezing).

Used automatically by `build_isomip`; pass it as the `forcing` argument of
`Model` to use it with another geometry.
"""
function ISOMIPForcing(FT::Type, isomipcond::Symbol)
    z = FT.(-5000.0:1.0:-1.0)
    dz = FT(1.0)
    z0 = z[1]
    z_pyc = FT(-720.0)
    T_surface = FT(-1.9)
    S_surface = FT(33.8)
    T_deep, S_deep = isomipcond == :warm ? (FT(1.0), FT(34.7)) : (FT(-1.9), FT(34.55))
    Tz = @. T_surface + z * (T_deep - T_surface) / z_pyc
    Sz = @. S_surface + z * (S_deep - S_surface) / z_pyc
    ISOMIPForcing(Tz, Sz, z, dz, z0, isomipcond)
end


struct ProfileForcing{FT,V<:AbstractVector{FT}} <: AbstractForcing
    Tz::V
    Sz::V
    z::V
    dz::FT
    z0::FT
end
