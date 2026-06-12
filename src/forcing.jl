
# ============================================================================
# Forcing types — carry the computed T/S profiles on a uniform z-grid.
# AbstractVector fields (not Vector) allow the same struct to hold CPU or GPU
# arrays after to_backend!.
# Constructors for types that require raw parameters (LinearForcing, etc.) or
# file I/O (FileForcing) are defined in geometry.jl where the computation
# logic already lives.
# ============================================================================

abstract type AbstractForcing end

struct ISOMIPForcing{FT} <: AbstractForcing
    Tz::AbstractVector{FT}
    Sz::AbstractVector{FT}
    z::AbstractVector{FT}
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

Used automatically by `build_isomip`; select via `option = "isomip"` in `[Forcing]`.
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
    ISOMIPForcing{FT}(Tz, Sz, z, dz, z0, isomipcond)
end

"""
$(TYPEDSIGNATURES)

Ambient T/S profiles that vary linearly with depth from the surface
values (T at freezing, S = `S0`) to (`T1`, `S1`) at depth `z0`.

Select via `option = "linear"` in `[Forcing]`.
"""
struct LinearForcing{FT} <: AbstractForcing
    Tz::AbstractVector{FT};
    Sz::AbstractVector{FT};
    z::AbstractVector{FT};
    dz::FT;
    z0::FT
    S0::FT;
    S1::FT;
    T1::FT;
    forc_z0::FT
end

"""
$(TYPEDSIGNATURES)

Like `LinearForcing` but the profiles are capped at the surface value so they
do not extrapolate beyond (`T1`, `S1`) above `z0`.

Select via `option = "linear2"` in `[Forcing]`.
"""
struct Linear2Forcing{FT} <: AbstractForcing
    Tz::AbstractVector{FT};
    Sz::AbstractVector{FT};
    z::AbstractVector{FT};
    dz::FT;
    z0::FT
    S0::FT;
    S1::FT;
    T1::FT;
    forc_z0::FT
end

"""
$(TYPEDSIGNATURES)

Ambient T/S profiles with a tanh transition between cold surface waters and
warm deep waters, with an additional density perturbation `drho0·√|z|`.

Select via `option = "tanh"` in `[Forcing]`.
"""
struct TanhForcing{FT} <: AbstractForcing
    Tz::AbstractVector{FT};
    Sz::AbstractVector{FT};
    z::AbstractVector{FT};
    dz::FT;
    z0::FT
    S0::FT;
    T1::FT;
    forc_z0::FT;
    forc_z1::FT;
    drho0::FT
end

"""
$(TYPEDSIGNATURES)

Ambient T/S profiles loaded from a NetCDF file and resampled to a 1-m depth
grid.  Accepts either a single file with `z`, `T`, `S` variables, or two
separate files (one for T, one for S).

Select via `option = "file"` in `[Forcing]` and set `filename` (or
`filename_T` + `filename_S`).
"""
struct FileForcing{FT} <: AbstractForcing
    Tz::AbstractVector{FT};
    Sz::AbstractVector{FT};
    z::AbstractVector{FT};
    dz::FT;
    z0::FT
    path::String
end

struct ProfileForcing{FT} <: AbstractForcing
    Tz::AbstractVector{FT}
    Sz::AbstractVector{FT}
    z::AbstractVector{FT}
    dz::FT
    z0::FT
end
