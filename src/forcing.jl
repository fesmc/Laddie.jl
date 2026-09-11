# ============================================================================
# Forcing types.
#
# A `CavityForcing` bundles the two external drivers of the meltwater layer:
#   - `ocean` — the ambient T/S the layer entrains and exchanges heat with
#   - `ice`   — the state of the ice above it; currently the basal ice
#               temperature, which sets the effective latent heat of melting
#
# Ocean profiles are carried on a uniform z-grid.  The concrete vector type is a
# type parameter V (Vector{FT} on CPU, CuArray{FT,1} on GPU after to_backend),
# keeping field access type-stable.  The resampling `OceanForcing1D` constructor
# (requires interpolation logic) is in geometry.jl.
# ============================================================================

#########################
# Ocean
#########################

"""
Supertype of every ocean forcing: the ambient temperature and salinity the
meltwater layer entrains and exchanges heat with.
"""
abstract type AbstractOceanForcing end

"""
$(TYPEDEF)

Ambient T/S as one vertical profile resampled onto a uniform 1 m z-grid.

Construct it from arbitrary samples with
[`OceanForcing1D(Tz, Sz, z)`](@ref OceanForcing1D), which sorts by depth,
resamples, and extrapolates flat beyond the data range.
"""
struct OceanForcing1D{FT,V<:AbstractVector{FT}} <: AbstractOceanForcing
    Tz::V
    Sz::V
    z::V
    dz::FT
    z0::FT
end

"""
$(TYPEDSIGNATURES)

Ambient T/S profiles for the ISOMIP+ protocol (Asay-Davis et al. 2016), returned
as an [`OceanForcing1D`](@ref).  Profiles are linear from the surface
(T = −1.9 °C, S = 33.8 psu) to 720 m depth.

- `isomipcond = :warm`: T = +1.0 °C, S = 34.7 psu at depth (strong melting).
- `isomipcond = :cold`: T = −1.9 °C, S = 34.55 psu at depth (near-freezing).

Used automatically by `build_isomip`; pass it as the `forcing` argument of
`Model` to use it with another geometry.
"""
function ISOMIPForcing(FT::Type, isomipcond::Symbol)
    isomipcond in (:warm, :cold) || throw(
        ArgumentError("isomipcond must be :warm or :cold, got :$isomipcond"),
    )
    z = FT.(-5000.0:1.0:-1.0)
    dz = FT(1.0)
    z0 = z[1]
    z_pyc = FT(-720.0)
    T_surface = FT(-1.9)
    S_surface = FT(33.8)
    T_deep, S_deep = isomipcond == :warm ? (FT(1.0), FT(34.7)) : (FT(-1.9), FT(34.55))
    Tz = @. T_surface + z * (T_deep - T_surface) / z_pyc
    Sz = @. S_surface + z * (S_deep - S_surface) / z_pyc
    OceanForcing1D(Tz, Sz, z, dz, z0)
end

"""
$(TYPEDEF)

Placeholder for a laterally varying ambient field (one profile per cell).
Not implemented: `Model` accepts only [`AbstractOceanForcing`](@ref), so
passing this raises a `MethodError` rather than silently ignoring the variation.
"""
struct OceanForcing2D <: AbstractOceanForcing end

#########################
# Ice
#########################

"""
Supertype of every ice forcing: the state of the ice above the layer.  Subtypes
must provide `T_ice_base`, the temperature of the ice at its base (°C, ≤ 0),
which sets the effective latent heat ``L_\\text{eff} = L - c_i T_i`` in the
three-equation melt solve.
"""
abstract type AbstractIceForcing end

"""
$(TYPEDEF)

Basal ice temperature held fixed for the whole run.

`T_ice_base` may be given as a scalar (uniform ice, the common standalone case)
or as a full-domain matrix the same size as `mask`, letting a colder interior sit
next to a temperate ice front.  `Model` converts either form to an `FT` matrix on
the model grid, cropping it alongside the mask, so `m.T_ice_base` is always a
grid-shaped field.

Colder ice absorbs more heat per unit melt, so a lower `T_ice_base` means less
melt for the same thermal driving: `L_eff` runs from 3.34e5 J kg⁻¹ at 0 °C to
3.84e5 at −25 °C, a 15 % spread.

```julia
PrescribedIceForcing(-25.0)          # uniform, the default
PrescribedIceForcing(T_ice_matrix)   # 2D, same size as `mask`
```
"""
struct PrescribedIceForcing{M} <: AbstractIceForcing
    T_ice_base::M
end

# Default basal ice temperature, applied when a bare ocean forcing is passed to
# `Model`.  Matches the historical `Params.T_i` default, so runs that never
# mention an ice forcing are unchanged.
const DEFAULT_T_ICE_BASE = -25.0

PrescribedIceForcing() = PrescribedIceForcing(DEFAULT_T_ICE_BASE)

########################
# Forcing
########################

"""
$(TYPEDEF)

The complete external forcing of a cavity: an ocean forcing and an ice forcing.

`Model` takes one of these. Passing a bare ocean forcing is shorthand for
`CavityForcing(ocean)`, which pairs it with a uniform
`PrescribedIceForcing($(DEFAULT_T_ICE_BASE))`.

```julia
CavityForcing(ISOMIPForcing(Float64, :warm))                      # default ice
CavityForcing(ocean, PrescribedIceForcing(T_ice_matrix))          # 2D ice
```
"""
struct CavityForcing{O<:AbstractOceanForcing,I<:AbstractIceForcing}
    ocean::O
    ice::I
end

CavityForcing(ocean::AbstractOceanForcing) =
    CavityForcing(ocean, PrescribedIceForcing())

# `Model` accepts either form; a bare ocean forcing picks up the default ice.
_as_cavity_forcing(f::CavityForcing) = f
_as_cavity_forcing(f::AbstractOceanForcing) = CavityForcing(f)
_as_cavity_forcing(f) = throw(
    ArgumentError(
        "forcing must be a CavityForcing or an AbstractOceanForcing, got $(typeof(f))",
    ),
)
