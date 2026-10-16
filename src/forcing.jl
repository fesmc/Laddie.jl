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

# Fields
$(TYPEDFIELDS)
"""
struct OceanForcing1D{FT,V<:AbstractVector} <: AbstractOceanForcing
    "ambient temperature on the uniform z-grid (°C)"
    Tz::V
    "ambient salinity on the uniform z-grid (psu)"
    Sz::V
    "depths of the uniform z-grid (m, negative downward)"
    z::V
    "z-grid spacing (m)"
    dz::FT
    "depth of the first z-grid point (m)"
    z0::FT
end

"""
Ambient T/S profiles for the ISOMIP+ protocol (Asay-Davis et al. 2016), returned
as an [`OceanForcing1D`](@ref).  Profiles are linear from the surface
(T = −1.9 °C, S = 33.8 psu) to 720 m depth.

- `isomipcond = :warm`: T = +1.0 °C, S = 34.7 psu at depth (strong melting).
- `isomipcond = :cold`: T = −1.9 °C, S = 34.55 psu at depth (near-freezing).

Used automatically by `build_isomip`; pass it as the `forcing` argument of
`Model` to use it with another geometry.

```julia
ISOMIPForcing(:cold; FT = Float32)
```
"""
function ISOMIPForcing(isomipcond::Symbol = :warm; FT = Float64)
    isomipcond in (:warm, :cold) ||
        throw(ArgumentError("isomipcond must be :warm or :cold, got :$isomipcond"))
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

Laterally varying ambient T/S: one vertical profile per grid cell, all on the same
uniform z-grid.  `Tz` and `Sz` are `(nx, ny, nz)` arrays covering the full domain
including the border ring (like a 2D `T_ice_base`); `Model` crops them with the
grid.  Each cell samples its own column at the depth of its plume base, and depths
outside the grid take the nearest level, as for [`OceanForcing1D`](@ref).

Not exported, and not yet supported by the Reactant backend.

# Fields
$(TYPEDFIELDS)
"""
struct OceanForcing2D{FT,A<:AbstractArray{FT,3},V<:AbstractVector{FT}} <: AbstractOceanForcing
    "ambient temperature per cell and level (°C)"
    Tz::A
    "ambient salinity per cell and level (psu)"
    Sz::A
    "depths of the uniform z-grid (m, negative downward, ascending)"
    z::V
    "z-grid spacing (m)"
    dz::FT
    "depth of the first z-grid point (m)"
    z0::FT
end

"""
    OceanForcing2D(Tz, Sz, z; FT = Float64)

Per-cell profiles `Tz`, `Sz` of size `(nx, ny, nz)` on the depths `z` (m, negative
downward), which must be uniformly spaced; they are sorted ascending.
"""
function OceanForcing2D(Tz::AbstractArray{<:Real,3}, Sz::AbstractArray{<:Real,3}, z::AbstractVector; FT = Float64)
    size(Tz) == size(Sz) || throw(ArgumentError("Tz is $(size(Tz)) but Sz is $(size(Sz))"))
    size(Tz, 3) == length(z) ||
        throw(ArgumentError("Tz has $(size(Tz, 3)) levels but z has $(length(z))"))
    length(z) ≥ 2 || throw(ArgumentError("need at least two levels"))
    p = sortperm(z)
    zs = Float64.(z[p])
    dzs = diff(zs)
    all(d -> isapprox(d, dzs[1]; rtol = 1e-6), dzs) ||
        throw(ArgumentError("the levels of an OceanForcing2D must be uniformly spaced"))
    T, S = FT.(Tz[:, :, p]), FT.(Sz[:, :, p])
    all(isfinite, T) && all(isfinite, S) ||
        throw(ArgumentError("Tz and Sz must be finite everywhere (fill the gaps first)"))
    OceanForcing2D(T, S, FT.(zs), FT(dzs[1]), FT(zs[1]))
end

Base.show(io::IO, f::OceanForcing2D) = print(
    io,
    "OceanForcing2D{$(eltype(f.Tz))}: $(size(f.Tz, 1))×$(size(f.Tz, 2)) profiles of ",
    "$(length(f.z)) levels, z ∈ [$(join(extrema(f.z), ", "))] m, ",
    "T ∈ [$(join(_r.(extrema(f.Tz), 3), ", "))] °C, S ∈ [$(join(_r.(extrema(f.Sz), 3), ", "))] psu",
)

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

# Fields
$(TYPEDFIELDS)
"""
struct PrescribedIceForcing{M} <: AbstractIceForcing
    "basal ice temperature (°C): a scalar, or a full-domain matrix the size of `mask`"
    T_ice_base::M
end

# Default basal ice temperature, applied when a bare ocean forcing is passed to
# `Model` (the ISOMIP+ value).
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
CavityForcing(ISOMIPForcing(:warm))                                 # default ice
CavityForcing(ocean, PrescribedIceForcing(T_ice_matrix))          # 2D ice
```

# Fields
$(TYPEDFIELDS)
"""
struct CavityForcing{O<:AbstractOceanForcing,I<:AbstractIceForcing}
    "the ocean forcing (ambient temperature and salinity)"
    ocean::O
    "the ice forcing (basal ice temperature)"
    ice::I
end

CavityForcing(ocean::AbstractOceanForcing) = CavityForcing(ocean, PrescribedIceForcing())

# `Model` accepts either form; a bare ocean forcing picks up the default ice.
_as_cavity_forcing(f::CavityForcing) = f
_as_cavity_forcing(f::AbstractOceanForcing) = CavityForcing(f)
_as_cavity_forcing(f) = throw(
    ArgumentError(
        "forcing must be a CavityForcing or an AbstractOceanForcing, got $(typeof(f))",
    ),
)
