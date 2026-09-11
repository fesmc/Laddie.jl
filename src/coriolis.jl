# ============================================================================
# Coriolis parameter — how `f` is set over the domain.
#
# `f` enters the momentum equations only as the scalar in -f k̂ × u, which is
# invariant under rotation of the horizontal axes.  So the grid's orientation
# relative to true north never matters; only latitude does, and only through
# f = 2Ω sin(φ).  That is why this is an `AbstractCoriolisParameter` and not a
# north-direction vector.
# ============================================================================

"""
Supertype of the Coriolis-parameter options.  Select with
`Params(; coriolis = ...)`; see [`CoriolisParameter0D`](@ref) (an f-plane) and
[`CoriolisParameter2D`](@ref) (f from latitude).
"""
abstract type AbstractCoriolisParameter end

"Earth's rotation rate Ω [rad s⁻¹]."
const EARTH_ROTATION_RATE = 7.2921e-5

"""
Default f-plane value [s⁻¹], the ISOMIP+/UFEMISM canonical
`uniform_laddie_coriolis_parameter`.
"""
const DEFAULT_CORIOLIS_F = -1.37e-4

"""
Latitude [°N] that reproduces [`DEFAULT_CORIOLIS_F`](@ref) through
``f = 2Ω\\sin φ`` — about 69.95 °S.

Derived from `DEFAULT_CORIOLIS_F` rather than written as a rounded −70.0 so that
`CoriolisParameter2D()` and `CoriolisParameter0D()` agree to the last bit; a
rounded value would leave the two defaults 0.1 % apart for no reason.
"""
const DEFAULT_LATITUDE = asind(DEFAULT_CORIOLIS_F / (2 * EARTH_ROTATION_RATE))

"""
$(TYPEDEF)

f-plane: one Coriolis parameter over the whole domain, given directly.

The default is the ISOMIP+ value ``f = -1.37\\times10^{-4}`` s⁻¹, which
corresponds to 69.95 °S.

Select via `Params(; coriolis = CoriolisParameter0D(-1.4e-4))`.
"""
@kwdef struct CoriolisParameter0D{FT} <: AbstractCoriolisParameter
    f::FT = DEFAULT_CORIOLIS_F
end

"""
$(TYPEDEF)

Coriolis parameter from latitude, ``f = 2Ω\\sin φ``.

`lat` is in degrees north (negative in the Southern Hemisphere) and may be

- a scalar — still an f-plane, but stated geographically rather than as a raw
  frequency. The default reproduces [`CoriolisParameter0D`](@ref)'s default
  exactly, so swapping one for the other changes nothing until `lat` is set.
- a full-domain matrix the same size as `mask` — a genuine β-plane. `Model`
  crops it alongside the mask, so `m.f` is always a grid-shaped field.

Across the Antarctic shelves `f` runs from −1.32e-4 at 65 °S to −1.44e-4 at
80 °S, roughly a 9 % spread; the Rossby radius scales as 1/f, so a large domain
feels it even though the range looks narrow.

```julia
Params(; coriolis = CoriolisParameter2D(-75.0))      # f-plane at 75°S
Params(; coriolis = CoriolisParameter2D(lat_matrix)) # varying with latitude
```
"""
@kwdef struct CoriolisParameter2D{L} <: AbstractCoriolisParameter
    lat::L = DEFAULT_LATITUDE
end

"Coriolis parameter from latitude in degrees north."
_f_from_lat(lat, FT) = FT(2 * EARTH_ROTATION_RATE * sind(Float64(lat)))

# Expand a Coriolis choice into a full-domain field on T-points.  Runs before
# cropping, on the same footing as `z_draft_raw`, so a 2D latitude is validated
# against the mask the caller passed and then sliced with it.
_coriolis_field(cp::CoriolisParameter0D, sz, FT) = fill(FT(cp.f), sz)

function _coriolis_field(cp::CoriolisParameter2D, sz, FT)
    lat = cp.lat
    if lat isa AbstractMatrix
        size(lat) == sz || throw(
            ArgumentError(
                "coriolis latitude is $(size(lat)) but the mask is $sz; a 2D latitude " *
                "must cover the full domain including the border ring",
            ),
        )
        all(l -> -90 <= l <= 90, lat) || throw(
            ArgumentError("coriolis latitude must lie in [-90, 90] degrees north"),
        )
        return FT[_f_from_lat(l, FT) for l in lat]
    end
    lat isa Real || throw(
        ArgumentError("coriolis latitude must be a real scalar or a matrix, got $(typeof(lat))"),
    )
    -90 <= lat <= 90 || throw(
        ArgumentError("coriolis latitude must lie in [-90, 90] degrees north, got $lat"),
    )
    return fill(_f_from_lat(lat, FT), sz)
end

Base.show(io::IO, cp::CoriolisParameter0D) = print(io, "CoriolisParameter0D(f = ", cp.f, ")")

function Base.show(io::IO, cp::CoriolisParameter2D)
    print(io, "CoriolisParameter2D(lat = ")
    if cp.lat isa AbstractMatrix
        lo, hi = extrema(cp.lat)
        print(io, round(Float64(lo); digits = 2), " … ", round(Float64(hi); digits = 2))
    else
        print(io, round(Float64(cp.lat); digits = 2))
    end
    print(io, "°N)")
end
