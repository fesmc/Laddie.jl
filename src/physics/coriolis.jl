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

`f` enters the momentum equations as `−f k̂ × u`.
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
`f = 2Ω sin φ` — about 69.95 °S.

Derived rather than rounded to −70.0, so that the defaults of `CoriolisParameter2D` and
`CoriolisParameter0D` agree to the last bit.
"""
const DEFAULT_LATITUDE = asind(DEFAULT_CORIOLIS_F / (2 * EARTH_ROTATION_RATE))

"""
$(TYPEDEF)

f-plane: one Coriolis parameter over the whole domain, given directly.

The default is the ISOMIP+ value `f = −1.37×10⁻⁴` s⁻¹, which
corresponds to 69.95 °S.

Select via `Params(; coriolis = CoriolisParameter0D(-1.4e-4))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct CoriolisParameter0D{FT} <: AbstractCoriolisParameter
    "Coriolis parameter (s⁻¹, default the ISOMIP+ value `-1.37e-4`)"
    f::FT = DEFAULT_CORIOLIS_F
end

"""
$(TYPEDEF)

Coriolis parameter from latitude, `f = 2Ω sin φ`.

`lat` (°N, negative in the Southern Hemisphere) is a scalar, an f-plane whose default equals
[`CoriolisParameter0D`](@ref)'s, or a matrix the size of `mask` (a β-plane), which `Model`
crops with the grid.

```jldoctest
julia> Params(; coriolis = CoriolisParameter2D(-75.0)).coriolis   # f-plane at 75°S
CoriolisParameter2D(lat = -75.0°N)
```

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct CoriolisParameter2D{L} <: AbstractCoriolisParameter
    "latitude in degrees north: a scalar (f-plane) or a full-domain matrix the size of `mask`"
    lat::L = DEFAULT_LATITUDE
end

"Coriolis parameter from latitude in degrees north."
_f_from_lat(lat, FT) = FT(2 * EARTH_ROTATION_RATE * sind(Float64(lat)))

"""
    coriolis_field(scheme, sz, FT)

The Coriolis parameter (s⁻¹) on T-points under the [`AbstractCoriolisParameter`](@ref)
`scheme`, as a matrix of the grid's input size `sz`, which `Model` crops with the grid.
"""
function coriolis_field end

coriolis_field(cp::CoriolisParameter0D, sz, FT) = fill(FT(cp.f), sz)

function coriolis_field(cp::CoriolisParameter2D, sz, FT)
    lat = cp.lat
    if lat isa AbstractMatrix
        size(lat) == sz || throw(
            ArgumentError(
                "coriolis latitude is $(size(lat)) but the mask is $sz; a 2D latitude " *
                "must cover the full domain including the border ring",
            ),
        )
        all(l -> -90 <= l <= 90, lat) ||
            throw(ArgumentError("coriolis latitude must lie in [-90, 90] degrees north"))
        return FT[_f_from_lat(l, FT) for l in lat]
    end
    lat isa Real || throw(
        ArgumentError(
            "coriolis latitude must be a real scalar or a matrix, got $(typeof(lat))",
        ),
    )
    -90 <= lat <= 90 || throw(
        ArgumentError("coriolis latitude must lie in [-90, 90] degrees north, got $lat"),
    )
    return fill(_f_from_lat(lat, FT), sz)
end

Base.show(io::IO, cp::CoriolisParameter0D) =
    print(io, "CoriolisParameter0D(f = ", cp.f, ")")

Base.show(io::IO, cp::CoriolisParameter2D) =
    print(io, "CoriolisParameter2D(lat = $(_range_str(cp.lat, 2))°N)")
