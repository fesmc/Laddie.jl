# ============================================================================
# Forcing types.
#
# A `CavityForcing` bundles the two external drivers of the meltwater layer:
#   - `ocean` — the ambient T/S the layer entrains and exchanges heat with
#   - `ice`   — the state of the ice above it; currently the ice temperature above
#               the basal thermal boundary layer (`T_ice_farfield`, typically the surface
#               temperature), which sets the heat conducted into the ice
#
# Ocean profiles are carried on a uniform z-grid.  The concrete vector type is a
# type parameter V (Vector{FT} on CPU, CuArray{FT,1} on GPU after to_backend),
# keeping field access type-stable.  The resampling `OceanForcing1D(Tz, Sz, z)`
# constructor follows the struct.
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
$(TYPEDEF)

Ambient T/S profiles from pre-loaded vectors — use this when the profile data
comes from a CSV file, an in-memory dataset, or any other source.

`z` is depth in metres (negative below sea level) and need not be sorted or
uniformly spaced: the profiles are sorted by depth and resampled to the 1-m
grid the interpolation kernel requires, with flat extrapolation beyond the
data range.

# Example
```julia
data = readdlm("profile-T.csv", ',', skipstart = 1)
forcing = OceanForcing1D(data[:, 1], S_values, data[:, 2] .* 1e3)
```
"""
function OceanForcing1D(
    Tz::AbstractVector,
    Sz::AbstractVector,
    z::AbstractVector;
    FT = Float64,
)
    length(Tz) == length(z) || throw(
        ArgumentError(
            "Tz and z must have the same length, got $(length(Tz)) vs $(length(z))",
        ),
    )
    length(Sz) == length(z) || throw(
        ArgumentError(
            "Sz and z must have the same length, got $(length(Sz)) vs $(length(z))",
        ),
    )
    isempty(z) && throw(ArgumentError("profile vectors must be non-empty"))
    p = sortperm(Float64.(z))
    z_s = Float64.(z[p])
    T_s = Float64.(Tz[p])
    S_s = Float64.(Sz[p])
    # Drop duplicate depths (keep first occurrence) so _interp1d never divides by zero.
    keep = [k == 1 || z_s[k] != z_s[k-1] for k in eachindex(z_s)]
    z_s = z_s[keep]
    T_s = T_s[keep]
    S_s = S_s[keep]
    if length(z_s) > 1 && all(≈(1.0), diff(z_s))
        z_new, T_new, S_new = z_s, T_s, S_s
    else
        z_new = collect(-5000.0:1.0:-1.0)
        T_new = _interp1d(z_s, T_s, z_new)
        S_new = _interp1d(z_s, S_s, z_new)
    end
    return OceanForcing1D(FT.(T_new), FT.(S_new), FT.(z_new), FT(1.0), FT(z_new[1]))
end

"""
Ambient T/S profiles of the ISOMIP+ protocol ([Asay-Davis et al. 2016](@cite asay-davis_experimental_2016)), as an
[`OceanForcing1D`](@ref): linear from the surface (−1.9 °C, 33.8 psu) to the values below at
720 m, and extrapolated linearly beneath, as in LADDIE v1.

- `isomipcond = :warm`: 1.0 °C, 34.7 psu at 720 m.
- `isomipcond = :cold`: −1.9 °C, 34.55 psu at 720 m.

```jldoctest
julia> ISOMIPForcing(:cold; FT = Float32)
OceanForcing1D{Float32}: 5000-point profile, z ∈ [-5000.0, -1.0] m, T ∈ [-1.9, -1.9] °C, S ∈ [33.801, 39.008] psu
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
    return OceanForcing1D(Tz, Sz, z, dz, z0)
end

# The arrays' element type is not tied to `FT`: under `trace_parameters` the scalars
# `dz` and `z0` are traced numbers while the profiles stay plain arrays.
"""
$(TYPEDEF)

Laterally varying ambient T/S: one vertical profile per grid cell, all on the same
uniform z-grid.  `Tz` and `Sz` are `(nx, ny, nz)` arrays covering the full domain
including the border ring (like a 2D `T_ice_farfield`); `Model` crops them with the
grid.  Each cell samples its own column at the depth of its plume base, and depths
outside the grid take the nearest level, as for [`OceanForcing1D`](@ref).

# Fields
$(TYPEDFIELDS)
"""
struct OceanForcing2D{FT,A<:AbstractArray{<:Any,3},V<:AbstractVector} <: AbstractOceanForcing
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
    return OceanForcing2D(T, S, FT.(zs), FT(dzs[1]), FT(zs[1]))
end

Base.show(io::IO, f::OceanForcing2D) = print(
    io,
    "OceanForcing2D{$(eltype(f.Tz))}: $(size(f.Tz, 1))×$(size(f.Tz, 2)) profiles of ",
    "$(length(f.z)) levels, z ∈ [$(join(extrema(f.z), ", "))] m, ",
    "T ∈ [$(join(_r.(extrema(f.Tz), 3), ", "))] °C, S ∈ [$(join(_r.(extrema(f.Sz), 3), ", "))] psu",
)

"""
$(TYPEDEF)

Ambient T/S from one vertical profile per basin and a map assigning every cell to a basin;
each cell samples its basin's profile as an [`OceanForcing1D`](@ref) would. For domains
spanning several basins: `n_basins × nz` numbers instead of the `nx × ny × nz` of an
[`OceanForcing2D`](@ref). `basin` covers the full domain, and every cell, land included,
must hold a valid index.

Construct it from per-basin `OceanForcing1D`s, `PerBasinOceanForcing1D(profiles, basin)`, or
from `(n_basins, nz)` matrices, `PerBasinOceanForcing1D(Tz, Sz, z, basin)`.

# Fields
$(TYPEDFIELDS)
"""
struct PerBasinOceanForcing1D{FT,M<:AbstractMatrix,I<:AbstractMatrix,V<:AbstractVector} <:
       AbstractOceanForcing
    "ambient temperature per basin and level, `(n_basins, nz)` (°C)"
    Tz::M
    "ambient salinity per basin and level, `(n_basins, nz)` (psu)"
    Sz::M
    "basin index of every cell, in `1:n_basins`"
    basin::I
    "depths of the uniform z-grid (m, negative downward, ascending)"
    z::V
    "z-grid spacing (m)"
    dz::FT
    "depth of the first z-grid point (m)"
    z0::FT
end

"""
    PerBasinOceanForcing1D(Tz, Sz, z, basin; FT = Float64)

Per-basin profiles `Tz`, `Sz` of size `(n_basins, nz)` on the depths `z` (m,
negative downward), which must be uniformly spaced; they are sorted ascending.
`basin` is the integer basin index of every cell of the full domain.
"""
function PerBasinOceanForcing1D(
    Tz::AbstractMatrix{<:Real},
    Sz::AbstractMatrix{<:Real},
    z::AbstractVector,
    basin::AbstractMatrix{<:Integer};
    FT = Float64,
)
    size(Tz) == size(Sz) || throw(ArgumentError("Tz is $(size(Tz)) but Sz is $(size(Sz))"))
    size(Tz, 2) == length(z) ||
        throw(ArgumentError("Tz has $(size(Tz, 2)) levels but z has $(length(z))"))
    length(z) ≥ 2 || throw(ArgumentError("need at least two levels"))
    nb = size(Tz, 1)
    all(b -> 1 ≤ b ≤ nb, basin) || throw(
        ArgumentError("every cell needs a basin index in 1:$nb; got $(join(extrema(basin), "…"))"),
    )
    p = sortperm(z)
    zs = Float64.(z[p])
    dzs = diff(zs)
    all(d -> isapprox(d, dzs[1]; rtol = 1e-6), dzs) ||
        throw(ArgumentError("the levels of a PerBasinOceanForcing1D must be uniformly spaced"))
    T, S = FT.(Tz[:, p]), FT.(Sz[:, p])
    all(isfinite, T) && all(isfinite, S) || throw(ArgumentError("Tz and Sz must be finite"))
    return PerBasinOceanForcing1D(T, S, Int32.(basin), FT.(zs), FT(dzs[1]), FT(zs[1]))
end

"""
    PerBasinOceanForcing1D(profiles, basin; FT = eltype of the profiles)

One basin per `OceanForcing1D` in `profiles` (basin `k` is `profiles[k]`), resampled
onto a common grid spanning all of them at their spacing, each extended flat beyond
its own depth range, as `OceanForcing1D` itself extrapolates. Front profiles of
different basins reach different depths, which this reconciles.
"""
function PerBasinOceanForcing1D(
    profiles::AbstractVector{<:OceanForcing1D},
    basin::AbstractMatrix{<:Integer};
    FT = eltype(first(profiles).Tz),
)
    isempty(profiles) && throw(ArgumentError("need at least one profile"))
    dz = Float64(first(profiles).dz)
    all(p -> isapprox(Float64(p.dz), dz; rtol = 1e-6), profiles) ||
        throw(ArgumentError("the profiles must share their z spacing"))
    zlo = minimum(p -> Float64(first(p.z)), profiles)
    zhi = maximum(p -> Float64(last(p.z)), profiles)
    z = collect(zlo:dz:(zhi+dz/2))
    sample(p, P) = map(z) do zk
        x = clamp((zk - Float64(p.z0)) / Float64(p.dz), 0.0, length(p.z) - 1.0)
        lo = floor(Int, x)
        hi = min(lo + 1, length(p.z) - 1)
        w = x - lo
        (1 - w) * Float64(P[lo+1]) + w * Float64(P[hi+1])
    end
    Tz = permutedims(stack(sample(p, p.Tz) for p in profiles))
    Sz = permutedims(stack(sample(p, p.Sz) for p in profiles))
    return PerBasinOceanForcing1D(Tz, Sz, z, basin; FT)
end

Base.show(io::IO, f::PerBasinOceanForcing1D) = print(
    io,
    "PerBasinOceanForcing1D{$(eltype(f.Tz))}: $(size(f.Tz, 1)) basin profiles of ",
    "$(length(f.z)) levels over $(size(f.basin, 1))×$(size(f.basin, 2)) cells, ",
    "z ∈ [$(join(extrema(f.z), ", "))] m, ",
    "T ∈ [$(join(_r.(extrema(f.Tz), 3), ", "))] °C, S ∈ [$(join(_r.(extrema(f.Sz), 3), ", "))] psu",
)

"""
$(TYPEDEF)

Ambient T/S that ramps linearly in time from `from` to `to`: `(1 − w)·from + w·to` with
`w = clamp((t − start)/duration, 0, 1)`, `t` in days of model time. `from` and `to` are of
the same type on the same depth grid (and basin map). The weight follows the clock, also
across a restart. Not supported by the Reactant backend, which would freeze the weight.

```julia
RampedOceanForcing(cold, warm; start = 600, duration = 100)   # days
```

# Fields
$(TYPEDFIELDS)
"""
struct RampedOceanForcing{F<:AbstractOceanForcing,R<:Base.RefValue} <: AbstractOceanForcing
    "the forcing up to `start`"
    from::F
    "the forcing from `start + duration` on"
    to::F
    "start of the ramp (days of model time)"
    start::Float64
    "length of the ramp (days)"
    duration::Float64
    "the current weight of `to`, in `[0, 1]`, set from the simulation clock"
    weight::R
end

function RampedOceanForcing(from::F, to::F; start::Real, duration::Real) where {F<:AbstractOceanForcing}
    from isa RampedOceanForcing && throw(ArgumentError("cannot ramp between RampedOceanForcings"))
    duration > 0 || throw(ArgumentError("duration must be positive, got $duration"))
    isfinite(start) || throw(ArgumentError("start must be finite, got $start"))
    from.z == to.z || throw(ArgumentError("from and to must share their depth grid"))
    size(from.Tz) == size(to.Tz) ||
        throw(ArgumentError("from has $(size(from.Tz)) profile values but to has $(size(to.Tz))"))
    from isa PerBasinOceanForcing1D && from.basin != to.basin &&
        throw(ArgumentError("from and to must share their basin map"))
    FT = eltype(from.Tz)
    return RampedOceanForcing(from, to, Float64(start), Float64(duration), Ref(_ramp_weight(start, duration, 0.0, FT)))
end
RampedOceanForcing(from::AbstractOceanForcing, to::AbstractOceanForcing; kwargs...) = throw(
    ArgumentError("from and to must be of the same type, got $(nameof(typeof(from))) and $(nameof(typeof(to)))"),
)

# The weight of `to` at `t_days`.
_ramp_weight(start, duration, t_days, ::Type{FT}) where {FT} = FT(clamp((t_days - start) / duration, 0, 1))

# Set the ramp weight from the model time (days); a no-op for every other forcing.
_set_forcing_time!(o::AbstractOceanForcing, t_days) = o
function _set_forcing_time!(o::RampedOceanForcing, t_days)
    o.weight[] = _ramp_weight(o.start, o.duration, t_days, typeof(o.weight[]))
    return o
end

Base.show(io::IO, f::RampedOceanForcing) = print(
    io,
    "RampedOceanForcing over days $(f.start)–$(f.start + f.duration) (weight now $(f.weight[])) from ",
    f.from, " to ", f.to,
)

#########################
# Ice
#########################

"""
Supertype of every ice forcing: the state of the ice above the layer. Subtypes provide a field
`T_ice_farfield` (°C, ≤ 0), read by the [`AbstractIceConduction`](@ref) scheme.
"""
abstract type AbstractIceForcing end

"""
$(TYPEDEF)

Ice temperature `T_ice_farfield` (°C) held fixed for the whole run: the temperature of the
ice above the basal boundary layer, typically the surface temperature, not the basal
temperature. A scalar, or a matrix the size of `mask` that `Model` crops with the grid. Its
meaning and range are on the [Ice forcing](@ref) page.

```jldoctest
julia> PrescribedIceForcing(-25.0)   # uniform, the default
PrescribedIceForcing(T_ice_farfield = -25.0 °C)
```

# Fields
$(TYPEDFIELDS)
"""
struct PrescribedIceForcing{M} <: AbstractIceForcing
    "ice temperature above the basal boundary layer, typically the surface temperature (°C): a scalar, or a full-domain matrix the size of `mask`"
    T_ice_farfield::M
end

# Default ice temperature `T_ice_farfield`, applied when a bare ocean forcing is passed to
# `Model` (the ISOMIP+ value).
const DEFAULT_T_ICE_FARFIELD = -25.0

PrescribedIceForcing() = PrescribedIceForcing(DEFAULT_T_ICE_FARFIELD)

########################
# Forcing
########################

"""
$(TYPEDEF)

The complete external forcing of a cavity: an ocean forcing and an ice forcing.

`Model` takes one of these. Passing a bare ocean forcing is shorthand for
`CavityForcing(ocean)`, which pairs it with a uniform
`PrescribedIceForcing($(DEFAULT_T_ICE_FARFIELD))`.

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
    "the ice forcing (the ice temperature `T_ice_farfield`)"
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

# ============================================================================
# Ambient T/S at the layer base: `update_ambient_fields!` and its kernels
# ============================================================================

# The profile levels around the layer base z_draft − H of cell (i, j): the 1-based
# indices `lo`, `hi` and the weight of `hi`.  The depth index is made finite and clamped
# to [0, nz − 1] in float first, so that `unsafe_trunc` always receives a representable
# value: the checked `trunc` leaves an InexactError branch (a trap) that Reactant cannot
# raise.  The weight subtracts `floor`, not the index converted back: Enzyme on raised
# kernels (Reactant 0.2.286) passes the derivative through the int round trip, so
# `depth_idx - FT(idx_lo)` gets a zero tangent.
@inline function _depth_levels(z_draft, H, z0, dz, nz, i, j)
    FT = typeof(z0)
    depth_idx = @inbounds (z_draft[i, j] - H[i, j] - z0) / dz
    depth_idx = ifelse(isfinite(depth_idx), depth_idx, zero(FT))
    depth_idx = clamp(depth_idx, zero(FT), FT(nz - 1))
    lo = floor(_primal(depth_idx))
    idx_lo = unsafe_trunc(Int, lo)
    idx_hi = clamp(idx_lo + 1, 0, nz - 1)
    return idx_lo + 1, idx_hi + 1, depth_idx - lo
end

# Level `k` of the ambient profile seen by cell (i, j): the one profile of an
# `OceanForcing1D`, the cell's own column of an `OceanForcing2D`, or the row of the
# cell's basin in the `(n_basins, nz)` table of a `PerBasinOceanForcing1D`.
@inline _profile_level(P::AbstractVector, ::Nothing, i, j, k) = @inbounds P[k]
@inline _profile_level(P::AbstractArray{<:Any,3}, ::Nothing, i, j, k) = @inbounds P[i, j, k]
@inline _profile_level(P::AbstractMatrix, basin::AbstractMatrix, i, j, k) = @inbounds P[basin[i, j], k]

# The profile `P` at the layer base, between levels `lo` and `hi` (weight `a` of `hi`).
@inline _sample(P, basin, i, j, lo, hi, a, FT) =
    a * _profile_level(P, basin, i, j, hi) + (one(FT) - a) * _profile_level(P, basin, i, j, lo)

@kernel function _ambient_interp_kernel!(Ta, Sa, @Const(z_draft), @Const(H), @Const(Tz), @Const(Sz), basin, z0, dz, nz)
    z0, dz = _val(z0), _val(dz)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(z0)
        lo, hi, a = _depth_levels(z_draft, H, z0, dz, nz, i, j)
        Ta[i, j] = _sample(Tz, basin, i, j, lo, hi, a, FT)
        Sa[i, j] = _sample(Sz, basin, i, j, lo, hi, a, FT)
    end
end

# The ambient sampling of a `RampedOceanForcing`: the two profile tables blended with
# the weight `w` of the second, exact at w = 0 and w = 1.
@kernel function _ambient_interp_ramp_kernel!(
    Ta, Sa, @Const(z_draft), @Const(H), @Const(Tz0), @Const(Sz0), @Const(Tz1), @Const(Sz1), basin, w, z0, dz, nz,
)
    z0, dz, w = _val(z0), _val(dz), _val(w)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = typeof(w)
        lo, hi, a = _depth_levels(z_draft, H, z0, dz, nz, i, j)
        Ta[i, j] = (one(FT) - w) * _sample(Tz0, basin, i, j, lo, hi, a, FT) + w * _sample(Tz1, basin, i, j, lo, hi, a, FT)
        Sa[i, j] = (one(FT) - w) * _sample(Sz0, basin, i, j, lo, hi, a, FT) + w * _sample(Sz1, basin, i, j, lo, hi, a, FT)
    end
end

_basin(o::PerBasinOceanForcing1D) = o.basin
_basin(::AbstractOceanForcing) = nothing

"""
$(TYPEDSIGNATURES)

Vertically interpolate the ambient T/S profiles to the depth of each grid cell's
plume base (z_draft − H), writing results into `m.Ta` and `m.Sa`. This sampling has
no numbered equation in [Lambert et al. (2023)](@cite lambert_modelling_2023); see `docs/src/physics/equations.md`.
"""
update_ambient_fields!(m) = update_ambient_fields!(m, m.forcing.ocean)

function update_ambient_fields!(m, o::RampedOceanForcing)
    from, to = o.from, o.to
    launch!(_ambient_interp_ramp_kernel!, m.Ta, m.Sa, m.z_draft, m.H.present, from.Tz, from.Sz, to.Tz, to.Sz,
            _basin(from), o.weight[], from.z0, from.dz, length(from.z))
    return
end

function update_ambient_fields!(m, o::AbstractOceanForcing)
    launch!(_ambient_interp_kernel!, m.Ta, m.Sa, m.z_draft, m.H.present, o.Tz, o.Sz, _basin(o), o.z0, o.dz, length(o.z))
    return
end
