# ============================================================================
# Geometry helpers for builders
# ============================================================================

# Adjust z_draft before Grid construction.  z_draft is defined by mask category:
#   ocean  (0): 0        (no ice above, sea-surface reference)
#   land   (1): 0        (bedrock at/above sea level, or the border ring; inactive)
#   grounded (2): z_bed  (ice base coincides with bed; keep z_draft_raw)
#   shelf  (3): z_draft_raw   (actual ice-base depth, clamped to ≤ -1 m)
#   gap    (4): 0        (ice-free, so the layer's upper boundary is the sea surface)
# The gap draft is deliberately not clamped to -1 m: the layer sits directly beneath
# the surface there, matching LADDIE v2 where `Hib = Hs - Hi` is exactly 0 for ice-free
# cells.  Any NaN fill values from the raw data are stripped before the mask logic.
function _adjust_z_draft(mask::AbstractMatrix{Int}, z_draft_raw::AbstractMatrix, FT)
    ice = FT.(mask .== 3)
    z_draft = FT.(z_draft_raw)
    z_draft = ifelse.(isnan.(z_draft), zero(FT), z_draft)              # strip NaN fill values
    z_draft = ifelse.((mask .== 0) .| (mask .== 1) .| (mask .== 4), zero(FT), z_draft)  # ocean + border + gap → 0
    z_draft = ifelse.((ice .> 0) .& (z_draft .> FT(-1)), FT(-1), z_draft)  # clamp shallow shelf
    return z_draft
end

# Initialise prognostic fields from scratch: all three time levels identical.  The
# secondary fields and the leapfrog bootstrap step depend on dt, so they are left
# to the `Simulation` constructor.
function _initialize_prognostics!(m)
    update_ambient_fields!(m)
    for level in (:past, :present, :future)
        setfield!(m.H, level, m.H_init .* m.tmask)
        setfield!(m.T, level, (m.Ta .+ m.dT_init) .* m.tmask)
        setfield!(m.S, level, (m.Sa .+ m.dS_init) .* m.tmask)
    end
    return
end

# ============================================================================
# Geometry ingestion utilities
# ============================================================================

_neighbours(c::CartesianIndex{2}) = (
    c + CartesianIndex(-1, 0),
    c + CartesianIndex(1, 0),
    c + CartesianIndex(0, -1),
    c + CartesianIndex(0, 1),
)

# A cell on the outermost ring of the array or directly inside it.
function _near_border(mask, c)
    nx, ny = size(mask)
    i, j = Tuple(c)
    return i <= 2 || j <= 2 || i >= nx - 1 || j >= ny - 1
end

# Flood fill: marks in `reached` every cell reachable from `seeds` through
# 4-connected cells whose mask value satisfies `passable`, and returns the cells
# reached from these seeds.
function _flood!(reached, mask, seeds, passable)
    queue = collect(CartesianIndex{2}, seeds)
    reached[queue] .= true
    k = 1
    while k <= length(queue)
        for n in _neighbours(queue[k])
            if checkbounds(Bool, mask, n) && !reached[n] && passable(mask[n])
                reached[n] = true
                push!(queue, n)
            end
        end
        k += 1
    end
    return queue
end

# The 4-connected components of the cells whose mask value satisfies `member`.
function _components(mask, member)
    reached = falses(size(mask))
    return [
        _flood!(reached, mask, (c,), member) for
        c in CartesianIndices(mask) if member(mask[c]) && !reached[c]
    ]
end

# Dynamically active mask values: floating shelf and ice-shelf gap.  Gaps are ice-free
# but still carry the plume, so connectivity-based mask cleaning must not treat them as
# a barrier (that would strand the shelf beyond a gap and silently ground it).
_is_active(v::Integer) = v == 3 || v == 4

"""
$(TYPEDSIGNATURES)

Derive the LADDIE domain mask from BedMachine-style bed elevation `bed` (m, positive up) and ice
`thickness` (m) on the interior `(nx, ny)`; the result is `(nx+2, ny+2)`, with a border ring
of `1`.

| Value | Meaning         | Condition                             |
|-------|-----------------|---------------------------------------|
| `0`   | open ocean      | `thickness ≤ 0` and `bed < 0`         |
| `1`   | land            | `thickness ≤ 0` and `bed ≥ 0`; border ring |
| `2`   | grounded ice    | `thickness > 0` and `h_af ≥ 0`        |
| `3`   | floating shelf  | `thickness > 0` and `h_af < 0`        |

with `h_af = thickness × rho_ice/rho_sw + bed` the height above flotation. Exposed bedrock is
land, so a rock island in a shelf is a wall, not an open-ocean sink. Gaps (`4`) are marked by
[`MarkGapsPreprocess`](@ref). The default densities (kg m⁻³) are BedMachine's.
"""
function build_laddie_mask(bed, thickness; rho_ice = 917.0, rho_sw = 1028.0)
    nx, ny = size(bed)
    size(bed) == size(thickness) || throw(
        ArgumentError(
            "bed and thickness must have the same size, got $(size(bed)) vs $(size(thickness))",
        ),
    )
    mask = zeros(Int, nx + 2, ny + 2)
    mask[1, :] .= 1
    mask[end, :] .= 1
    mask[:, 1] .= 1
    mask[:, end] .= 1
    r = rho_ice / rho_sw
    for j = 1:ny, i = 1:nx
        h = Float64(thickness[i, j])
        b = Float64(bed[i, j])
        mask[i+1, j+1] = if h > 0
            (h * r + b >= 0) ? 2 : 3     # grounded ice / floating shelf
        else
            (b >= 0) ? 1 : 0             # exposed bedrock / open ocean
        end
    end
    return mask
end

"""
$(TYPEDSIGNATURES)

Pad a bed-elevation array into the `(nx+2, ny+2)` format expected by [`Grid`](@ref).
The one-cell border ring is zeroed; interior values are copied from `bed` unchanged.
Pass the result as `Grid(...; z_bed)`; only the topographic caps of
[`AbstractMaxLayerThickness`](@ref) use it.

# Arguments
- `bed`: bed elevation (m, positive above sea level), size `(nx, ny)`.
"""
function bed_elevation(bed; FT = Float64)
    nx, ny = size(bed)
    z_bed = zeros(FT, nx + 2, ny + 2)
    for j = 1:ny, i = 1:nx
        z_bed[i+1, j+1] = FT(bed[i, j])
    end
    return z_bed
end

"""
$(TYPEDSIGNATURES)

Ice-base depth (m, negative below sea level) from BedMachine-style arrays, as a
`(nx+2, ny+2)` matrix for [`Grid`](@ref): `-thickness * rho_ice/rho_sw` where the ice floats
(`h_af < 0`), `bed` where it is grounded, `0` elsewhere and on the border ring. Arguments as for
[`build_laddie_mask`](@ref).
"""
function ice_base_depth(bed, thickness; rho_ice = 917.0, rho_sw = 1028.0)
    nx, ny = size(bed)
    size(bed) == size(thickness) || throw(
        ArgumentError(
            "bed and thickness must have the same size, got $(size(bed)) vs $(size(thickness))",
        ),
    )
    z_draft = zeros(Float64, nx + 2, ny + 2)
    r = rho_ice / rho_sw
    for j = 1:ny, i = 1:nx
        h = Float64(thickness[i, j])
        b = Float64(bed[i, j])
        if h > 0
            z_draft[i+1, j+1] = (h * r + b >= 0) ? b : -h * r
        end
    end
    return z_draft
end

"""
$(TYPEDSIGNATURES)

Turn enclosed ocean pockets of `mask` into land (`0` → `1`): ocean cells not 4-connected to the
ocean on the outer ring. Land, not grounded ice, since there is no ice there. Modifies `mask`
and returns the number of cells changed.
"""
function fill_ocean_holes!(mask::AbstractMatrix{Int})
    # Seed: ocean cells on the outermost ring of the array, or directly inside it.
    # The ring is the domain boundary and is normally land, so the seeds are the
    # ocean cells of the second ring.  The test is positional because land also
    # marks interior bedrock (nunataks, rock islands), and seeding off those would
    # declare every pocket beside an island part of the open ocean.
    cells = CartesianIndices(mask)
    seeds = [c for c in cells if mask[c] == 0 && _near_border(mask, c)]
    reached = falses(size(mask))
    _flood!(reached, mask, seeds, ==(0))
    holes = [c for c in cells if mask[c] == 0 && !reached[c]]
    mask[holes] .= 1
    return length(holes)
end

"""
$(TYPEDSIGNATURES)

Turn shelf patches with no 4-connected path to the open ocean into grounded ice (`3` → `2`): a
shelf with no ice front. Call [`fill_ocean_holes!`](@ref) first, so that enclosed pockets do
not count as ocean. Modifies `mask` and returns the number of cells changed.
"""
function fill_shelf_holes!(mask::AbstractMatrix{Int})
    # Seed: active cells next to the ocean.  The fill runs through active cells only,
    # and gaps (4) conduct it: a shelf region reachable only through a gap is still
    # attached to the ocean.
    cells = CartesianIndices(mask)
    at_ocean(c) = any(n -> checkbounds(Bool, mask, n) && mask[n] == 0, _neighbours(c))
    seeds = [c for c in cells if _is_active(mask[c]) && at_ocean(c)]
    reached = falses(size(mask))
    _flood!(reached, mask, seeds, _is_active)
    holes = [c for c in cells if mask[c] == 3 && !reached[c]]
    mask[holes] .= 2
    return length(holes)
end

"""
$(TYPEDSIGNATURES)

Turn isolated grounded patches (not connected to the outer ring) of fewer than `min_cells`
cells into shelf (`2` → `3`): pinning points and rumples too small to resolve. Call
[`fill_ocean_holes!`](@ref) and [`fill_shelf_holes!`](@ref) first. Modifies `mask` and
returns the number of cells changed.
"""
function fill_small_grounded_patches!(mask::AbstractMatrix{Int}, min_cells::Int = 10)
    n_filled = 0
    for component in _components(mask, ==(2))
        # Positional border test: `mask == 1` also marks interior bedrock, which must
        # not count as "attached to the ice sheet".
        any(c -> _near_border(mask, c), component) && continue
        length(component) < min_cells || continue
        mask[component] .= 3
        n_filled += length(component)
    end
    return n_filled
end

"""
$(TYPEDSIGNATURES)

Turn shelf patches of fewer than `min_cells` cells into grounded ice (`3` → `2`): a patch of a
few cells cannot resolve a coherent flow. Call [`fill_ocean_holes!`](@ref) and
[`fill_shelf_holes!`](@ref) first. Modifies `mask` and returns the number of cells changed.
"""
function fill_small_shelf_patches!(mask::AbstractMatrix{Int}, min_cells::Int = 10)
    n_filled = 0
    # Gaps (4) belong to the component they sit in, so a gap never splits one shelf
    # into two.
    for component in _components(mask, _is_active)
        length(component) < min_cells || continue
        for c in component
            # Shelf becomes grounded; a gap has no ice to ground, so it reverts to
            # open ocean.
            mask[c] = mask[c] == 4 ? 0 : 2
        end
        n_filled += length(component)
    end
    return n_filled
end

# ============================================================================
# Mask preprocessing pipeline
# ============================================================================

"""
Abstract supertype for a mask-preprocessing step.  Pass a list of concrete
instances as `Grid(...; preprocess = [...])`; they run in order on a copy of the
mask, before domain cropping.
"""
abstract type AbstractPreprocess end

"""
$(TYPEDEF)

Preprocessing step that applies [`fill_ocean_holes!`](@ref): enclosed ocean
pockets become land.
"""
struct FillOceanHolesPreprocess <: AbstractPreprocess end

"""
$(TYPEDEF)

Preprocessing step that applies [`fill_shelf_holes!`](@ref): shelf patches with no
connection to the open ocean become grounded ice.
"""
struct FillShelfHolesPreprocess <: AbstractPreprocess end

"""
$(TYPEDEF)

Preprocessing step that applies [`fill_small_shelf_patches!`](@ref) with
`min_cells = min_size` (default 10).

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct FillSmallShelfPatchesPreprocess{I} <: AbstractPreprocess
    "patches of fewer cells are filled (default `10`)"
    min_size::I = 10
end

"""
$(TYPEDEF)

Preprocessing step that applies [`fill_small_grounded_patches!`](@ref) with
`min_cells = min_size` (default 10).

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct FillSmallGroundedPatchesPreprocess{I} <: AbstractPreprocess
    "patches of fewer cells are filled (default `10`)"
    min_size::I = 10
end

"""
$(TYPEDEF)

Mark ice-shelf gaps from a reference ice footprint: every open-ocean cell (`0`) where `refgeo`
marks ice (`true`, or positive) becomes a gap (`4`), as the `refgeo_Hi > 0` test of LADDIE v2.
`refgeo` has the size of the mask. Place it after any hole-filling steps; what happens in a gap
is the [`AbstractGapsBC`](@ref).

# Fields
$(TYPEDFIELDS)
"""
struct MarkGapsPreprocess{R<:AbstractMatrix} <: AbstractPreprocess
    "reference ice footprint (`Bool`, or numeric with positive entries marking ice)"
    refgeo::R
end

"""
    preprocess!(mask, step)

Apply the [`AbstractPreprocess`](@ref) `step` to `mask` in place.
"""
function preprocess! end

function preprocess!(mask, p::MarkGapsPreprocess)
    size(p.refgeo) == size(mask) || throw(
        ArgumentError(
            "MarkGapsPreprocess refgeo must have the same size as the mask, got " *
            "$(size(p.refgeo)) vs $(size(mask)); note the mask is the one passed to " *
            "`Grid`, before any domain cropping",
        ),
    )
    had_ice = p.refgeo isa AbstractMatrix{Bool} ? p.refgeo : p.refgeo .> 0
    gaps = (mask .== 0) .& had_ice
    mask[gaps] .= 4
    return count(gaps)
end

preprocess!(mask, ::FillOceanHolesPreprocess) = fill_ocean_holes!(mask)
preprocess!(mask, ::FillShelfHolesPreprocess) = fill_shelf_holes!(mask)
preprocess!(mask, p::FillSmallShelfPatchesPreprocess) =
    fill_small_shelf_patches!(mask, p.min_size)
preprocess!(mask, p::FillSmallGroundedPatchesPreprocess) =
    fill_small_grounded_patches!(mask, p.min_size)

# ============================================================================
# Domain cropping
# ============================================================================

"""
Abstract supertype for how [`Grid`](@ref) crops its inputs: pass
`Grid(...; domain_cropping = ...)` with [`MinRectangleDomainCropping`](@ref) (the
default) or [`NoDomainCropping`](@ref).
"""
abstract type AbstractDomainCropping end

"""
$(TYPEDEF)

Keep the full input arrays; the grid is exactly the mask that was passed in.
"""
struct NoDomainCropping <: AbstractDomainCropping end

"""
$(TYPEDEF)

Crop the grid inputs (and any full-domain field a model is given later) to the smallest
rectangle holding the active cells (`3` and `4`), expanded by `margin ≥ 2` cells and clipped
to the input. `multiple` rounds the cropped size up to a multiple, per axis or for both, as a
grid sharded over a device mesh needs. The default; see [Domain cropping](@ref).

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct MinRectangleDomainCropping{I,M} <: AbstractDomainCropping
    "cells of padding kept around the active region (minimum 2)"
    margin::I = 4
    "the grid size is rounded up to a multiple of this, per axis (default `1`: no rounding)"
    multiple::M = 1
end

"""
    crop_ranges(mask, cropping) -> (rows, cols)

The index ranges of `mask` that [`Grid`](@ref) keeps under the
[`AbstractDomainCropping`](@ref) `cropping`; every full-domain input is sliced with them.
"""
function crop_ranges end

crop_ranges(mask, ::NoDomainCropping) = axes(mask, 1), axes(mask, 2)

function crop_ranges(mask, cropping::MinRectangleDomainCropping)
    margin = cropping.margin
    margin >= 2 || throw(
        ArgumentError(
            "MinRectangleDomainCropping margin must be at least 2 — an ice front at the " *
            "edge of the cropped box must not touch the outermost ring — got $margin",
        ),
    )
    # Gaps (4) are active cells too — cropping them away would silently remove the
    # very region the connected-gaps treatment is about.
    shelf_inds = findall(m -> m == 3 || m == 4, mask)
    # Let validation report the empty domain; crop to everything in the meantime.
    isempty(shelf_inds) && return axes(mask, 1), axes(mask, 2)
    rows = getindex.(shelf_inds, 1)
    cols = getindex.(shelf_inds, 2)
    rmin, rmax = extrema(rows)
    cmin, cmax = extrema(cols)
    kr, kc = cropping.multiple isa Integer ? (cropping.multiple, cropping.multiple) :
             cropping.multiple
    r = _round_up(max(1, rmin-margin):min(size(mask, 1), rmax+margin), kr, size(mask, 1))
    c = _round_up(max(1, cmin-margin):min(size(mask, 2), cmax+margin), kc, size(mask, 2))
    if length(r) < size(mask, 1) || length(c) < size(mask, 2)
        # Report the margin actually achieved on each side, not just the requested
        # one: the active region is rarely centred, so `margin` is clipped by the
        # array edge on whichever side runs out of room first and the padding ends
        # up asymmetric.
        pad = (rmin - first(r), last(r) - rmax, cmin - first(c), last(c) - cmax)
        note =
            all(==(margin), pad) ? "" :
            "  (clipped by the array edge; kept top/bottom/left/right = $pad)"
        @info "Domain cropped from $(size(mask)) to ($(length(r)), $(length(c))) " *
              "with margin = $margin" *
              note
    end
    return r, c
end

# `rng` grown to a multiple of `k` cells within `1:n`: at its end first, then at its start.
function _round_up(rng, k, n)
    k >= 1 || throw(ArgumentError("MinRectangleDomainCropping multiple must be ≥ 1, got $k"))
    extra = mod(-length(rng), k)
    hi = min(n, last(rng) + extra)
    lo = first(rng) - (extra - (hi - last(rng)))
    lo >= 1 || throw(ArgumentError(
        "cannot round the cropped axis of $(length(rng)) cells up to a multiple of $k: " *
        "the input has only $n cells along it"))
    return lo:hi
end
