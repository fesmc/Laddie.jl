"""
$(TYPEDSIGNATURES)

Ambient T/S profiles from pre-loaded vectors — use this when the profile data
comes from a CSV file, an in-memory dataset, or any source other than the
NetCDF layout that `FileForcing` expects.

`z` is depth in metres (negative below sea level) and need not be sorted or
uniformly spaced: the profiles are sorted by depth and resampled to the 1-m
grid the interpolation kernel requires, with flat extrapolation beyond the
data range.

# Example
```julia
data = readdlm("profile-T.csv", ',', skipstart = 1)
forcing = ProfileForcing(data[:, 1], S_values, data[:, 2] .* 1e3)
```
"""
function ProfileForcing(
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
    ProfileForcing(FT.(T_new), FT.(S_new), FT.(z_new), FT(1.0), FT(z_new[1]))
end

function _interp1d(x, y, xi)
    out = similar(xi)
    for (k, xk) in enumerate(xi)
        if xk <= x[1]
            out[k] = y[1]
        elseif xk >= x[end]
            out[k] = y[end]
        else
            j = searchsortedlast(x, xk)
            t = (xk - x[j]) / (x[j+1] - x[j])
            out[k] = y[j] + t * (y[j+1] - y[j])
        end
    end
    return out
end


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

# Initialise prognostic fields from scratch (no restart file).
function _initialize_prognostics!(m)
    update_ambient_fields!(m)
    for level in (:past, :present, :future)
        setfield!(m.D, level, m.D_init .* m.tmask)
        setfield!(m.T, level, (m.Ta .+ m.dT_init) .* m.tmask)
        setfield!(m.S, level, (m.Sa .+ m.dS_init) .* m.tmask)
    end
    update_secondary_fields!(m)
    leapfrog_step!(m, 1)
    return
end

# ============================================================================
# Geometry ingestion utilities
# ============================================================================

const _cardinal_dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Dynamically active mask values: floating shelf and ice-shelf gap.  Gaps are ice-free
# but still carry the plume, so connectivity-based mask cleaning must not treat them as
# a barrier (that would strand the shelf beyond a gap and silently ground it).
_is_active(v::Integer) = v == 3 || v == 4

"""
$(TYPEDSIGNATURES)

Derive the 4-class LADDIE domain mask from BedMachine-style bed-elevation and
ice-thickness arrays.  Both arrays should cover the *interior* domain of size
`(ny, nx)`; the returned mask has size `(ny+2, nx+2)` with a one-cell border
ring of `1` (land/boundary).

| Value | Meaning         | Condition                             |
|-------|-----------------|---------------------------------------|
| `0`   | open ocean      | `thickness ≤ 0` and `bed < 0`         |
| `1`   | land            | `thickness ≤ 0` and `bed ≥ 0`; border ring |
| `2`   | grounded ice    | `thickness > 0` and `h_af ≥ 0`        |
| `3`   | floating shelf  | `thickness > 0` and `h_af < 0`        |

Height above flotation: `h_af = thickness × (rho_ice/rho_sw) + bed`.

Ice-free cells are split by bed elevation, which is the `thickness → 0` limit of the
flotation test: exposed bedrock — nunataks, rock islands inside a shelf, ice-free
coastline — is **land**, not ocean.  Classifying it as ocean (as this function did
before) makes every such island an open-boundary sink in the middle of the cavity:
`Grid.ocn` drives the ice-front indicators, so the surrounding shelf cells grow a
velocity point into the island and lose heat and momentum through it.

# Arguments
- `bed`:      bed elevation (m, positive above sea level).
- `thickness`: ice thickness (m, positive where ice is present).
- `rho_ice`:  ice density (kg m⁻³, default 917).
- `rho_sw`:   seawater density (kg m⁻³, default 1028).

# Example
```julia
ds    = NCDataset("BedMachineAntarctica-v3.nc")
z_bed = Float64.(Array(ds["bed"][i1:i2, j1:j2]))
h_ice = Float64.(Array(ds["thickness"][i1:i2, j1:j2]))
close(ds)
mask = build_laddie_mask(z_bed, h_ice)
```
"""
function build_laddie_mask(bed, thickness; rho_ice = 917.0, rho_sw = 1028.0)
    ny, nx = size(bed)
    size(bed) == size(thickness) || throw(
        ArgumentError(
            "bed and thickness must have the same size, got $(size(bed)) vs $(size(thickness))",
        ),
    )
    mask = zeros(Int, ny + 2, nx + 2)
    mask[1, :] .= 1
    mask[end, :] .= 1
    mask[:, 1] .= 1
    mask[:, end] .= 1
    r = rho_ice / rho_sw
    for j = 1:nx, i = 1:ny
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

Pad a bed-elevation array into the `(ny+2, nx+2)` format expected by `Model`.
The one-cell border ring is zeroed; interior values are copied from `bed` unchanged.
Pass the result as the `z_bed_raw` keyword argument to `Model` to enable the
water-column upper bound on plume thickness (`D <= z_draft - z_bed`).

# Arguments
- `bed`: bed elevation (m, positive above sea level), size `(ny, nx)`.
"""
function bed_elevation(bed; FT = Float64)
    ny, nx = size(bed)
    z_bed = zeros(FT, ny + 2, nx + 2)
    for j = 1:nx, i = 1:ny
        z_bed[i+1, j+1] = FT(bed[i, j])
    end
    return z_bed
end

"""
$(TYPEDSIGNATURES)

Compute ice-base depth (m, negative below sea level) from BedMachine-style arrays.
Returns a `(ny+2, nx+2)` matrix (interior domain with border ring zeroed).

- **Floating cells** (`h_af < 0`): `z_draft = -thickness * rho_ice/rho_sw` (Archimedes).
- **Grounded cells** (`h_af >= 0`): `z_draft = bed` (ice base rests on the bed).
- **Ocean / border cells**: `z_draft = 0`.

Pass the result directly as `z_draft_raw` to `Model`; `_adjust_z_draft` will clamp
very shallow shelf cells and zero the ice-front ocean strip.

# Arguments
- `bed`:      bed elevation (m, positive above sea level).
- `thickness`: ice thickness (m, positive where ice is present).
- `rho_ice`:  ice density (kg m⁻³, default 917).
- `rho_sw`:   seawater density (kg m⁻³, default 1028).
"""
function ice_base_depth(bed, thickness; rho_ice = 917.0, rho_sw = 1028.0)
    ny, nx = size(bed)
    size(bed) == size(thickness) || throw(
        ArgumentError(
            "bed and thickness must have the same size, got $(size(bed)) vs $(size(thickness))",
        ),
    )
    z_draft = zeros(Float64, ny + 2, nx + 2)
    r = rho_ice / rho_sw
    for j = 1:nx, i = 1:ny
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

Remove isolated ocean pockets from a LADDIE mask by flood-filling from the main
ocean.  Ocean cells (`mask == 0`) that are not connected (4-connectivity) to the
outer ocean are reclassified as land (`mask == 1`).

The outer ocean is identified as all ocean cells reachable from the outermost ring
of the array.  Noisy topography (e.g. BedMachine) occasionally creates small
enclosed ocean patches fully surrounded by ice; these cause spurious ice-front
dynamics and numerical instabilities.

An enclosed pocket becomes **land**, not grounded ice: there is no ice there, and
conflating the two hides which walls are rock and which are the grounding line.

Modifies `mask` in-place and returns the number of cells that were reclassified.

# Arguments
- `mask`: integer mask matrix as returned by [`build_laddie_mask`](@ref).

# Example
```julia
mask = build_laddie_mask(z_bed, h_ice)
n = fill_ocean_holes!(mask)
println("Reclassified \$n isolated ocean cells")
```
"""
function fill_ocean_holes!(mask::AbstractMatrix{Int})
    ny, nx = size(mask)
    visited = falses(ny, nx)
    queue = Tuple{Int,Int}[]

    # Seed: ocean cells on the outermost ring of the array, or directly inside it.
    # The ring is the domain boundary and is normally land, so the seeds are the
    # ocean cells of the second ring — the same set the old `mask == 1` adjacency
    # test picked out.  It has to be positional now: land also marks interior
    # bedrock (nunataks, rock islands), and seeding off those would declare every
    # pocket beside an island part of the open ocean.
    for i in 1:ny, j in 1:nx
        if mask[i, j] == 0 && (i <= 2 || j <= 2 || i >= ny - 1 || j >= nx - 1)
            visited[i, j] = true
            push!(queue, (i, j))
        end
    end

    # BFS to mark all reachable ocean cells
    while !isempty(queue)
        i, j = popfirst!(queue)
        for (di, dj) in _cardinal_dirs
            ni, nj = i + di, j + dj
            if 1 <= ni <= ny && 1 <= nj <= nx && !visited[ni, nj] && mask[ni, nj] == 0
                visited[ni, nj] = true
                push!(queue, (ni, nj))
            end
        end
    end

    # Reclassify unreachable ocean cells as land
    n_filled = 0
    for i in 1:ny, j in 1:nx
        if mask[i, j] == 0 && !visited[i, j]
            mask[i, j] = 1
            n_filled += 1
        end
    end
    return n_filled
end

"""
$(TYPEDSIGNATURES)

Remove isolated floating-shelf patches from a LADDIE mask by flood-filling from
shelf cells connected to the open ocean.  Shelf cells (`mask == 3`) that have no
4-connected path to any ocean cell (`mask == 0`) are reclassified as grounded ice
(`mask == 2`).

Noisy topography can produce small shelf patches fully enclosed by grounded ice
with no ice front; these are physically inconsistent and can cause instabilities.

Modifies `mask` in-place and returns the number of cells reclassified.  Call
[`fill_ocean_holes!`](@ref) first so that isolated ocean pockets do not
artificially seed shelf connectivity.

# Arguments
- `mask`: integer mask matrix as returned by [`build_laddie_mask`](@ref).

# Example
```julia
mask = build_laddie_mask(z_bed, h_ice)
fill_ocean_holes!(mask)
n = fill_shelf_holes!(mask)
println("Reclassified \$n isolated shelf cells")
```
"""
function fill_shelf_holes!(mask::AbstractMatrix{Int})
    ny, nx = size(mask)
    visited = falses(ny, nx)
    queue = Tuple{Int,Int}[]

    # Seed: active cells adjacent to at least one ocean cell
    for i in 1:ny, j in 1:nx
        if _is_active(mask[i, j]) && !visited[i, j]
            for (di, dj) in _cardinal_dirs
                ni, nj = i + di, j + dj
                if 1 <= ni <= ny && 1 <= nj <= nx && mask[ni, nj] == 0
                    visited[i, j] = true
                    push!(queue, (i, j))
                    break
                end
            end
        end
    end

    # BFS through active cells only.  Gaps (4) conduct connectivity: a shelf region
    # reachable only through a gap is still attached to the ocean.
    while !isempty(queue)
        i, j = popfirst!(queue)
        for (di, dj) in _cardinal_dirs
            ni, nj = i + di, j + dj
            if 1 <= ni <= ny && 1 <= nj <= nx && !visited[ni, nj] && _is_active(mask[ni, nj])
                visited[ni, nj] = true
                push!(queue, (ni, nj))
            end
        end
    end

    # Reclassify isolated shelf cells as grounded ice
    n_filled = 0
    for i in 1:ny, j in 1:nx
        if mask[i, j] == 3 && !visited[i, j]
            mask[i, j] = 2
            n_filled += 1
        end
    end
    return n_filled
end

"""
$(TYPEDSIGNATURES)

Remove undersized isolated grounded-ice patches from a LADDIE mask.  Each
4-connected component of grounded cells (`mask == 2`) that is *not* connected
to the outermost ring of the array is identified; any such component with
fewer than `min_cells` cells is reclassified as floating shelf (`mask == 3`).

Components that touch the border ring are part of the main grounded ice sheet
and are never modified, regardless of size.  Only truly isolated grounded
patches — pinning points, rumples, or topographic artefacts inside the shelf —
are candidates for removal.

Modifies `mask` in-place and returns the number of cells reclassified.
Recommended to call [`fill_ocean_holes!`](@ref) and
[`fill_shelf_holes!`](@ref) first so that the shelf geometry is clean before
removing pinning points.

# Arguments
- `mask`:      integer mask matrix as returned by [`build_laddie_mask`](@ref).
- `min_cells`: minimum number of cells an isolated grounded component must have
  to be retained (default 10).  Components strictly smaller than this are
  reclassified as shelf.

# Example
```julia
mask = build_laddie_mask(z_bed, h_ice)
fill_ocean_holes!(mask)
fill_shelf_holes!(mask)
n = fill_small_grounded_patches!(mask, 20)
println("Reclassified \$n cells in undersized isolated grounded patches")
```
"""
function fill_small_grounded_patches!(mask::AbstractMatrix{Int}, min_cells::Int = 10)
    ny, nx = size(mask)
    visited = falses(ny, nx)
    n_filled = 0

    for i in 1:ny, j in 1:nx
        mask[i, j] == 2 && !visited[i, j] || continue

        component = Tuple{Int,Int}[]
        queue = Tuple{Int,Int}[(i, j)]
        visited[i, j] = true
        touches_border = false
        while !isempty(queue)
            c_i, cj = popfirst!(queue)
            push!(component, (c_i, cj))
            for (di, dj) in _cardinal_dirs
                ni, nj = c_i + di, cj + dj
                1 <= ni <= ny && 1 <= nj <= nx || continue
                # Positional border test: `mask == 1` now also marks interior
                # bedrock, which must not count as "attached to the ice sheet".
                (ni == 1 || ni == ny || nj == 1 || nj == nx) && (touches_border = true)
                if !visited[ni, nj] && mask[ni, nj] == 2
                    visited[ni, nj] = true
                    push!(queue, (ni, nj))
                end
            end
        end

        if !touches_border && length(component) < min_cells
            for (c_i, cj) in component
                mask[c_i, cj] = 3
            end
            n_filled += length(component)
        end
    end
    return n_filled
end

"""
$(TYPEDSIGNATURES)

Remove undersized floating-shelf patches from a LADDIE mask.  Each
4-connected component of shelf cells (`mask == 3`) is identified; any
component with fewer than `min_cells` cells is reclassified as grounded
ice (`mask == 2`).

A component of only a few cells cannot meaningfully resolve the LADDIE
plume dynamics: the centred-difference stencil spans ≥ 2 cells in each
direction, and the depth-averaged momentum balance requires at least
O(10) cells to develop a coherent flow.  Removing these micro-patches
eliminates spurious gradients and numerical instabilities in noisy
real-world topography (e.g. BedMachine).

Modifies `mask` in-place and returns the number of cells reclassified.
Recommended to call [`fill_ocean_holes!`](@ref) and
[`fill_shelf_holes!`](@ref) first.

# Arguments
- `mask`:      integer mask matrix as returned by [`build_laddie_mask`](@ref).
- `min_cells`: minimum number of cells a shelf component must have to be
  retained (default 10).  Components strictly smaller than this are removed.

# Example
```julia
mask = build_laddie_mask(z_bed, h_ice)
fill_ocean_holes!(mask)
fill_shelf_holes!(mask)
n = fill_small_shelf_patches!(mask, 20)
println("Removed \$n cells in undersized shelf patches")
```
"""
function fill_small_shelf_patches!(mask::AbstractMatrix{Int}, min_cells::Int = 10)
    ny, nx = size(mask)
    visited = falses(ny, nx)
    n_filled = 0

    for i in 1:ny, j in 1:nx
        _is_active(mask[i, j]) && !visited[i, j] || continue

        # BFS to collect the full connected component.  Gaps (4) belong to the
        # component they sit in, so a gap never splits one shelf into two.
        component = Tuple{Int,Int}[]
        queue = Tuple{Int,Int}[(i, j)]
        visited[i, j] = true
        while !isempty(queue)
            c_i, cj = popfirst!(queue)
            push!(component, (c_i, cj))
            for (di, dj) in _cardinal_dirs
                ni, nj = c_i + di, cj + dj
                if 1 <= ni <= ny && 1 <= nj <= nx && !visited[ni, nj] && _is_active(mask[ni, nj])
                    visited[ni, nj] = true
                    push!(queue, (ni, nj))
                end
            end
        end

        if length(component) < min_cells
            for (c_i, cj) in component
                # Shelf becomes grounded; a gap has no ice to ground, so it reverts
                # to open ocean.
                mask[c_i, cj] = mask[c_i, cj] == 4 ? 0 : 2
            end
            n_filled += length(component)
        end
    end
    return n_filled
end

# ============================================================================
# Mask preprocessing pipeline
# ============================================================================

abstract type AbstractPreprocess end

struct FillOceanHolesPreprocess <: AbstractPreprocess end
struct FillShelfHolesPreprocess <: AbstractPreprocess end

@kwdef struct FillSmallShelfPatchesPreprocess <: AbstractPreprocess
    min_size::Int = 10
end

@kwdef struct FillSmallGroundedPatchesPreprocess <: AbstractPreprocess
    min_size::Int = 10
end

preprocess!(mask, ::FillOceanHolesPreprocess)       = fill_ocean_holes!(mask)
preprocess!(mask, ::FillShelfHolesPreprocess)        = fill_shelf_holes!(mask)
preprocess!(mask, p::FillSmallShelfPatchesPreprocess)    = fill_small_shelf_patches!(mask, p.min_size)
preprocess!(mask, p::FillSmallGroundedPatchesPreprocess) = fill_small_grounded_patches!(mask, p.min_size)

# ============================================================================
# Domain cropping
# ============================================================================

abstract type AbstractDomainCropping end

struct NoDomainCropping <: AbstractDomainCropping end

"""
$(TYPEDSIGNATURES)

Crop `mask`, `z_draft_raw`, and (optionally) `z_bed_raw` to the smallest rectangle
that contains all dynamically active cells (floating shelf `mask == 3` and gaps
`mask == 4`), expanded by `margin` cells in every direction.

`margin` must be at least 1, since the outermost ring has to stay free of active
cells (the stencils wrap periodically).  The default of 4 leaves a little context
around the cavity, which mostly matters for plotting — a tight crop puts the ice
front hard against the frame.  Use `margin = 1` for the tightest domain the
solver accepts.

The result is clipped to the input array, so a `margin` larger than the available
padding simply keeps what is there.  If the domain is already minimal, the arrays
are returned unchanged.

# Fields
 - `margin`: cells of padding kept around the active region (default 4, minimum 1).
"""
@kwdef struct MinRectangleDomainCropping <: AbstractDomainCropping
    margin::Int = 4
end

_crop_domain(mask, z_draft_raw, z_bed_raw, ::NoDomainCropping) = mask, z_draft_raw, z_bed_raw

function _crop_domain(mask, z_draft_raw, z_bed_raw, cropping::MinRectangleDomainCropping)
    margin = cropping.margin
    margin >= 1 || throw(
        ArgumentError(
            "MinRectangleDomainCropping margin must be at least 1 — the outermost " *
            "ring must stay free of active cells — got $margin",
        ),
    )
    # Gaps (4) are active cells too — cropping them away would silently remove the
    # very region the connected-gaps treatment is about.
    shelf_inds = findall(m -> m == 3 || m == 4, mask)
    isempty(shelf_inds) && return mask, z_draft_raw, z_bed_raw  # let validation catch it
    rows = getindex.(shelf_inds, 1)
    cols = getindex.(shelf_inds, 2)
    rmin, rmax = extrema(rows)
    cmin, cmax = extrema(cols)
    r = max(1, rmin - margin) : min(size(mask, 1), rmax + margin)
    c = max(1, cmin - margin) : min(size(mask, 2), cmax + margin)
    if length(r) < size(mask, 1) || length(c) < size(mask, 2)
        # Report the margin actually achieved on each side, not just the requested
        # one: the active region is rarely centred, so `margin` is clipped by the
        # array edge on whichever side runs out of room first and the padding ends
        # up asymmetric.
        pad = (rmin - first(r), last(r) - rmax, cmin - first(c), last(c) - cmax)
        note = all(==(margin), pad) ? "" :
               "  (clipped by the array edge; kept top/bottom/left/right = $pad)"
        @info "Domain cropped from $(size(mask)) to ($(length(r)), $(length(c))) " *
              "with margin = $margin" * note
    end
    new_zbed = z_bed_raw === nothing ? nothing : z_bed_raw[r, c]
    return mask[r, c], z_draft_raw[r, c], new_zbed
end