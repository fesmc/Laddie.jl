using NCDatasets

# ============================================================================
# Typed forcing constructors
# ============================================================================

function LinearForcing(FT::Type, S0, S1, T1, forc_z0, l1, l2)
    z = FT.(-5000.0:1.0:-1.0)
    dz = FT(1.0);
    z0 = z[1]
    T0 = FT(l1) * FT(S0) + FT(l2)
    Tz = @. T0 + z * (FT(T1) - T0) / FT(forc_z0)
    Sz = @. FT(S0) + z * (FT(S1) - FT(S0)) / FT(forc_z0)
    LinearForcing(Tz, Sz, z, dz, z0, FT(S0), FT(S1), FT(T1), FT(forc_z0))
end

function Linear2Forcing(FT::Type, S0, S1, T1, forc_z0, l1, l2)
    z = FT.(-5000.0:1.0:-1.0)
    dz = FT(1.0);
    z0 = z[1]
    T0 = FT(l1) * FT(S0) + FT(l2)
    raw_T = @. T0 + z * (FT(T1) - T0) / FT(forc_z0)
    raw_S = @. FT(S0) + z * (FT(S1) - FT(S0)) / FT(forc_z0)
    Tz = FT(T1) > T0 ? min.(raw_T, FT(T1)) : max.(raw_T, FT(T1))
    Sz = min.(raw_S, FT(S1))
    Linear2Forcing(Tz, Sz, z, dz, z0, FT(S0), FT(S1), FT(T1), FT(forc_z0))
end

function TanhForcing(
    FT::Type,
    S0,
    T1,
    forc_z0,
    forc_z1,
    drho0,
    rho0_seawater,
    alpha,
    beta,
    l1,
    l2,
)
    z = FT.(-5000.0:1.0:-1.0)
    dz = FT(1.0);
    z0 = z[1]
    T0 = FT(l1) * FT(S0) + FT(l2)
    drho = FT(drho0) .* sqrt.(abs.(z))
    Tz = @. FT(T1) + (T0 - FT(T1)) * (1 + tanh((z - FT(forc_z0)) / FT(forc_z1))) / 2
    Sz = @. FT(S0) + FT(alpha) * (Tz - T0) / FT(beta) + drho / (FT(beta) * FT(rho0_seawater))
    TanhForcing(Tz, Sz, z, dz, z0, FT(S0), FT(T1), FT(forc_z0), FT(forc_z1), FT(drho0))
end

function _load_file_profiles(forcfile::String, forcfile_T::String, forcfile_S::String)
    if !isempty(forcfile) && isfile(forcfile)
        return NCDataset(forcfile, "r") do ds
            z = Float64.(Array(ds["z"][:]))
            Tz = Float64.(coalesce.(Array(ds["T"][:]), 0.0))
            Sz = Float64.(coalesce.(Array(ds["S"][:]), 34.0))
            z, Tz, Sz
        end
    end
    if !isempty(forcfile_T) || !isempty(forcfile_S)
        z, Tz = NCDataset(forcfile_T, "r") do ds
            z = Float64.(Array(ds["z"][:]))
            Tz = Float64.(coalesce.(Array(ds["temperature"][:]), 0.0))
            z, Tz
        end
        Sz = NCDataset(forcfile_S, "r") do ds
            Float64.(coalesce.(Array(ds["salinity"][:]), 34.0))
        end
        return z, Tz, Sz
    end
    error("Could not open forcing file(s): $forcfile")
end

function FileForcing(FT::Type, forcfile::String, forcfile_T::String, forcfile_S::String)
    z_raw, Tz_raw, Sz_raw = _load_file_profiles(forcfile, forcfile_T, forcfile_S)
    if !all(≈(1.0), diff(z_raw))
        z_new = collect(-5000.0:1.0:-1.0)
        Tz_raw = _interp1d(z_raw, Tz_raw, z_new)
        Sz_raw = _interp1d(z_raw, Sz_raw, z_new)
        z_raw = z_new
    end
    dz = FT(1.0);
    z0 = FT(z_raw[1])
    FileForcing(FT.(Tz_raw), FT.(Sz_raw), FT.(z_raw), dz, z0, forcfile)
end

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
    z_s = Float64.(z[p]);
    T_s = Float64.(Tz[p]);
    S_s = Float64.(Sz[p])
    # Drop duplicate depths (keep first occurrence) so _interp1d never divides by zero.
    keep = [k == 1 || z_s[k] != z_s[k-1] for k in eachindex(z_s)]
    z_s = z_s[keep];
    T_s = T_s[keep];
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

# Adjust zb before Grid construction.  zb is defined by mask category:
#   ocean  (0): 0        (no ice above, sea-surface reference)
#   border (1): 0        (not physically active)
#   grounded (2): z_bed  (ice base coincides with bed; keep zb_raw)
#   shelf  (3): zb_raw   (actual ice-base depth, clamped to ≤ -1 m)
# Any NaN fill values from the raw data are stripped before the mask logic.
function _adjust_zb(mask::AbstractMatrix{Int}, zb_raw::AbstractMatrix, FT)
    tmask = FT.(mask .== 3)
    zb = FT.(zb_raw)
    zb = ifelse.(isnan.(zb), zero(FT), zb)              # strip NaN fill values
    zb = ifelse.((mask .== 0) .| (mask .== 1), zero(FT), zb)  # ocean + border → 0
    zb = ifelse.((tmask .> 0) .& (zb .> FT(-1)), FT(-1), zb)  # clamp shallow shelf
    return zb
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

"""
$(TYPEDSIGNATURES)

Derive the 4-class LADDIE domain mask from BedMachine-style bed-elevation and
ice-thickness arrays.  Both arrays should cover the *interior* domain of size
`(ny, nx)`; the returned mask has size `(ny+2, nx+2)` with a one-cell border
ring of `1` (land/boundary).

| Value | Meaning         | Condition                          |
|-------|-----------------|------------------------------------|
| `0`   | open ocean      | `thickness ≤ 0`                    |
| `1`   | land / boundary | border ring                        |
| `2`   | grounded ice    | `thickness > 0` and `h_af ≥ 0`    |
| `3`   | floating shelf  | `thickness > 0` and `h_af < 0`    |

Height above flotation: `h_af = thickness × (rho_ice/rho_sw) + bed`.

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
        if h > 0
            mask[i+1, j+1] = (h * r + b >= 0) ? 2 : 3
        end
    end
    return mask
end

"""
$(TYPEDSIGNATURES)

Pad a bed-elevation array into the `(ny+2, nx+2)` format expected by `build_model`.
The one-cell border ring is zeroed; interior values are copied from `bed` unchanged.
Pass the result as the `z_bed_raw` keyword argument to `build_model` to enable the
water-column upper bound on plume thickness (`D ≤ zb − z_bed`).

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

- **Floating cells** (`h_af < 0`): `zb = −thickness × rho_ice/rho_sw` (Archimedes).
- **Grounded cells** (`h_af ≥ 0`): `zb = bed` (ice base rests on the bed).
- **Ocean / border cells**: `zb = 0`.

Pass the result directly as `zb_raw` to `build_model`; `_adjust_zb` will clamp
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
    zb = zeros(Float64, ny + 2, nx + 2)
    r = rho_ice / rho_sw
    for j = 1:nx, i = 1:ny
        h = Float64(thickness[i, j])
        b = Float64(bed[i, j])
        if h > 0
            zb[i+1, j+1] = (h * r + b >= 0) ? b : -h * r
        end
    end
    return zb
end

"""
$(TYPEDSIGNATURES)

Remove isolated ocean pockets from a LADDIE mask by flood-filling from the main
ocean.  Ocean cells (`mask == 0`) that are not connected (4-connectivity) to the
outer ocean are reclassified as grounded ice (`mask == 2`).

The outer ocean is identified as all ocean cells reachable from the border ring
(`mask == 1`).  Noisy topography (e.g. BedMachine) occasionally creates small
enclosed ocean patches fully surrounded by ice; these cause spurious ice-front
dynamics and numerical instabilities.

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

    # Seed: ocean cells touching the border ring (mask == 1)
    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for i in 1:ny, j in 1:nx
        if mask[i, j] == 0 && !visited[i, j]
            for (di, dj) in dirs
                ni, nj = i + di, j + dj
                if 1 <= ni <= ny && 1 <= nj <= nx && mask[ni, nj] == 1
                    visited[i, j] = true
                    push!(queue, (i, j))
                    break
                end
            end
        end
    end

    # BFS to mark all reachable ocean cells
    while !isempty(queue)
        i, j = popfirst!(queue)
        for (di, dj) in dirs
            ni, nj = i + di, j + dj
            if 1 <= ni <= ny && 1 <= nj <= nx && !visited[ni, nj] && mask[ni, nj] == 0
                visited[ni, nj] = true
                push!(queue, (ni, nj))
            end
        end
    end

    # Reclassify unreachable ocean cells as grounded ice
    n_filled = 0
    for i in 1:ny, j in 1:nx
        if mask[i, j] == 0 && !visited[i, j]
            mask[i, j] = 2
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

    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

    # Seed: shelf cells adjacent to at least one ocean cell
    for i in 1:ny, j in 1:nx
        if mask[i, j] == 3 && !visited[i, j]
            for (di, dj) in dirs
                ni, nj = i + di, j + dj
                if 1 <= ni <= ny && 1 <= nj <= nx && mask[ni, nj] == 0
                    visited[i, j] = true
                    push!(queue, (i, j))
                    break
                end
            end
        end
    end

    # BFS through shelf cells only
    while !isempty(queue)
        i, j = popfirst!(queue)
        for (di, dj) in dirs
            ni, nj = i + di, j + dj
            if 1 <= ni <= ny && 1 <= nj <= nx && !visited[ni, nj] && mask[ni, nj] == 3
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
to the domain border ring (`mask == 1`) is identified; any such component with
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
    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
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
            for (di, dj) in dirs
                ni, nj = c_i + di, cj + dj
                1 <= ni <= ny && 1 <= nj <= nx || continue
                mask[ni, nj] == 1 && (touches_border = true)
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
    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    n_filled = 0

    for i in 1:ny, j in 1:nx
        mask[i, j] == 3 && !visited[i, j] || continue

        # BFS to collect the full connected component
        component = Tuple{Int,Int}[]
        queue = Tuple{Int,Int}[(i, j)]
        visited[i, j] = true
        while !isempty(queue)
            c_i, cj = popfirst!(queue)
            push!(component, (c_i, cj))
            for (di, dj) in dirs
                ni, nj = c_i + di, cj + dj
                if 1 <= ni <= ny && 1 <= nj <= nx && !visited[ni, nj] && mask[ni, nj] == 3
                    visited[ni, nj] = true
                    push!(queue, (ni, nj))
                end
            end
        end

        if length(component) < min_cells
            for (c_i, cj) in component
                mask[c_i, cj] = 2
            end
            n_filled += length(component)
        end
    end
    return n_filled
end