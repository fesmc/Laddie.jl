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
    rho0,
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
    Sz = @. FT(S0) + FT(alpha) * (Tz - T0) / FT(beta) + drho / (FT(beta) * FT(rho0))
    TanhForcing(
        Tz,
        Sz,
        z,
        dz,
        z0,
        FT(S0),
        FT(T1),
        FT(forc_z0),
        FT(forc_z1),
        FT(drho0),
    )
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
function ProfileForcing(Tz::AbstractVector, Sz::AbstractVector, z::AbstractVector; FT = Float64)
    length(Tz) == length(z) || throw(ArgumentError(
        "Tz and z must have the same length, got $(length(Tz)) vs $(length(z))"))
    length(Sz) == length(z) || throw(ArgumentError(
        "Sz and z must have the same length, got $(length(Sz)) vs $(length(z))"))
    isempty(z) && throw(ArgumentError("profile vectors must be non-empty"))
    p = sortperm(Float64.(z))
    z_s = Float64.(z[p]); T_s = Float64.(Tz[p]); S_s = Float64.(Sz[p])
    # Drop duplicate depths (keep first occurrence) so _interp1d never divides by zero.
    keep = [k == 1 || z_s[k] != z_s[k-1] for k in eachindex(z_s)]
    z_s = z_s[keep]; T_s = T_s[keep]; S_s = S_s[keep]
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

# Adjust zb before Grid construction: zero out ice-front ocean cells and
# clamp very shallow ice-shelf cells to -1 m (mirrors initialize_from_scratch!
# in the old dict-bag flow, but applied pre-Grid so the immutable Grid is final).
function _adjust_zb(mask::AbstractMatrix{Int}, zb_raw::AbstractMatrix, FT)
    tmask = FT.(mask .== 3)
    ocn = FT.(mask .== 0)
    isf = ocn .* (xp1(tmask) .+ xm1(tmask) .+ yp1(tmask) .+ ym1(tmask))
    zb = FT.(zb_raw)
    zb = ifelse.(isf .> 0, zero(FT), zb)
    zb = ifelse.((tmask .> 0) .& (zb .> FT(-1)), FT(-1), zb)
    return zb
end

# Initialise prognostic fields from scratch (no restart file).
function _initialize_prognostics!(m)
    update_ambient_fields!(m)
    for level in (:past, :present, :future)
        setfield!(m.D, level, m.Dinit .* m.tmask)
        setfield!(m.T, level, (m.Ta .+ m.dTinit) .* m.tmask)
        setfield!(m.S, level, (m.Sa .+ m.dSinit) .* m.tmask)
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
    size(bed) == size(thickness) || throw(ArgumentError(
        "bed and thickness must have the same size, got $(size(bed)) vs $(size(thickness))"))
    mask = zeros(Int, ny + 2, nx + 2)
    mask[1, :]   .= 1
    mask[end, :] .= 1
    mask[:, 1]   .= 1
    mask[:, end] .= 1
    r = rho_ice / rho_sw
    for j in 1:nx, i in 1:ny
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
    size(bed) == size(thickness) || throw(ArgumentError(
        "bed and thickness must have the same size, got $(size(bed)) vs $(size(thickness))"))
    zb = zeros(Float64, ny + 2, nx + 2)
    r = rho_ice / rho_sw
    for j in 1:nx, i in 1:ny
        h = Float64(thickness[i, j])
        b = Float64(bed[i, j])
        if h > 0
            zb[i+1, j+1] = (h * r + b >= 0) ? b : -h * r
        end
    end
    return zb
end