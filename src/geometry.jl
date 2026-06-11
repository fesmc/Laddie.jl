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
    LinearForcing{FT}(Tz, Sz, z, dz, z0, FT(S0), FT(S1), FT(T1), FT(forc_z0))
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
    Linear2Forcing{FT}(Tz, Sz, z, dz, z0, FT(S0), FT(S1), FT(T1), FT(forc_z0))
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
    TanhForcing{FT}(
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
    FileForcing{FT}(FT.(Tz_raw), FT.(Sz_raw), FT.(z_raw), dz, z0, forcfile)
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
