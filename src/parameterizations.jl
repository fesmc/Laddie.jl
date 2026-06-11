# Parameterization type hierarchy — Phase 2
# Multiple dispatch replaces all symbol/integer-flag if-elseif switches in physics.

abstract type AbstractEntrainmentParam end

"""
    GasparEntrainment(mu)

Buoyancy-flux-driven entrainment following Gaspar (1988) as used in
Gladish et al. (2012) and Lambert et al. (2023, Eq. 11).

- `mu`: dimensionless efficiency parameter (ISOMIP+ default: `2.5`).

Select via `entpar = "Gaspar"` and `mu = 2.5` in the `[Parameters]` config section.
"""
struct GasparEntrainment{FT} <: AbstractEntrainmentParam
    ;
    mu::FT;
end

"""
    HollandEntrainment(cl)

Shear-driven entrainment following Holland & Jenkins (1999).

- `cl`: drag coefficient for entrainment velocity (default: `0.01775`).

Select via `entpar = "Holland"` in the `[Parameters]` config section.
"""
struct HollandEntrainment{FT} <: AbstractEntrainmentParam
    ;
    cl::FT;
end

abstract type AbstractMeltParam end

"""
    FixedGamT(gamTfix)

Three-equation ice–ocean melt parameterisation (Jenkins 1991) with a constant
turbulent heat transfer coefficient γ_T.

- `gamTfix`: heat transfer coefficient (ISOMIP+ default: `1.8e-4`).

Select via `usegamtfix = true` and `gamTfix = 0.00018` in `[Parameters]`.
"""
struct FixedGamT{FT} <: AbstractMeltParam
    ;
    gamTfix::FT;
end

"""
    TurbulentGamT(Pr, Sc, nu0)

Three-equation melt parameterisation with turbulence-dependent transfer
coefficients γ_T and γ_S via the log-layer formulation
(Holland & Jenkins 1999; Lambert et al. 2023, Eqs. 8–10).

- `Pr`:  Prandtl number (default `13.8`).
- `Sc`:  Schmidt number (default `2432.0`).
- `nu0`: molecular kinematic viscosity, m² s⁻¹ (default `1.95e-6`).

Select via `usegamtfix = false` in `[Parameters]`; tune `Pr`, `Sc`, `nu0`
in `[Constants]`.
"""
struct TurbulentGamT{FT} <: AbstractMeltParam
    ;
    Pr::FT;
    Sc::FT;
    nu0::FT;
end

abstract type AbstractConvectionScheme end

"""
    ClampDensity(mindrho)

Handle convective instability (δρ < 0) by clamping the density contrast to a
minimum positive value so the plume remains denser than ambient.

- `mindrho`: minimum density contrast, kg m⁻³ (default `0.005`).

Select via `convop = 0` in `[Options]`; tune via `mindrho` in `[Convection]`.
"""
struct ClampDensity{FT} <: AbstractConvectionScheme
    ;
    mindrho::FT;
end

"""
    ResetToAmbient(mindrho)

Handle convective instability by instantly resetting T and S of unstable cells
to their ambient values, restoring a stable density contrast.

- `mindrho`: threshold density contrast triggering the reset, kg m⁻³ (default `0.005`).

Select via `convop = 1` in `[Options]`.
"""
struct ResetToAmbient{FT} <: AbstractConvectionScheme
    ;
    mindrho::FT;
end

"""
    RelaxToAmbient(convtime)

Handle convective instability by relaxing T and S of unstable cells toward
ambient values over a prescribed timescale (applied implicitly in the tracer
time step via the `conv2` term).

- `convtime`: relaxation timescale, s (default `10000.0`).

Select via `convop = 2` in `[Options]`; tune via `convtime` in `[Convection]`.
"""
struct RelaxToAmbient{FT} <: AbstractConvectionScheme
    ;
    convtime::FT;
end

abstract type AbstractOpenBoundary end

"""
    ZeroGradientInflow()

Open-boundary condition at the ice front: zero-gradient extrapolation of all
fields, with inflow from the ambient ocean permitted.

Select via `boundop = 1` in `[Options]`.
"""
struct ZeroGradientInflow <: AbstractOpenBoundary end

"""
    NoInflow()

Open-boundary condition at the ice front: outflow only — inflow velocities are
clipped to zero so ambient water cannot advect into the domain.

Select via `boundop = 0` (or any value other than `1`) in `[Options]`.
"""
struct NoInflow <: AbstractOpenBoundary end

# ============================================================================
# Forcing types — carry the computed T/S profiles on a uniform z-grid.
# AbstractVector fields (not Vector) allow the same struct to hold CPU or GPU
# arrays after to_backend!.
# Constructors for types that require raw parameters (LinearForcing, etc.) or
# file I/O (FileForcing) are defined in geometry.jl where the computation
# logic already lives.
# ============================================================================

abstract type AbstractForcing end

"""
    ISOMIPForcing(FT, isomipcond)

Ambient T/S profiles for the ISOMIP+ protocol (Asay-Davis et al. 2016).
Profiles are linear from the surface (T = −1.9 °C, S = 33.8 psu) to 720 m depth.

- `isomipcond = :warm`: T = +1.0 °C, S = 34.7 psu at depth (strong melting).
- `isomipcond = :cold`: T = −1.9 °C, S = 34.55 psu at depth (near-freezing).

Used automatically by `build_isomip`; select via `option = "isomip"` in `[Forcing]`.
"""
struct ISOMIPForcing{FT} <: AbstractForcing
    Tz::AbstractVector{FT}
    Sz::AbstractVector{FT}
    z::AbstractVector{FT}
    dz::FT
    z0::FT
    isomipcond::Symbol
end

function ISOMIPForcing(FT::Type, isomipcond::Symbol)
    z = FT.(-5000.0:1.0:-1.0)
    dz = FT(1.0)
    z0 = z[1]
    z_pyc = FT(-720.0)
    T_surface = FT(-1.9)
    S_surface = FT(33.8)
    T_deep, S_deep = isomipcond == :warm ? (FT(1.0), FT(34.7)) : (FT(-1.9), FT(34.55))
    Tz = @. T_surface + z * (T_deep - T_surface) / z_pyc
    Sz = @. S_surface + z * (S_deep - S_surface) / z_pyc
    ISOMIPForcing{FT}(Tz, Sz, z, dz, z0, isomipcond)
end

"""
    LinearForcing

Ambient T/S profiles that vary linearly with depth from the surface
values (T at freezing, S = `S0`) to (`T1`, `S1`) at depth `z0`.

Select via `option = "linear"` in `[Forcing]`.
"""
struct LinearForcing{FT} <: AbstractForcing
    Tz::AbstractVector{FT};
    Sz::AbstractVector{FT};
    z::AbstractVector{FT};
    dz::FT;
    z0::FT
    S0::FT;
    S1::FT;
    T1::FT;
    forc_z0::FT
end

"""
    Linear2Forcing

Like `LinearForcing` but the profiles are capped at the surface value so they
do not extrapolate beyond (`T1`, `S1`) above `z0`.

Select via `option = "linear2"` in `[Forcing]`.
"""
struct Linear2Forcing{FT} <: AbstractForcing
    Tz::AbstractVector{FT};
    Sz::AbstractVector{FT};
    z::AbstractVector{FT};
    dz::FT;
    z0::FT
    S0::FT;
    S1::FT;
    T1::FT;
    forc_z0::FT
end

"""
    TanhForcing

Ambient T/S profiles with a tanh transition between cold surface waters and
warm deep waters, with an additional density perturbation `drho0·√|z|`.

Select via `option = "tanh"` in `[Forcing]`.
"""
struct TanhForcing{FT} <: AbstractForcing
    Tz::AbstractVector{FT};
    Sz::AbstractVector{FT};
    z::AbstractVector{FT};
    dz::FT;
    z0::FT
    S0::FT;
    T1::FT;
    forc_z0::FT;
    forc_z1::FT;
    drho0::FT
end

"""
    FileForcing

Ambient T/S profiles loaded from a NetCDF file and resampled to a 1-m depth
grid.  Accepts either a single file with `z`, `T`, `S` variables, or two
separate files (one for T, one for S).

Select via `option = "file"` in `[Forcing]` and set `filename` (or
`filename_T` + `filename_S`).
"""
struct FileForcing{FT} <: AbstractForcing
    Tz::AbstractVector{FT};
    Sz::AbstractVector{FT};
    z::AbstractVector{FT};
    dz::FT;
    z0::FT
    path::String
end
