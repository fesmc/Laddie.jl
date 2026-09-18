"""
$(TYPEDEF)

All physical constants and parameterization choices of a [`Model`](@ref), in one
immutable, concretely typed struct.  Build it with the keyword constructor
`Params(; ...)`, which fills in the ISOMIP+-canonical defaults.
Time integration is configured on the [`Simulation`](@ref), and the boundary
conditions in [`BoundaryConditions`](@ref).

Rates follow the model's convention: `melt` and the entrainment
rates are freshwater-equivalent layer-thickness rates in m s⁻¹, and ``\\delta\\rho``
is the reduced density contrast ``(\\rho_a - \\rho)/\\rho_0``.

# Fields
$(TYPEDFIELDS)
"""
struct Params{
    FT,
    EP<:AbstractEntrainment,
    MP<:AbstractMelting,
    CS<:AbstractConvectionScheme,
    MLT<:AbstractMaxLayerThickness,
    LV<:AbstractLateralViscosity,
    LW<:AbstractLaplacianWeights,
    FP<:AbstractFrontPressure,
    MA<:AbstractMomentumAdvection,
    CP<:AbstractCoriolisParameter,
}
    # Dynamics
    "gravitational acceleration (m s⁻², default `9.81`)"
    g::FT
    "quadratic drag coefficient of the layer against the ice base, momentum equations (–, default `2.5e-3`)"
    C_d::FT
    "drag coefficient in the friction velocity u★ that drives melt and entrainment (–, default `1.1e-3`)"
    C_d_top::FT
    "lateral momentum viscosity, and the wall-drag coefficient under every viscosity scheme (m² s⁻¹, default `6`)"
    A_h::FT
    "lateral tracer diffusivity (m² s⁻¹, default `1`)"
    K_h::FT
    "upper bound on the detrainment rate (m s⁻¹, default `1e6`, i.e. unbounded)"
    max_detrainment::FT
    "lower bound on δρ in the buoyancy-entrainment production term, which divides by it (–, default `1e-4`)"
    drho_floor::FT
    "minimum layer thickness, kept by extra entrainment and a final clamp (m, default `1`)"
    D_min::FT
    "speed cap applied to the velocity after each momentum step (m s⁻¹, default `1.414`)"
    v_cut::FT
    # Thermodynamics
    "tidal velocity added in quadrature in the friction velocity (m s⁻¹, default `0.01`)"
    u_tide::FT
    "freshwater density, for the melt mass flux (kg m⁻³, default `1000`)"
    rho_freshwater::FT
    "reference seawater density ρ₀, which scales δρ (kg m⁻³, default `1028`)"
    rho0_seawater::FT
    "ice density (kg m⁻³, default `910`); for conversions by the caller, unused by the solver"
    rho_ice::FT
    "latent heat of fusion (J kg⁻¹, default `3.34e5`)"
    L::FT
    "specific heat capacity of seawater (J kg⁻¹ K⁻¹, default `3974`)"
    c_p::FT
    "specific heat capacity of ice (J kg⁻¹ K⁻¹, default `2009`)"
    c_i::FT
    # EOS (linear liquidus + thermal/haline expansion)
    "thermal expansion coefficient (K⁻¹, default `3.733e-5`)"
    alpha::FT
    "haline contraction coefficient (psu⁻¹, default `7.843e-4`)"
    beta::FT
    "liquidus slope in salinity, T_f = l1·S + l2 + l3·z (K psu⁻¹, default `-5.73e-2`)"
    l1::FT
    "liquidus offset (°C, default `8.32e-2`)"
    l2::FT
    "liquidus slope in depth (K m⁻¹, default `7.61e-4`)"
    l3::FT
    # Initialisation scalars
    "initial layer thickness (m, default `10`)"
    D_init::FT
    "initial temperature offset from the ambient water (K, default `0`)"
    dT_init::FT
    "initial salinity offset from the ambient water (psu, default `-0.1`)"
    dS_init::FT
    # Tracer bounds (applied in the active domain after each tracer step)
    "lower temperature bound (°C, default `-5`)"
    T_min::FT
    "upper temperature bound (°C, default `5`)"
    T_max::FT
    "lower salinity bound (psu, default `32`)"
    S_min::FT
    "upper salinity bound (psu, default `36`)"
    S_max::FT
    # Unit conversions
    "seconds per day, for the day-based simulation and output times (default `86400`)"
    seconds_per_day::FT
    "seconds per year, for rates reported in m yr⁻¹ (default `365.25 × 86400`)"
    seconds_per_year::FT
    # Typed parameterization objects
    "entrainment scheme, an [`AbstractEntrainment`](@ref) (default `LambertEntrainment(2.5)`)"
    entrainment::EP
    "melt scheme, an [`AbstractMelting`](@ref) (default `FixedGamTMelting(1.8e-4)`)"
    melting::MP
    "convective-instability treatment, an [`AbstractConvectionScheme`](@ref) (default `ResetToAmbient(0.005)`)"
    convection_scheme::CS
    "upper bound on the layer thickness, an [`AbstractMaxLayerThickness`](@ref) (default none)"
    max_layer_thickness::MLT
    "lateral momentum viscosity, an [`AbstractLateralViscosity`](@ref) (default constant `A_h`)"
    lateral_viscosity::LV
    "time level of the Laplacian thickness weights, an [`AbstractLaplacianWeights`](@ref)"
    laplacian_weights::LW
    "ice-front pressure-gradient treatment, an [`AbstractFrontPressure`](@ref)"
    front_pressure::FP
    "momentum-advection scheme, an [`AbstractMomentumAdvection`](@ref) (default centred, as in v1.x)"
    momentum_advection::MA
    "Coriolis parameter, an [`AbstractCoriolisParameter`](@ref) (default f-plane at `-1.37e-4` s⁻¹)"
    coriolis::CP
end

# Promote a parameterization object's floating-point fields to FT so it stays
# consistent with Params{FT} (e.g. Params(; FT = Float32, melting = FixedGamTMelting(...))
# where the default object was built at Float64).  Integer fields and field-less
# singletons pass through unchanged.  Generic over the field list, so new
# parameterization types are handled automatically; also applied to the boundary
# conditions and the time stepper.
_to_ft(v::AbstractFloat, ::Type{FT}) where {FT} = FT(v)
_to_ft(v, ::Type) = v
function _promote_param(x, ::Type{FT}) where {FT}
    fieldcount(typeof(x)) == 0 && return x
    ctor = Base.typename(typeof(x)).wrapper
    return ctor(ntuple(i -> _to_ft(getfield(x, i), FT), fieldcount(typeof(x)))...)
end

"""
$(TYPEDSIGNATURES)

Keyword constructor of [`Params`](@ref); every keyword is a field of the same
name (see there for units and defaults), and all scalars and the floating-point
fields of the parameterization objects are converted to `FT` (default `Float64`).
All parameters default to ISOMIP+-canonical values, so `Params()` is a valid
ready-to-use parameter set.  Override individual fields as needed:

```julia
params = Params(; coriolis = CoriolisParameter0D(0.0), melting = TurbulentGamTMelting(), FT = Float32)
```

`drho_floor` is the value the reference LADDIE code hard-wires.

`T_min`/`T_max` and `S_min`/`S_max` bound the layer temperature and salinity inside
the active domain after every tracer step.  They are a stability safeguard for real-world domains — the reference
LADDIE v2 has no such bounds — and they hide a diverging run from the blow-up
check, since bounded tracers keep the melt rate finite.  Widen them for fresher
settings such as Greenland fjords.
"""
function Params(;
    FT = Float64,
    g = 9.81,
    coriolis = CoriolisParameter0D(),
    C_d = 2.5e-3,
    C_d_top = 1.1e-3,
    A_h = 6.0,
    K_h = 1.0,
    max_detrainment = 1e6,
    drho_floor = 1e-4,
    D_min = 1.0,
    v_cut = 1.414,
    u_tide = 0.01,
    rho_freshwater = 1000.0,
    rho0_seawater = 1028.0,
    rho_ice = 910.0,
    L = 3.34e5,
    c_p = 3.974e3,
    c_i = 2009.0,
    alpha = 3.733e-5,
    beta = 7.843e-4,
    l1 = -5.73e-2,
    l2 = 8.32e-2,
    l3 = 7.61e-4,
    D_init = 10.0,
    dT_init = 0.0,
    dS_init = -0.1,
    T_min = -5.0,
    T_max = 5.0,
    S_min = 32.0,
    S_max = 36.0,
    seconds_per_day = 86400.0,
    seconds_per_year = 365.25 * 24 * 3600,
    entrainment = LambertEntrainment(2.5),
    melting = FixedGamTMelting(0.00018),
    convection_scheme = ResetToAmbient(0.005),
    max_layer_thickness = NoMaxLayerThickness(),
    lateral_viscosity = PrescribedLateralViscosity(),
    laplacian_weights = PresentLaplacianWeights(),
    front_pressure = FullDepthGradient(),
    momentum_advection = CentredMomentumAdvection(),
)
    # Keep every parameterization object's precision aligned with Params{FT}.
    entrainment = _promote_param(entrainment, FT)
    melting = _promote_param(melting, FT)
    convection_scheme = _promote_param(convection_scheme, FT)
    max_layer_thickness = _promote_param(max_layer_thickness, FT)
    lateral_viscosity = _promote_param(lateral_viscosity, FT)
    front_pressure = _promote_param(front_pressure, FT)
    momentum_advection = _promote_param(momentum_advection, FT)
    coriolis = _promote_param(coriolis, FT)
    Params(
        FT(g),
        FT(C_d),
        FT(C_d_top),
        FT(A_h),
        FT(K_h),
        FT(max_detrainment),
        FT(drho_floor),
        FT(D_min),
        FT(v_cut),
        FT(u_tide),
        FT(rho_freshwater),
        FT(rho0_seawater),
        FT(rho_ice),
        FT(L),
        FT(c_p),
        FT(c_i),
        FT(alpha),
        FT(beta),
        FT(l1),
        FT(l2),
        FT(l3),
        FT(D_init),
        FT(dT_init),
        FT(dS_init),
        FT(T_min),
        FT(T_max),
        FT(S_min),
        FT(S_max),
        FT(seconds_per_day),
        FT(seconds_per_year),
        entrainment,
        melting,
        convection_scheme,
        max_layer_thickness,
        lateral_viscosity,
        laplacian_weights,
        front_pressure,
        momentum_advection,
        coriolis,
    )
end
