# ============================================================================
# Params{FT,EP,MP,CS,MLT,LV,FP,CP} — all scalar physical constants +
# parameterization objects bundled in one immutable typed struct.  Nothing about
# time integration lives here (that belongs to the `Simulation`), nor the boundary
# conditions (a `BoundaryConditions` held by the `Model`).
# ============================================================================

struct Params{
    FT,
    EP,     #<:AbstractEntrainment,
    MP,     #<:AbstractMelting,
    CS,     #<:AbstractConvectionScheme,
    MLT,    #<:AbstractMaximumLayerThickness,
    LV,     #<:AbstractLateralViscosity,
    FP,     #<:AbstractFrontPressure,
    CP,     #<:AbstractCoriolisParameter,
}
    # Dynamics
    g::FT
    slip::FT
    C_d::FT
    C_d_top::FT
    A_h::FT
    K_h::FT
    max_detrainment::FT
    D_min::FT
    v_cut::FT
    # Thermodynamics
    u_tide::FT
    rho_freshwater::FT
    rho0_seawater::FT
    rho_ice::FT
    L::FT
    c_p::FT
    c_i::FT
    # EOS (linear liquidus + thermal/haline expansion)
    alpha::FT
    beta::FT
    l1::FT
    l2::FT
    l3::FT
    # Initialisation scalars
    D_init::FT
    dT_init::FT
    dS_init::FT
    # Unit conversions
    seconds_per_day::FT
    seconds_per_year::FT
    # Typed parameterization objects
    entrainment::EP
    melting::MP
    convection_scheme::CS
    max_layer_thickness::MLT
    lateral_viscosity::LV
    front_pressure::FP
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

Keyword-argument constructor; all scalar fields are converted to `FT`.
All parameters default to ISOMIP+-canonical values, so `Params()` is a valid
ready-to-use parameter set.  Override individual fields as needed:

```julia
params = Params(; coriolis = CoriolisParameter0D(0.0), melting = TurbulentGamTMelting(), FT = Float32)
```
"""
function Params(;
    FT = Float64,
    g = 9.81,
    coriolis = CoriolisParameter0D(),
    slip = 1.0,
    C_d = 2.5e-3,
    C_d_top = 1.1e-3,
    A_h = 6.0,
    K_h = 1.0,
    max_detrainment = 1e6,
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
    seconds_per_day = 86400.0,
    seconds_per_year = 365.25 * 24 * 3600,
    entrainment = LambertEntrainment(2.5),
    melting = FixedGamTMelting(0.00018),
    convection_scheme = ResetToAmbient(0.005),
    max_layer_thickness = NoMaxLayerThickness(),
    lateral_viscosity = PrescribedLateralViscosity(),
    front_pressure = FullDepthGradient(),
)
    # Keep every parameterization object's precision aligned with Params{FT}.
    entrainment = _promote_param(entrainment, FT)
    melting = _promote_param(melting, FT)
    convection_scheme = _promote_param(convection_scheme, FT)
    max_layer_thickness = _promote_param(max_layer_thickness, FT)
    lateral_viscosity = _promote_param(lateral_viscosity, FT)
    front_pressure = _promote_param(front_pressure, FT)
    coriolis = _promote_param(coriolis, FT)
    Params(
        FT(g),
        FT(slip),
        FT(C_d),
        FT(C_d_top),
        FT(A_h),
        FT(K_h),
        FT(max_detrainment),
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
        FT(seconds_per_day),
        FT(seconds_per_year),
        entrainment,
        melting,
        convection_scheme,
        max_layer_thickness,
        lateral_viscosity,
        front_pressure,
        coriolis,
    )
end
