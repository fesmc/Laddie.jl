# ============================================================================
# Params{FT,EP,MP,CS,OB,GL,TS} — all scalar physical constants + parameterization
# objects bundled in one immutable typed struct.
# ============================================================================

struct Params{
    FT,
    EP,     #<:AbstractEntrainment,
    MP,     #<:AbstractMeltParam,
    CS,     #<:AbstractConvectionScheme,
    OB,     #<:AbstractOpenBoundary,
    GL,     #<:AbstractGroundingLineBC,
    TS,     #<:AbstractTimeStepper,
    MLT,    #<:AbstractMaximumLayerThickness,
}
    # Time stepping (dt0 is the initial step; the runtime dt lives in IOState
    # so it can vary under adaptive time stepping — `m.dt` resolves there)
    dt0::FT
    nu::FT
    # Dynamics
    g::FT
    f::FT
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
    T_i::FT
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
    # Typed parameterization objects
    entrainment::EP
    melting::MP
    convection_scheme::CS
    open_bc::OB
    grline_bc::GL
    tstep::TS
    max_layer_thickness::MLT
end

# Promote a parameterization object's floating-point fields to FT so it stays
# consistent with Params{FT} (e.g. Params(; FT = Float32, melting = FixedGamT(...))
# where the default object was built at Float64).  Integer fields (such as
# AdaptiveDt's ncheck) and field-less singletons (open/grounding-line BCs) pass
# through unchanged.  Generic over the field list, so new parameterization types
# are handled automatically.
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
params = Params(; f = 0.0, melting = TurbulentGamT(), FT = Float32)
```
"""
function Params(;
    FT = Float64,
    dt = 210.0,
    nu = 0.8,
    g = 9.81,
    f = -1.37e-4,
    slip = 1.0,
    C_d = 2.5e-3,
    C_d_top = 1.1e-3,
    A_h = 6.0,
    K_h = 1.0,
    max_detrainment = 0.5,
    D_min = 1.0,
    v_cut = 1.414,
    u_tide = 0.01,
    T_i = -25.0,
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
    entrainment = LambertEntrainment(2.5),
    melting = FixedGamT(0.00018),
    convection_scheme = ResetToAmbient(0.005),
    open_bc = ZeroGradientInflow(),
    grline_bc = FreeSlipGL(),
    tstep = FixedDt(),
    max_layer_thickness = AbsoluteMaxLayerThickness(),
)
    # Keep every parameterization object's precision aligned with Params{FT}.
    entrainment = _promote_param(entrainment, FT)
    melting = _promote_param(melting, FT)
    convection_scheme = _promote_param(convection_scheme, FT)
    open_bc = _promote_param(open_bc, FT)
    grline_bc = _promote_param(grline_bc, FT)
    tstep = _promote_param(tstep, FT)
    max_layer_thickness = _promote_param(max_layer_thickness, FT)
    Params(
        FT(dt),
        FT(nu),
        FT(g),
        FT(f),
        FT(slip),
        FT(C_d),
        FT(C_d_top),
        FT(A_h),
        FT(K_h),
        FT(max_detrainment),
        FT(D_min),
        FT(v_cut),
        FT(u_tide),
        FT(T_i),
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
        entrainment,
        melting,
        convection_scheme,
        open_bc,
        grline_bc,
        tstep,
        max_layer_thickness,
    )
end
