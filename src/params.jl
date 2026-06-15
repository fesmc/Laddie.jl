# ============================================================================
# Params{FT,EP,MP,CS,OB,GL,TS} — all scalar physical constants + parameterization
# objects bundled in one immutable typed struct.
# ============================================================================

struct Params{
    FT,
    EP<:AbstractEntrainmentParam,
    MP<:AbstractMeltParam,
    CS<:AbstractConvectionScheme,
    OB<:AbstractOpenBoundary,
    GL<:AbstractGroundingLineBC,
    TS<:AbstractTimeStepper,
}
    # Time stepping (dt0 is the initial step; the runtime dt lives in IOState
    # so it can vary under adaptive time stepping — `m.dt` resolves there)
    dt0::FT
    nu::FT
    # Dynamics
    g::FT
    f::FT
    slip::FT
    Cd::FT
    Cdtop::FT
    Ah::FT
    Kh::FT
    maxdetr::FT
    minD::FT
    vcut::FT
    # Thermodynamics
    utide::FT
    Ti::FT
    rhofw::FT
    rho0::FT
    rhoi::FT
    L::FT
    cp::FT
    ci::FT
    # EOS (linear liquidus + thermal/haline expansion)
    alpha::FT
    beta::FT
    l1::FT
    l2::FT
    l3::FT
    # Initialisation scalars
    Dinit::FT
    dTinit::FT
    dSinit::FT
    # Typed parameterization objects
    entpar::EP
    meltpar::MP
    convpar::CS
    openbc::OB
    glbc::GL
    tstep::TS
end

# Promote a parameterization object's floating-point fields to FT so it stays
# consistent with Params{FT} (e.g. Params(; FT = Float32, meltpar = FixedGamT(...))
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
params = Params(; f = 0.0, meltpar = TurbulentGamT(), FT = Float32)
```
"""
function Params(;
    FT = Float64,
    dt = 210.0,
    nu = 0.8,
    g = 9.81,
    f = -1.37e-4,
    slip = 1.0,
    Cd = 2.5e-3,
    Cdtop = 1.1e-3,
    Ah = 6.0,
    Kh = 1.0,
    maxdetr = 0.5,
    minD = 1.0,
    vcut = 1.414,
    utide = 0.01,
    Ti = -25.0,
    rhofw = 1000.0,
    rho0 = 1028.0,
    rhoi = 910.0,
    L = 3.34e5,
    cp = 3.974e3,
    ci = 2009.0,
    alpha = 3.733e-5,
    beta = 7.843e-4,
    l1 = -5.73e-2,
    l2 = 8.32e-2,
    l3 = 7.61e-4,
    Dinit = 10.0,
    dTinit = 0.0,
    dSinit = -0.1,
    entpar = GasparEntrainment(2.5),
    meltpar = FixedGamT(0.00018),
    convpar = ResetToAmbient(0.005),
    openbc = ZeroGradientInflow(),
    glbc = FreeSlipGL(),
    tstep = FixedDt(),
)
    # Keep every parameterization object's precision aligned with Params{FT}.
    entpar = _promote_param(entpar, FT)
    meltpar = _promote_param(meltpar, FT)
    convpar = _promote_param(convpar, FT)
    openbc = _promote_param(openbc, FT)
    glbc = _promote_param(glbc, FT)
    tstep = _promote_param(tstep, FT)
    EP = typeof(entpar);
    MP = typeof(meltpar)
    CS = typeof(convpar);
    OB = typeof(openbc)
    GL = typeof(glbc);
    TS = typeof(tstep)
    Params{FT,EP,MP,CS,OB,GL,TS}(
        FT(dt),
        FT(nu),
        FT(g),
        FT(f),
        FT(slip),
        FT(Cd),
        FT(Cdtop),
        FT(Ah),
        FT(Kh),
        FT(maxdetr),
        FT(minD),
        FT(vcut),
        FT(utide),
        FT(Ti),
        FT(rhofw),
        FT(rho0),
        FT(rhoi),
        FT(L),
        FT(cp),
        FT(ci),
        FT(alpha),
        FT(beta),
        FT(l1),
        FT(l2),
        FT(l3),
        FT(Dinit),
        FT(dTinit),
        FT(dSinit),
        entpar,
        meltpar,
        convpar,
        openbc,
        glbc,
        tstep,
    )
end
