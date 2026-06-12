# ============================================================================
# Params{FT,EP,MP,CS,OB} — all scalar physical constants + parameterization
# objects bundled in one immutable typed struct.
# ============================================================================

struct Params{
    FT,
    EP<:AbstractEntrainmentParam,
    MP<:AbstractMeltParam,
    CS<:AbstractConvectionScheme,
    OB<:AbstractOpenBoundary,
}
    # Time stepping
    dt::FT
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
    FT      = Float64,
    dt      = 210.0,
    nu      = 0.8,
    g       = 9.81,
    f       = -1.37e-4,
    slip    = 1.0,
    Cd      = 2.5e-3,
    Cdtop   = 1.1e-3,
    Ah      = 6.0,
    Kh      = 1.0,
    maxdetr = 0.5,
    minD    = 1.0,
    vcut    = 1.414,
    utide   = 0.01,
    Ti      = -25.0,
    rhofw   = 1000.0,
    rho0    = 1028.0,
    rhoi    = 910.0,
    L       = 3.34e5,
    cp      = 3.974e3,
    ci      = 2009.0,
    alpha   = 3.733e-5,
    beta    = 7.843e-4,
    l1      = -5.73e-2,
    l2      =  8.32e-2,
    l3      =  7.61e-4,
    Dinit   = 10.0,
    dTinit  = 0.0,
    dSinit  = -0.1,
    entpar  = GasparEntrainment(2.5),
    meltpar = FixedGamT(0.00018),
    convpar = ResetToAmbient(0.005),
    openbc  = ZeroGradientInflow(),
)
    EP = typeof(entpar); MP = typeof(meltpar)
    CS = typeof(convpar); OB = typeof(openbc)
    Params{FT, EP, MP, CS, OB}(
        FT(dt), FT(nu), FT(g), FT(f), FT(slip), FT(Cd), FT(Cdtop),
        FT(Ah), FT(Kh), FT(maxdetr), FT(minD), FT(vcut),
        FT(utide), FT(Ti), FT(rhofw), FT(rho0), FT(rhoi),
        FT(L), FT(cp), FT(ci),
        FT(alpha), FT(beta), FT(l1), FT(l2), FT(l3),
        FT(Dinit), FT(dTinit), FT(dSinit),
        entpar, meltpar, convpar, openbc,
    )
end