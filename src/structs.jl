# Phase 3 typed structs: Grid, State, Cache.
# Included inside the Laddie module after Var, Center/Face, and the shift
# operators are defined so that the constructors can reference them.

# ============================================================================
# Grid{FT} — static geometry, masks, stagger-count denominators.
# ============================================================================

struct Grid{FT}
    Nx::Int;
    Ny::Int
    dx::FT;
    dy::FT

    mask::AbstractMatrix{Int}
    zb::AbstractMatrix{FT}
    dzdx::AbstractMatrix{FT}
    dzdy::AbstractMatrix{FT}

    tmask::AbstractMatrix{FT}
    grd::AbstractMatrix{FT}
    ocn::AbstractMatrix{FT}

    ocnym1::AbstractMatrix{FT};
    ocnyp1::AbstractMatrix{FT}
    ocnxm1::AbstractMatrix{FT};
    ocnxp1::AbstractMatrix{FT}

    tmaskym1::AbstractMatrix{FT};
    tmaskyp1::AbstractMatrix{FT}
    tmaskxm1::AbstractMatrix{FT};
    tmaskxp1::AbstractMatrix{FT}
    tmaskxm1ym1::AbstractMatrix{FT};
    tmaskxm1yp1::AbstractMatrix{FT}
    tmaskxp1ym1::AbstractMatrix{FT}

    grdNu::AbstractMatrix{FT};
    grdSu::AbstractMatrix{FT}
    grdEv::AbstractMatrix{FT};
    grdWv::AbstractMatrix{FT}
    isfE::AbstractMatrix{FT};
    isfW::AbstractMatrix{FT}
    isfN::AbstractMatrix{FT};
    isfS::AbstractMatrix{FT};
    isf::AbstractMatrix{FT}
    grlE::AbstractMatrix{FT};
    grlW::AbstractMatrix{FT}
    grlN::AbstractMatrix{FT};
    grlS::AbstractMatrix{FT};
    grl::AbstractMatrix{FT}

    umask::AbstractMatrix{FT};
    vmask::AbstractMatrix{FT}
    umaskym1::AbstractMatrix{FT};
    umaskyp1::AbstractMatrix{FT}
    umaskxm1::AbstractMatrix{FT};
    umaskxp1::AbstractMatrix{FT}
    vmaskym1::AbstractMatrix{FT};
    vmaskyp1::AbstractMatrix{FT}
    vmaskxm1::AbstractMatrix{FT};
    vmaskxp1::AbstractMatrix{FT}

    tmask_im::AbstractMatrix{FT};
    tmask_ip::AbstractMatrix{FT}
    tmask_jm::AbstractMatrix{FT};
    tmask_jp::AbstractMatrix{FT}
    umask_im::AbstractMatrix{FT};
    umask_ip::AbstractMatrix{FT}
    umask_jm::AbstractMatrix{FT};
    umask_jp::AbstractMatrix{FT}
    vmask_im::AbstractMatrix{FT};
    vmask_ip::AbstractMatrix{FT}
    vmask_jm::AbstractMatrix{FT};
    vmask_jp::AbstractMatrix{FT}
end

"""
    Grid(mask, zb, dx, dy; FT=Float64) → Grid{FT}

Build all masks and stagger-count denominators from the raw integer `mask` and
ice-draft array `zb`.  Mirrors the logic of `build_masks!` but returns an
immutable typed struct instead of populating a Dict-bag model.
"""
function Grid(mask::AbstractMatrix{Int}, zb::AbstractMatrix, dx, dy; FT = Float64)
    dx_ft = FT(dx);
    dy_ft = FT(dy)
    zb_ft = FT.(zb)
    dzdx = gradient_x(zb_ft, dx_ft)
    dzdy = gradient_y(zb_ft, dy_ft)

    # Primary classification
    tmask = FT.(mask .== 3)
    grd = FT.((mask .== 2) .| (mask .== 1))
    ocn = FT.(mask .== 0)

    ocnym1 = ym1(ocn);
    ocnyp1 = yp1(ocn)
    ocnxm1 = xm1(ocn);
    ocnxp1 = xp1(ocn)

    tmaskym1 = ym1(tmask);
    tmaskyp1 = yp1(tmask)
    tmaskxm1 = xm1(tmask);
    tmaskxp1 = xp1(tmask)
    tmaskxm1ym1 = ym1(xm1(tmask))
    tmaskxm1yp1 = yp1(xm1(tmask))
    tmaskxp1ym1 = ym1(xp1(tmask))

    # Boundary geometry
    o = one(FT)
    grdNu = o .- ym1((o .- grd) .* (o .- xm1(grd)))
    grdSu = o .- yp1((o .- grd) .* (o .- xm1(grd)))
    grdEv = o .- xm1((o .- grd) .* (o .- ym1(grd)))
    grdWv = o .- xp1((o .- grd) .* (o .- ym1(grd)))
    isfE = ocn .* tmaskxp1;
    isfW = ocn .* tmaskxm1
    isfN = ocn .* tmaskyp1;
    isfS = ocn .* tmaskym1
    isf = isfE .+ isfN .+ isfW .+ isfS
    grlE = grd .* tmaskxp1;
    grlW = grd .* tmaskxm1
    grlN = grd .* tmaskyp1;
    grlS = grd .* tmaskym1
    grl = grlE .+ grlN .+ grlW .+ grlS

    # Velocity masks
    umask = (tmask .+ isfW) .* (o .- xm1(grlE))
    vmask = (tmask .+ isfS) .* (o .- ym1(grlN))
    umaskym1 = ym1(umask);
    umaskyp1 = yp1(umask)
    umaskxm1 = xm1(umask);
    umaskxp1 = xp1(umask)
    vmaskym1 = ym1(vmask);
    vmaskyp1 = yp1(vmask)
    vmaskxm1 = xm1(vmask);
    vmaskxp1 = xp1(vmask)

    # Stagger-count denominators
    tmask_im = tmask .+ tmaskxp1;
    tmask_ip = tmask .+ tmaskxm1
    tmask_jm = tmask .+ tmaskyp1;
    tmask_jp = tmask .+ tmaskym1
    umask_im = umask .+ umaskxp1;
    umask_ip = umask .+ umaskxm1
    umask_jm = umask .+ umaskyp1;
    umask_jp = umask .+ umaskym1
    vmask_im = vmask .+ vmaskxp1;
    vmask_ip = vmask .+ vmaskxm1
    vmask_jm = vmask .+ vmaskyp1;
    vmask_jp = vmask .+ vmaskym1

    ny, nx = size(mask)
    Grid{FT}(
        nx,
        ny,
        dx_ft,
        dy_ft,
        mask,
        zb_ft,
        dzdx,
        dzdy,
        tmask,
        grd,
        ocn,
        ocnym1,
        ocnyp1,
        ocnxm1,
        ocnxp1,
        tmaskym1,
        tmaskyp1,
        tmaskxm1,
        tmaskxp1,
        tmaskxm1ym1,
        tmaskxm1yp1,
        tmaskxp1ym1,
        grdNu,
        grdSu,
        grdEv,
        grdWv,
        isfE,
        isfW,
        isfN,
        isfS,
        isf,
        grlE,
        grlW,
        grlN,
        grlS,
        grl,
        umask,
        vmask,
        umaskym1,
        umaskyp1,
        umaskxm1,
        umaskxp1,
        vmaskym1,
        vmaskyp1,
        vmaskxm1,
        vmaskxp1,
        tmask_im,
        tmask_ip,
        tmask_jm,
        tmask_jp,
        umask_im,
        umask_ip,
        umask_jm,
        umask_jp,
        vmask_im,
        vmask_ip,
        vmask_jm,
        vmask_jp,
    )
end

# ============================================================================
# State{FT} — five prognostic leapfrog variables.
# ============================================================================

mutable struct State{FT}
    D::Var{Center,Center,FT}
    U::Var{Face,Center,FT}
    V::Var{Center,Face,FT}
    T::Var{Center,Center,FT}
    S::Var{Center,Center,FT}
end

State(FT::Type, ny::Int, nx::Int) = State(
    Var(Center, Center, FT, ny, nx),
    Var(Face, Center, FT, ny, nx),
    Var(Center, Face, FT, ny, nx),
    Var(Center, Center, FT, ny, nx),
    Var(Center, Center, FT, ny, nx),
)

# ============================================================================
# Cache{FT, GamT, Conv2} — pre-allocated diagnostic + stencil scratch arrays.
# GamT is FT for FixedGamT, AbstractMatrix{FT} for TurbulentGamT.
# Conv2 is FT for ClampDensity/ResetToAmbient, AbstractMatrix{FT} for RelaxToAmbient.
# ============================================================================

mutable struct Cache{FT, GamT, Conv2}
    # Physics outputs
    melt::AbstractMatrix{FT};
    Tb::AbstractMatrix{FT};
    Tf::AbstractMatrix{FT}
    Ta::AbstractMatrix{FT};
    Sa::AbstractMatrix{FT};
    drho::AbstractMatrix{FT}
    ustar::AbstractMatrix{FT};
    entr::AbstractMatrix{FT};
    detr::AbstractMatrix{FT}
    ent2::AbstractMatrix{FT};
    nentr::AbstractMatrix{FT};
    ent::AbstractMatrix{FT}
    Sb::AbstractMatrix{FT};
    drhob::AbstractMatrix{FT};
    convection::AbstractMatrix{FT}
    convD::AbstractMatrix{FT};
    dDdt::AbstractMatrix{FT};
    Ddrho::AbstractMatrix{FT}
    conv2::Conv2
    gamT::GamT
    gamS::GamT
    # Advection stencil temporaries
    Dym1::AbstractMatrix{FT};
    Dyp1::AbstractMatrix{FT}
    Dxm1::AbstractMatrix{FT};
    Dxp1::AbstractMatrix{FT}
    Dxm1ym1::AbstractMatrix{FT};
    Dxp1ym1::AbstractMatrix{FT};
    Dxm1yp1::AbstractMatrix{FT}
    Vyp1::AbstractMatrix{FT};
    Uxp1::AbstractMatrix{FT}
    Upos::AbstractMatrix{FT};
    Uneg::AbstractMatrix{FT}
    Vpos::AbstractMatrix{FT};
    Vneg::AbstractMatrix{FT}
    Vyp1pos::AbstractMatrix{FT};
    Vyp1neg::AbstractMatrix{FT}
    Uxp1pos::AbstractMatrix{FT};
    Uxp1neg::AbstractMatrix{FT}
    Vip::AbstractMatrix{FT};
    Vim::AbstractMatrix{FT}
    Uip::AbstractMatrix{FT};
    Uim::AbstractMatrix{FT}
    Vjp::AbstractMatrix{FT};
    Vjm::AbstractMatrix{FT}
    Ujp::AbstractMatrix{FT};
    Ujm::AbstractMatrix{FT}
    signU::AbstractMatrix{FT};
    signV::AbstractMatrix{FT}
    # Laplacian stencil temporaries
    D0ip::AbstractMatrix{FT};
    D0im::AbstractMatrix{FT}
    D0jp::AbstractMatrix{FT};
    D0jm::AbstractMatrix{FT}
    # lapU/lapV thickness grids (D0ip/D0jp * tmask, precomputed with laplacian stencils)
    D_on_ugrid::AbstractMatrix{FT};
    D_on_vgrid::AbstractMatrix{FT}
    # Stencil output arrays (eliminate similar() per step)
    cU::AbstractMatrix{FT};
    cV::AbstractMatrix{FT}
    cT::AbstractMatrix{FT};
    cS::AbstractMatrix{FT}
    lU::AbstractMatrix{FT};
    lV::AbstractMatrix{FT}
    lT::AbstractMatrix{FT};
    lS::AbstractMatrix{FT}
    # Tracer flux intermediates (eliminate D*q allocation per step)
    DT::AbstractMatrix{FT};
    DS::AbstractMatrix{FT}
end

_gamT_init(FT, _,  _,  ::Type{<:FixedGamT})     = zero(FT)
_gamT_init(FT, ny, nx, ::Type{<:TurbulentGamT}) = zeros(FT, ny, nx)
_conv2_init(FT, _,  _,  ::Type{<:Union{ClampDensity,ResetToAmbient}}) = zero(FT)
_conv2_init(FT, ny, nx, ::Type{<:RelaxToAmbient})                     = zeros(FT, ny, nx)

"""
    Cache(FT, MP, CS, ny, nx) → Cache{FT, GamT, Conv2}

Allocate all scratch matrices for a grid of size `(ny, nx)`.
`MP` (melt param type) determines whether `gamT`/`gamS` are scalars or arrays;
`CS` (convection scheme type) determines whether `conv2` is a scalar or array.
"""
function Cache(FT::Type, MP::Type, CS::Type, ny::Int, nx::Int)
    z         = zeros(FT, ny, nx)
    gamT_init = _gamT_init(FT, ny, nx, MP)
    gamS_init = _gamT_init(FT, ny, nx, MP)
    conv2_init = _conv2_init(FT, ny, nx, CS)
    GamT  = typeof(gamT_init)
    Conv2 = typeof(conv2_init)
    Cache{FT,GamT,Conv2}(
        copy(z),
        copy(z),
        copy(z),          # melt, Tb, Tf
        copy(z),
        copy(z),
        copy(z),          # Ta, Sa, drho
        copy(z),
        copy(z),
        copy(z),          # ustar, entr, detr
        copy(z),
        copy(z),
        copy(z),          # ent2, nentr, ent
        copy(z),
        copy(z),
        copy(z),          # Sb, drhob, convection
        copy(z),
        copy(z),
        copy(z),          # convD, dDdt, Ddrho
        conv2_init,
        gamT_init,
        gamS_init,                  # conv2, gamT, gamS
        copy(z),
        copy(z),                   # Dym1, Dyp1
        copy(z),
        copy(z),                   # Dxm1, Dxp1
        copy(z),
        copy(z),
        copy(z),          # Dxm1ym1, Dxp1ym1, Dxm1yp1
        copy(z),
        copy(z),                   # Vyp1, Uxp1
        copy(z),
        copy(z),
        copy(z),
        copy(z), # Upos, Uneg, Vpos, Vneg
        copy(z),
        copy(z),
        copy(z),
        copy(z), # Vyp1pos, Vyp1neg, Uxp1pos, Uxp1neg
        copy(z),
        copy(z),
        copy(z),
        copy(z), # Vip, Vim, Uip, Uim
        copy(z),
        copy(z),
        copy(z),
        copy(z), # Vjp, Vjm, Ujp, Ujm
        copy(z),
        copy(z),                   # signU, signV
        copy(z),
        copy(z),
        copy(z),
        copy(z), # D0ip, D0im, D0jp, D0jm
        copy(z), copy(z),                   # D_on_ugrid, D_on_vgrid
        copy(z), copy(z), copy(z), copy(z), # cU, cV, cT, cS
        copy(z), copy(z), copy(z), copy(z), # lU, lV, lT, lS
        copy(z), copy(z),                   # DT, DS
    )
end

# ============================================================================
# RunConfig — static run + I/O configuration (all fields have defaults).
# ============================================================================

"""
    RunConfig(; name, days, saveday, ...) → RunConfig

Static configuration for a model run: duration, I/O cadence, output flags,
and restart options.  All fields have sensible defaults; a plain `RunConfig()`
disables file I/O (`saveday = 0`).

Set `saveday > 0` to enable NetCDF output at that interval (days).
"""
Base.@kwdef struct RunConfig
    name        :: String  = "run"
    days        :: Float64 = 30.0
    saveday     :: Float64 = 0.0     # 0 = I/O disabled
    diagday     :: Float64 = 1.0
    restday     :: Float64 = 30.0
    resultdir   :: String  = "./output/"
    logfilename :: String  = "log.txt"
    forcenewdir :: Bool    = true
    fromrestart :: Bool    = false
    restartfile :: String  = ""
    save_Ut     :: Bool    = true
    save_Uu     :: Bool    = false
    save_Vt     :: Bool    = true
    save_Vv     :: Bool    = false
    save_D      :: Bool    = true
    save_T      :: Bool    = true
    save_S      :: Bool    = true
    save_melt   :: Bool    = true
    save_entr   :: Bool    = false
    save_ent2   :: Bool    = false
    save_detr   :: Bool    = false
    save_Tbase  :: Bool    = false
    save_Tamb   :: Bool    = false
    save_gammaT :: Bool    = false
    save_mask   :: Bool    = true
    save_zb     :: Bool    = true
end

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
    Params(; FT=Float64, ...) → Params

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
    EP = typeof(entpar);
    MP = typeof(meltpar)
    CS = typeof(convpar);
    OB = typeof(openbc)
    Params{FT,EP,MP,CS,OB}(
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
    )
end

"""
    Model{FT,F,EP,MP,CS,OB}

The top-level model container.  Constructed by `build_isomip`; advanced by `run!`.

Fields are accessed directly on `m` through a flat forwarding layer:

| Access pattern | Source struct | Examples |
|----------------|--------------|---------|
| `m.D`, `m.U`, `m.V`, `m.T`, `m.S` | `State` | `m.D.present`, `m.U.past` |
| `m.melt`, `m.entr`, `m.drho`, `m.Ta`, `m.Sa`, … | `Cache` | `m.melt .* spy` |
| `m.tmask`, `m.zb`, `m.dx`, `m.dy`, … | `Grid` | `m.tmask .> 0` |
| `m.dt`, `m.f`, `m.Cd`, `m.Ah`, `m.minD`, … | `Params` | `m.dt` |
| `m.name`, `m.saveday`, `m.save_D`, … | `RunConfig` | `m.rc.saveday` |
| `m.Tz`, `m.Sz`, `m.z` | Forcing | ambient profile arrays |
`m.FT` returns the floating-point type (`Float64` or `Float32`).

`Grid` and `Params` are immutable after construction.  `Cache` and `State`
fields are mutable and updated in place each time step.
"""
mutable struct Model{
    FT,
    F<:AbstractForcing,
    EP<:AbstractEntrainmentParam,
    MP<:AbstractMeltParam,
    CS<:AbstractConvectionScheme,
    OB<:AbstractOpenBoundary,
    C<:Cache,
}
    d   :: Dict{Symbol,Any}   # runtime I/O state (counters, accumulators, coords)
    rc  :: RunConfig
    grid:: Grid{FT}
    state::State{FT}
    cache::C
    params::Params{FT,EP,MP,CS,OB}
    forcing::F
end

function Base.getproperty(m::Model{FT}, k::Symbol) where {FT}
    # Direct struct fields — fast path
    k === :d       && return getfield(m, :d)
    k === :rc      && return getfield(m, :rc)
    k === :grid    && return getfield(m, :grid)
    k === :state   && return getfield(m, :state)
    k === :cache   && return getfield(m, :cache)
    k === :params  && return getfield(m, :params)
    k === :forcing && return getfield(m, :forcing)
    k === :FT      && return FT
    # Interior dimensions derived from grid (total minus 2 border cells)
    k === :ny && return getfield(m, :grid).Ny - 2
    k === :nx && return getfield(m, :grid).Nx - 2
    # Grid: geometry, masks, stagger denominators
    g = getfield(m, :grid)
    hasfield(typeof(g), k) && return getfield(g, k)
    # State: prognostic Var objects
    s = getfield(m, :state)
    hasfield(typeof(s), k) && return getfield(s, k)
    # Cache: mutable scratch / diagnostic arrays
    c = getfield(m, :cache)
    hasfield(typeof(c), k) && return getfield(c, k)
    # Params: physical constants + parameterization objects
    p = getfield(m, :params)
    hasfield(typeof(p), k) && return getfield(p, k)
    # RunConfig: static run + I/O configuration
    r = getfield(m, :rc)
    hasfield(typeof(r), k) && return getfield(r, k)
    # Forcing: ambient T/S profiles on the uniform z-grid
    f = getfield(m, :forcing)
    hasfield(typeof(f), k) && return getfield(f, k)
    # Dict fallback: runtime I/O state (counters, accumulators, coordinates)
    return getfield(m, :d)[k]
end

function Base.setproperty!(m::Model, k::Symbol, v)
    # Cache is mutable — all physics scratch arrays live here
    c = getfield(m, :cache)
    if hasfield(typeof(c), k)
        setfield!(c, k, v)
        return
    end
    # State is mutable — D/U/V/T/S Var objects can be replaced
    s = getfield(m, :state)
    if hasfield(typeof(s), k)
        setfield!(s, k, v)
        return
    end
    # Dict fallback for I/O fields and mutable geometry (zb etc.)
    getfield(m, :d)[k] = v
end

Base.haskey(m::Model, k::Symbol) = haskey(getfield(m, :d), k)
