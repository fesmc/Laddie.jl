# ============================================================================
# Cache{FT, A, GamT, Conv2} — pre-allocated diagnostic + stencil scratch arrays.
# A    is the concrete matrix type (matches Grid and State).
# GamT is FT for FixedGamT, A for TurbulentGamT.
# Conv2 is FT for ClampDensity/ResetToAmbient, A for RelaxToAmbient.
# ============================================================================

mutable struct Cache{FT,A<:AbstractMatrix{FT},GamT,Conv2}
    # Physics outputs
    melt::A;
    Tb::A;
    Tf::A
    Ta::A;
    Sa::A;
    drho::A
    ustar::A;
    entr::A;
    detr::A
    ent2::A;
    nentr::A;
    ent::A
    Sb::A;
    drhob::A;
    convection::A
    convD::A;
    dDdt::A;
    Ddrho::A
    conv2::Conv2
    gamT::GamT;
    gamS::GamT
    # Advection stencil temporaries
    Dym1::A;
    Dyp1::A
    Dxm1::A;
    Dxp1::A
    Dxm1ym1::A;
    Dxp1ym1::A;
    Dxm1yp1::A
    Vyp1::A;
    Uxp1::A
    Upos::A;
    Uneg::A;
    Vpos::A;
    Vneg::A
    Vyp1pos::A;
    Vyp1neg::A;
    Uxp1pos::A;
    Uxp1neg::A
    Vip::A;
    Vim::A;
    Uip::A;
    Uim::A
    Vjp::A;
    Vjm::A;
    Ujp::A;
    Ujm::A
    signU::A;
    signV::A
    # Laplacian stencil temporaries
    D0ip::A;
    D0im::A;
    D0jp::A;
    D0jm::A
    # laplace_U/laplace_V thickness grids (D0ip/D0jp * tmask, precomputed with laplacian stencils)
    D_on_ugrid::A;
    D_on_vgrid::A
    # Stencil output arrays (eliminate similar() per step)
    cU::A;
    cV::A;
    cT::A;
    cS::A
    lU::A;
    lV::A;
    lT::A;
    lS::A
    # Tracer flux intermediates (eliminate D*q allocation per step)
    DT::A;
    DS::A
end

_gamT_init(FT, _, _, ::Type{<:FixedGamT}) = zero(FT)
_gamT_init(FT, _, _, ::Type{<:PrescribedMelt}) = zero(FT)
_gamT_init(FT, ny, nx, ::Type{<:TurbulentGamT}) = zeros(FT, ny, nx)
_conv2_init(FT, _, _, ::Type{<:Union{ClampDensity,ResetToAmbient}}) = zero(FT)
_conv2_init(FT, ny, nx, ::Type{<:RelaxToAmbient}) = zeros(FT, ny, nx)

"""
$(TYPEDSIGNATURES)

Allocate all scratch matrices for a grid of size `(ny, nx)`.
`MP` (melt param type) determines whether `gamT`/`gamS` are scalars or arrays;
`CS` (convection scheme type) determines whether `conv2` is a scalar or array.
"""
function Cache(FT::Type, MP::Type, CS::Type, ny::Int, nx::Int)
    z = zeros(FT, ny, nx)
    gamT_init = _gamT_init(FT, ny, nx, MP)
    gamS_init = _gamT_init(FT, ny, nx, MP)
    conv2_init = _conv2_init(FT, ny, nx, CS)
    GamT = typeof(gamT_init)
    Conv2 = typeof(conv2_init)
    Cache{FT,Matrix{FT},GamT,Conv2}(
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
        gamS_init,   # conv2, gamT, gamS
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
        copy(z),
        copy(z),                   # D_on_ugrid, D_on_vgrid
        copy(z),
        copy(z),
        copy(z),
        copy(z), # cU, cV, cT, cS
        copy(z),
        copy(z),
        copy(z),
        copy(z), # lU, lV, lT, lS
        copy(z),
        copy(z),                   # DT, DS
    )
end
