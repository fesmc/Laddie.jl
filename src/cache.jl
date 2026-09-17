"""
$(TYPEDEF)

Pre-allocated diagnostic and scratch fields of a [`Model`](@ref), updated in place
every time step.  The physics outputs are read through the model, e.g.
`model.melt` (m s⁻¹), `model.entr`, `model.drho`, `model.ustar`, `model.Ta`.

`A` is the concrete matrix type (matching the grid and state).  The scheme-dependent
slots are scalars or fields: `GamT` (`gamT`, `gamS`) is a field for
`TurbulentGamTMelting` and `PrescribedMelting`, `Conv2` (`conv2`) for
`RelaxToAmbient`, and `PM` (`melt_prescribed`) for `PrescribedMelting`.
"""
mutable struct Cache{FT,A<:AbstractMatrix{FT},GamT,Conv2,PM}
    # Physics outputs
    melt::A
    Tb::A
    Tf::A
    Ta::A
    Sa::A
    drho::A
    ustar::A
    entr::A
    detr::A
    ent2::A
    nentr::A
    ent::A
    Sb::A
    drhob::A
    convection::A
    convD::A
    dDdt::A
    Ddrho::A
    conv2::Conv2
    gamT::GamT
    gamS::GamT
    melt_prescribed::PM
    # Staggered velocity averages (precompute_advection_stencils!)
    Vip::A
    Vim::A
    Uip::A
    Uim::A
    Vjp::A
    Vjm::A
    Ujp::A
    Ujm::A
    # Laplacian thickness averages (precompute_laplacian_stencils!)
    D0ip::A
    D0im::A
    D0jp::A
    D0jm::A
    D_on_ugrid::A
    D_on_vgrid::A
    # Work buffers of the stepped equations: the advection and Laplacian terms, and
    # the tracer content D·q that is advected.  U, V, T and S are stepped one after
    # another, and each consumes its terms before the next one writes them, so a
    # single set serves all four.
    adv::A
    lap::A
    Dq::A
    # Per-point scale factors for the speed-preserving velocity limiter
    scaleU::A
    scaleV::A
    # Cross-component velocity collocated onto the other grid's points, so
    # NonlinearLateralViscosity can form |Δu| = √(ΔU² + ΔV²) across a face the
    # way the reference does.  Unused by PrescribedLateralViscosity.
    VatU::A
    UatV::A
    # Scratch for host-side diagnostics (CFL, melt statistics); never read by the
    # time step, so any diagnostic may overwrite it.
    diag::A
end

_gamT_init(FT, _, _, ::Type{<:FixedGamTMelting}) = zero(FT)
# PrescribedMelting reports a per-cell equivalent gamT (see `update_melt!`), so it
# takes the array path like TurbulentGamTMelting.
_gamT_init(FT, nx, ny, ::Type{<:PrescribedMelting}) = zeros(FT, nx, ny)
_gamT_init(FT, nx, ny, ::Type{<:TurbulentGamTMelting}) = zeros(FT, nx, ny)
_prescribed_init(FT, _, _, ::Type) = zero(FT)
_prescribed_init(FT, nx, ny, ::Type{<:PrescribedMelting}) = zeros(FT, nx, ny)
_conv2_init(FT, _, _, ::Type{<:Union{ClampDensity,ResetToAmbient}}) = zero(FT)
_conv2_init(FT, nx, ny, ::Type{<:RelaxToAmbient}) = zeros(FT, nx, ny)

"""
$(TYPEDSIGNATURES)

Allocate all scratch matrices for a grid of size `(nx, ny)`.
`MP` (melt param type) determines whether `gamT`/`gamS` are scalars or arrays and
whether a prescribed melt field is carried; `CS` (convection scheme type)
determines whether `conv2` is a scalar or array.
"""
function Cache(FT::Type, MP::Type, CS::Type, nx::Int, ny::Int)
    special = (
        conv2 = _conv2_init(FT, nx, ny, CS),
        gamT = _gamT_init(FT, nx, ny, MP),
        gamS = _gamT_init(FT, nx, ny, MP),
        melt_prescribed = _prescribed_init(FT, nx, ny, MP),
    )
    C = Cache{
        FT,
        Matrix{FT},
        typeof(special.gamT),
        typeof(special.conv2),
        typeof(special.melt_prescribed),
    }
    fields = map(fn -> haskey(special, fn) ? special[fn] : zeros(FT, nx, ny), fieldnames(C))
    return C(fields...)
end
