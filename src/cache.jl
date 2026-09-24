"""
$(TYPEDEF)

Pre-allocated diagnostic and scratch fields of a [`Model`](@ref), updated in place
every time step.  The physics outputs are read through the model, e.g.
`model.melt` (m s⁻¹), `model.entr`, `model.drho`, `model.ustar`, `model.Ta`.

`A` is the concrete matrix type (matching the grid and state).  The scheme-dependent
slots are scalars or fields: `GamT` (`gamT`, `gamS`) is a field for
`TurbulentGamTMelting`, `UStarGamTMelting` and `PrescribedMelting`, `Conv2` (`conv2`) for
`RelaxToAmbient`, and `PM` (`melt_prescribed`) for `PrescribedMelting`.

`adv`, `lap` and `Dq` are shared by the four stepped equations: U, V, T and S are
stepped one after another, and each consumes its terms before the next one writes
them.  In between, `adv` and `lap` hold the speed limiter's per-point scale factors,
and `Dq` the collocated cross-velocity of `NonlinearLateralViscosity`.  `diag` is never read by the time step, so any host-side diagnostic may
overwrite it.

# Fields
$(TYPEDFIELDS)
"""
mutable struct Cache{A<:AbstractMatrix,GamT,Conv2,PM}
    "basal melt rate (m s⁻¹)"
    melt::A
    "temperature at the ice–ocean interface (°C)"
    Tb::A
    "ambient temperature at the layer base (°C)"
    Ta::A
    "ambient salinity at the layer base (psu)"
    Sa::A
    "reduced density contrast with the ambient water (dimensionless)"
    drho::A
    "friction velocity at the ice base (m s⁻¹)"
    ustar::A
    "entrainment rate, the positive part of the signed rate ė (m s⁻¹)"
    entr::A
    "detrainment rate, the capped negative part of ė (m s⁻¹)"
    detr::A
    "extra entrainment that keeps `D ≥ D_min` after the thickness step (m s⁻¹)"
    ent2::A
    "net entrainment `entr + ent2 - detr` of the thickness and tracer equations (m s⁻¹)"
    nentr::A
    "convective-instability indicator: `1` where `drho < 0` was corrected"
    convection::A
    "upwind advection term of the thickness equation (m s⁻¹)"
    convD::A
    "thickness tendency of the current step (m s⁻¹)"
    dDdt::A
    "product `D * drho` read by the momentum equations (m)"
    Ddrho::A
    "relaxation rate toward ambient under `RelaxToAmbient` (a field), else an unused scalar"
    conv2::Conv2
    "turbulent heat transfer velocity (m s⁻¹): a field, or a scalar under `FixedGamTMelting`"
    gamT::GamT
    "turbulent salt transfer velocity (m s⁻¹): a field, or a scalar under `FixedGamTMelting`"
    gamS::GamT
    "prescribed melt rate under `PrescribedMelting` (m s⁻¹), else an unused scalar"
    melt_prescribed::PM
    "advection term of the equation being stepped (work buffer)"
    adv::A
    "Laplacian term of the equation being stepped (work buffer)"
    lap::A
    "advected tracer content `D * q` of the equation being stepped (work buffer)"
    Dq::A
    "scratch for host-side diagnostics (CFL, melt statistics)"
    diag::A
end

_gamT_init(FT, _, _, ::Type{<:FixedGamTMelting}) = zero(FT)
# PrescribedMelting reports a per-cell equivalent gamT (see `update_melt!`), so it
# takes the array path like TurbulentGamTMelting.
_gamT_init(FT, nx, ny, ::Type{<:PrescribedMelting}) = zeros(FT, nx, ny)
_gamT_init(FT, nx, ny, ::Type{<:TurbulentGamTMelting}) = zeros(FT, nx, ny)
_gamT_init(FT, nx, ny, ::Type{<:UStarGamTMelting}) = zeros(FT, nx, ny)
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
        Matrix{FT},
        typeof(special.gamT),
        typeof(special.conv2),
        typeof(special.melt_prescribed),
    }
    fields = map(fn -> haskey(special, fn) ? special[fn] : zeros(FT, nx, ny), fieldnames(C))
    return C(fields...)
end
