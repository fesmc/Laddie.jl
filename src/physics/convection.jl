"""
Abstract supertype for the treatment of convective instability: cells where the
layer is denser than the ambient water beneath it, i.e. where the reduced density
contrast `δρ` falls below zero.  Pass a concrete instance as
`Params(; convection = ...)`: [`RelaxToAmbientConvection`](@ref) (the default),
[`ResetToAmbientConvection`](@ref) or [`ClampDensityConvection`](@ref).

The layer is unstable where `δρ = (ρ_a − ρ)/ρ₀ < 0`. No LADDIE paper describes a convection
closure, so the schemes follow the reference codes: `ResetToAmbientConvection` and `RelaxToAmbientConvection` the
LADDIE v1 code of [Lambert et al. (2023)](@cite lambert_modelling_2023), `ClampDensityConvection` the
buoyancy floor of the LADDIE v2 code of [Lambert et al. (2026)](@cite lambert_one-layer_2026).
`RelaxToAmbientConvection` adds the tracer sink `−(q − q_a) conv2`, `conv2 = [δρ < 0] H/convection_time`.
"""
abstract type AbstractConvection end

"""
$(TYPEDEF)

Handle convective instability (`δρ < 0`) by clamping the density contrast to a
minimum positive value, `d_rho_min / Params.rho0_seawater`, so the plume remains lighter
than ambient.  T and S are left unchanged.  This is the buoyancy floor of LADDIE v2,
`max(drho_amb, buoyancy_min/rho_sw)` with `buoyancy_min = 5e-3`.

Select via `Params(; convection = ClampDensityConvection(0.005))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct ClampDensityConvection{FT} <: AbstractConvection
    "minimum density contrast `ρ_a − ρ` (kg m⁻³, default `0.005`); divided by `Params.rho0_seawater` to compare with `δρ`"
    d_rho_min::FT = 0.005
end

"""
$(TYPEDEF)

Handle convective instability by instantly resetting T and S to their ambient values
wherever the contrast falls below `d_rho_min / Params.rho0_seawater`, with S lowered by
`d_rho_min / (rho0_seawater * beta)` so that the reset cell has exactly that contrast.

Select via `Params(; convection = ResetToAmbientConvection(0.005))`.

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct ResetToAmbientConvection{FT} <: AbstractConvection
    "threshold density contrast `ρ_a − ρ` that triggers the reset (kg m⁻³, default `0.005`); divided by `Params.rho0_seawater` to compare with `δρ`"
    d_rho_min::FT = 0.005
end

"""
$(TYPEDEF)

Handle convective instability by relaxing T and S of unstable cells toward
ambient values over a prescribed timescale (applied implicitly in the tracer
time step via the `conv2` term).  The default: a continuous relaxation rather than the
instantaneous reset of [`ResetToAmbientConvection`](@ref).  One hour is the order of the time an
unstable layer takes to overturn, and gave good pan-Antarctic runs.

Select via `Params(; convection = RelaxToAmbientConvection(3600.0))` (the default).

# Fields
$(TYPEDFIELDS)
"""
@kwdef struct RelaxToAmbientConvection{FT} <: AbstractConvection
    "relaxation timescale (s, default `3600`)"
    convection_time::FT = 3600.0
end

# ============================================================================
# `update_convection!` and its kernels
# ============================================================================

"""
$(TYPEDSIGNATURES)

Flag convectively unstable cells and clamp `δρ` to a minimum positive value
so the plume remains lighter than ambient.

Applies in gap cells too (`tmask` but not `imask`): the buoyancy floor is the one
convection treatment LADDIE v2 also has, and it applies there over its whole active
domain, gaps included.
"""
function update_convection!(m, cs::ClampDensityConvection)
    thr = cs.d_rho_min / m.rho0_seawater
    launch!(_clamp_density_kernel!, m.convective_instability, m.drho, thr, m.params.kinks)
    return nothing
end

# The convection schemes are pointwise, and run as kernels rather than broadcasts so
# that they are threaded on the CPU: they run twice per step (see
# `apply_robert_asselin_filter!`).  Each reads the `drho` of the current T/S.
@kernel function _clamp_density_kernel!(convective_instability, drho, thr, kinks)
    thr = _val(thr)
    i, j = @index(Global, NTuple)
    FT = typeof(thr)
    @inbounds begin
        d = drho[i, j]
        convective_instability[i, j] = ifelse(d < 0, one(FT), zero(FT))
        drho[i, j] = _smax(kinks, d, thr, _kink_width(kinks, :drho))
    end
end

"""
$(TYPEDSIGNATURES)

Flag convectively unstable cells, then instantly reset their T/S to ambient
values so the density remains stable.

Restricted to ice-covered cells (`imask`).  In a gap the ambient profile is sampled at
the sea surface, where it is cold and fresh, so `drho < 0` is close to unconditional
there; resetting would overwrite the T/S anomaly the layer is carrying across the gap
and rebuild the meltwater sink that [`ConnectedGapsBC`](@ref) exists to remove.  LADDIE
v2 has no reset scheme to copy here, so this is a Laddie.jl-only decision.
"""
function update_convection!(m, cs::ResetToAmbientConvection)
    thr = cs.d_rho_min / m.rho0_seawater
    eos = _eos_coefficients(m)
    S_adj = cs.d_rho_min / (m.rho0_seawater * eos.beta)
    launch!(
        _reset_to_ambient_kernel!,
        m.convective_instability,
        m.T.present,
        m.S.present,
        m.drho,
        m.Ta,
        m.Sa,
        m.tmask,
        m.imask,
        m.z_draft,
        m.H.present,
        thr,
        S_adj,
        eos,
    )
    return nothing
end

# Reset T/S of unstable ice-covered cells to ambient, and refresh `drho` there with
# the expression of `_density_kernel!` (elsewhere T/S are unchanged, so is `drho`).
@kernel function _reset_to_ambient_kernel!(
    convective_instability,
    T,
    S,
    drho,
    @Const(Ta),
    @Const(Sa),
    @Const(tmask),
    @Const(imask),
    @Const(z_draft),
    @Const(H),
    thr,
    S_adj,
    eos,
)
    thr, S_adj, eos = _val(thr), _val(S_adj), _val(eos)
    i, j = @index(Global, NTuple)
    FT = typeof(thr)
    @inbounds begin
        d = drho[i, j]
        ice = imask[i, j] > 0
        convective_instability[i, j] = ifelse((d < 0) & ice, one(FT), zero(FT))
        reset = (d < thr) & ice
        T_new = ifelse(reset, Ta[i, j], T[i, j])
        S_new = ifelse(reset, Sa[i, j] - S_adj, S[i, j])
        drho_reset =
            _drho(Sa[i, j], S_new, Ta[i, j], T_new, _layer_depth(z_draft[i, j], H[i, j]), eos) * tmask[i, j]
        T[i, j] = T_new
        S[i, j] = S_new
        drho[i, j] = ifelse(reset, drho_reset, d)
    end
end

"""
$(TYPEDSIGNATURES)

Flag convectively unstable cells; relaxation is applied implicitly during the
tracer time step via `conv2`.

Restricted to ice-covered cells (`imask`) for the same reason as
[`ResetToAmbientConvection`](@ref) — relaxing a gap cell towards surface ambient is a slower
version of the same sink.
"""
function update_convection!(m, ::RelaxToAmbientConvection)
    launch!(_flag_unstable_ice_kernel!, m.convective_instability, m.drho, m.imask)
end

@kernel function _flag_unstable_ice_kernel!(convective_instability, @Const(drho), @Const(imask))
    i, j = @index(Global, NTuple)
    FT = eltype(convective_instability)
    @inbounds convective_instability[i, j] =
        ifelse((drho[i, j] < 0) & (imask[i, j] > 0), one(FT), zero(FT))
end

update_convection!(m) = update_convection!(m, m.convection)
