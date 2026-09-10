#############################
# Grounding Line BC
#############################

abstract type AbstractGroundingLineBC end

"""
$(TYPEDSIGNATURES)

Grounding-line momentum boundary condition of LADDIE v1.x (the default):
grounding-line walls use the same slip factor as land walls, i.e.
`Params.slip` (`1.0` = free slip).

Select via `Params(; grline_bc = FreeSlipGL())` (the default).
"""
struct FreeSlipGL <: AbstractGroundingLineBC end

"""
$(TYPEDSIGNATURES)

No-slip momentum boundary condition at the grounding line: the tangential
velocity is forced to zero at walls bordering grounded ice (mask value `2`),
while land/border walls (mask value `1`) keep the global `Params.slip`
factor.  Implemented as a slip factor of `2` on grounding-line faces
(ghost velocity = -interior velocity).

Motivated by LADDIE v2.0 (Lambert et al., in review, 2026), where a no-slip
grounding-line condition improves melt patterns near the grounding line
compared to observations.  Not yet validated against LADDIE v2.0 output.

Select via `Params(; grline_bc = NoSlipGL())`.
"""
struct NoSlipGL <: AbstractGroundingLineBC end

# Slip factor applied at grounding-line faces (land faces always use `slip`).
# Ghost tangential velocity is (1 − factor)·u: 1 → free slip, 2 → no slip.
_gl_slip(::FreeSlipGL, slip) = slip
_gl_slip(::NoSlipGL, slip) = oftype(slip, 2)

#############################
# Open BC
#############################

abstract type AbstractOpenOceanBC end

"""
$(TYPEDSIGNATURES)

Open-boundary condition at the ice front: zero-gradient extrapolation of all
fields, with inflow from the ambient ocean permitted.

Select via `Params(; open_bc = ZeroGradientInflow())` (the default).
"""
struct ZeroGradientInflow <: AbstractOpenOceanBC end

"""
$(TYPEDSIGNATURES)

Open-boundary condition at the ice front: outflow only — inflow velocities are
clipped to zero so ambient water cannot advect into the domain.

Select via `Params(; open_bc = NoInflow())`.
"""
struct NoInflow <: AbstractOpenOceanBC end

#############################
# Gaps BC
#############################

abstract type AbstractGapsBC end

"""
$(TYPEDSIGNATURES)

Treat ice-shelf gaps — ice-free cells inside the ice-shelf domain, marked `4` in
the domain mask — as sinks of the meltwater layer (the default, and the behaviour
of LADDIE v1.x).  Gap cells are demoted to open ocean (`0`) at model construction,
so a meltwater current reaching a gap loses its heat and momentum to the ambient
ocean exactly as it would at the calving front, and cannot drive melting further
downstream.

Select via `Params(; gaps_bc = SinkGapsBC())` (the default).

See also [`ConnectedGapsBC`](@ref).
"""
struct SinkGapsBC <: AbstractGapsBC end

"""
$(TYPEDSIGNATURES)

Treat ice-shelf gaps as dynamically connected: the meltwater layer keeps being
integrated across ice-free cells inside the ice-shelf domain, so heat and momentum
are advected downstream and can drive melting on the far side of the gap.  No
melting occurs within a gap itself — there is no glacial ice to melt — so no
meltwater volume or buoyancy is added there.

Gap cells are those marked `4` in the domain mask.  Mark them either directly (a
coupled ice-sheet driver knows its own reference geometry, and this mirrors the
`refgeo_Hi > 0` test of the reference implementation) or by passing a reference
ice footprint as `refgeo`: every ocean cell (`0`) where `refgeo` is true is
promoted to a gap at model construction.

`refgeo` may be a `Bool` matrix, or any numeric matrix in which positive entries
mark ice (e.g. a reference ice thickness).  It must match the size of the mask
*as passed to* `Model`, i.e. before domain cropping.

Motivated by Jesse et al. (2026, https://doi.org/10.5194/egusphere-2026-4237),
who show that the choice between this treatment and [`SinkGapsBC`](@ref) can alter
coupled ice-volume loss by an amount comparable to 1 °C of ocean warming.  The two
are deliberate end-members; reality lies somewhere between them.

# Example

```julia
Params(; gaps_bc = ConnectedGapsBC())              # gaps already marked as 4
Params(; gaps_bc = ConnectedGapsBC(refgeo_H))      # derive gaps from a footprint
```

# Fields
 - `refgeo`: reference ice footprint, or `nothing` (default) to use the mask as given.
"""
struct ConnectedGapsBC{R} <: AbstractGapsBC
    refgeo::R
end

ConnectedGapsBC() = ConnectedGapsBC(nothing)

# Resolve mask value 4 (gap) into the classification the selected treatment implies.
# Runs at model construction, before Grid: SinkGapsBC demotes gaps to open ocean, so
# every downstream mask derivation sees exactly the v1.x geometry; ConnectedGapsBC
# keeps them, and additionally promotes ocean cells inside `refgeo` to gaps.
_apply_gaps_bc(mask, ::SinkGapsBC) = ifelse.(mask .== 4, 0, mask)

_apply_gaps_bc(mask, bc::ConnectedGapsBC{Nothing}) = mask

function _apply_gaps_bc(mask, bc::ConnectedGapsBC)
    size(bc.refgeo) == size(mask) || throw(
        ArgumentError(
            "ConnectedGapsBC refgeo must have the same size as the mask, got " *
            "$(size(bc.refgeo)) vs $(size(mask)); note the mask is the one passed to " *
            "`Model`, before any domain cropping",
        ),
    )
    had_ice = bc.refgeo isa AbstractMatrix{Bool} ? bc.refgeo : bc.refgeo .> 0
    return ifelse.((mask .== 0) .& had_ice, 4, mask)
end