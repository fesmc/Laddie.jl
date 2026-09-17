#############################
# Wall slip (grounding line and land)
#############################

# A wall's tangential-momentum condition is one slip factor `s` on the faces it
# bounds, with the Python LADDIE convention: 0 = free slip, 2 = no slip, anything
# in between partial slip.  The kernels follow the Python stencils exactly:
#   - lateral viscosity: the wall-face flux is −(1 + s)·D·u/Δ² — the zero velocity
#     stored beyond the wall, plus an extra wall drag s·D·u/Δ²;
#   - momentum advection: the face velocity at the wall is (1 − s)·u.
# So s = 0 removes the extra wall drag rather than making the wall exactly
# stress-free; the naming follows the reference.

"Check a partial-slip factor lies in the admissible range [0, 2]."
function _check_slip(factor::Real)
    0 <= factor <= 2 || throw(
        ArgumentError(
            "slip factor must lie in [0, 2] (0 = free slip, 2 = no slip), got $factor",
        ),
    )
    return float(factor)
end

"""
Abstract supertype for the momentum boundary condition at grounding-line walls
(mask value `2`).  Pass a concrete instance as
`BoundaryConditions(; grounding_line = ...)`: [`NoSlipGL`](@ref) (the default),
[`FreeSlipGL`](@ref) or [`PartialSlipGL`](@ref).

Walls bordering exposed rock are governed separately by [`AbstractLandBC`](@ref).
"""
abstract type AbstractGroundingLineBC end

"""
$(TYPEDEF)

No-slip momentum boundary condition at the grounding line (the default): the
tangential velocity vanishes at walls bordering grounded ice (mask value `2`).
Implemented as a slip factor of `2` on grounding-line faces.  Land walls (mask
value `1`) are governed independently by `BoundaryConditions.land`.

Motivated by LADDIE v2.0 (Lambert et al., in review, 2026), where a no-slip
grounding-line condition improves melt patterns near the grounding line
compared to observations.

Select via `BoundaryConditions(; grounding_line = NoSlipGL())` (the default).
"""
struct NoSlipGL <: AbstractGroundingLineBC end

"""
$(TYPEDEF)

Free-slip momentum boundary condition at the grounding line: slip factor `0` at
walls bordering grounded ice, the free-slip end of the Python LADDIE convention (no
wall drag beyond the zero velocity stored in the wall cell).

Select via `BoundaryConditions(; grounding_line = FreeSlipGL())`.
"""
struct FreeSlipGL <: AbstractGroundingLineBC end

"""
$(TYPEDEF)

Partial-slip momentum boundary condition at the grounding line, with a slip
`factor` between `0` (free slip) and `2` (no slip).

Python LADDIE v1.x applies one factor, `slip = 1`, to every wall; reproduce it with
`BoundaryConditions(; grounding_line = PartialSlipGL(1.0), land = PartialSlipLand(1.0))`.

Select via `BoundaryConditions(; grounding_line = PartialSlipGL(1.0))`.
"""
struct PartialSlipGL{FT} <: AbstractGroundingLineBC
    factor::FT
    PartialSlipGL(factor::Real) = (f = _check_slip(factor); new{typeof(f)}(f))
end

# Slip factor applied at grounding-line faces (0 = free slip, 2 = no slip).
_gl_slip(::FreeSlipGL, FT) = zero(FT)
_gl_slip(::NoSlipGL, FT) = FT(2)
_gl_slip(bc::PartialSlipGL, FT) = FT(bc.factor)

"""
Abstract supertype for the momentum boundary condition at land walls (mask value
`1`: exposed bedrock, islands, and the outer border ring).  Pass a concrete
instance as `BoundaryConditions(; land = ...)`: [`NoSlipLand`](@ref) (the default),
[`FreeSlipLand`](@ref) or [`PartialSlipLand`](@ref).

The grounding line is governed separately by [`AbstractGroundingLineBC`](@ref); at a
coastline corner whose stencil touches both grounded ice and exposed rock, the
grounding-line condition takes precedence (the land wall indicators exclude faces
already flagged as grounding line), so each wall face gets exactly one slip factor.
"""
abstract type AbstractLandBC end

"""
$(TYPEDEF)

No-slip momentum boundary condition at land walls (the default): the tangential
velocity vanishes at walls bordering exposed bedrock (mask value `1`), which
includes islands and ice-free coastline inside the domain as well as the outer
border ring.  Implemented as a slip factor of `2` on land faces.

Select via `BoundaryConditions(; land = NoSlipLand())` (the default).
"""
struct NoSlipLand <: AbstractLandBC end

"""
$(TYPEDEF)

Free-slip momentum boundary condition at land walls: slip factor `0` at walls
bordering exposed bedrock (see [`FreeSlipGL`](@ref)).

Select via `BoundaryConditions(; land = FreeSlipLand())`.
"""
struct FreeSlipLand <: AbstractLandBC end

"""
$(TYPEDEF)

Partial-slip momentum boundary condition at land walls, with a slip `factor`
between `0` (free slip) and `2` (no slip).  See [`PartialSlipGL`](@ref) for
reproducing Python LADDIE v1.x.

Select via `BoundaryConditions(; land = PartialSlipLand(1.0))`.
"""
struct PartialSlipLand{FT} <: AbstractLandBC
    factor::FT
    PartialSlipLand(factor::Real) = (f = _check_slip(factor); new{typeof(f)}(f))
end

# Slip factor applied at land faces.  Mirrors `_gl_slip`; kept as a separate
# dispatch point so the two wall treatments can diverge later.
_land_slip(::FreeSlipLand, FT) = zero(FT)
_land_slip(::NoSlipLand, FT) = FT(2)
_land_slip(bc::PartialSlipLand, FT) = FT(bc.factor)

# Per-face slip factors of a model, as passed to the momentum kernels.
_wall_slips(m) = (
    _gl_slip(m.boundary.grounding_line, m.FT),
    _land_slip(m.boundary.land, m.FT),
)

#############################
# Open BC
#############################

"""
Abstract supertype for the open-ocean boundary condition at the ice front.  Pass a
concrete instance as `BoundaryConditions(; open_ocean = ...)`:
[`ZeroGradientInflow`](@ref) (the default) or [`NoInflow`](@ref).
"""
abstract type AbstractOpenOceanBC end

"""
$(TYPEDEF)

Open-boundary condition at the ice front: zero-gradient extrapolation of all
fields, with inflow from the ambient ocean permitted.

Select via `BoundaryConditions(; open_ocean = ZeroGradientInflow())` (the default).
"""
struct ZeroGradientInflow <: AbstractOpenOceanBC end

"""
$(TYPEDEF)

Open-boundary condition at the ice front: no inflow of layer properties.  Where
the flow enters the domain through an ice-front face, the thickness and tracer
fluxes carry the (zero) value stored in the open-ocean cell instead of the
zero-gradient extrapolation of [`ZeroGradientInflow`](@ref), so no volume, heat or
salt is advected in.  The velocities themselves are not clipped, and momentum
advection is unaffected.

Select via `BoundaryConditions(; open_ocean = NoInflow())`.
"""
struct NoInflow <: AbstractOpenOceanBC end

#############################
# Ice-front pressure gradient
#############################

"""
Abstract supertype for the treatment of the layer-thickness-gradient part of the
pressure-gradient force at one-sided faces — the ice front, and the edges of a gap
demoted to ocean by [`SinkGapsBC`](@ref).  Pass a concrete instance as
`Params(; front_pressure = ...)`: [`FullDepthGradient`](@ref) (Python LADDIE v1.x,
the default) or [`TruncatedDepthGradient`](@ref) (LADDIE v2).

The grounding line is unaffected either way — no momentum equation is solved there.
"""
abstract type AbstractFrontPressure end

"""
$(TYPEDEF)

Keep the layer-thickness-gradient part of the pressure-gradient force at
one-sided faces — the ice front, and the edges of a gap demoted to ocean by
[`SinkGapsBC`](@ref) — evaluating it as `(D_neighbour - D)/Δ` with the
neighbour's stored thickness, which masking pins to `0` there. The term is thus
a full one-sided gradient, as if `D` fell to zero across one grid cell.

This is what Python LADDIE v1.x does (`integrate.py`, `-g·ip_t(Ddrho)·(Dxm1 -
D)/dx` with `Dxm1 = roll(D*tmask)`), so it is **the default** and the setting
under which Laddie.jl reproduces the Python reference.

Select via `Params(; front_pressure = FullDepthGradient())` (the default).

See also [`TruncatedDepthGradient`](@ref).
"""
struct FullDepthGradient <: AbstractFrontPressure end

"""
$(TYPEDEF)

Drop the layer-thickness-gradient part of the pressure-gradient force at
one-sided faces — the ice front, and the edges of a gap demoted to ocean by
[`SinkGapsBC`](@ref) — keeping only the ice-base-slope and density-gradient
parts there.

This is what the LADDIE v2 Fortran reference does at calving-front faces
(`laddie_velocity.f90:118-126`, `IF (mask_cf_b .OR. mask_gl_b) ... assume
dH/dx = 0`). It avoids treating a masked neighbour's stored `D = 0` as a real
thickness: under [`FullDepthGradient`](@ref) that term is roughly 150× larger
at the ice front than in the interior on a warm ISOMIP+ run.

Neither choice affects the grounding line, where the velocity masks are zero
and no momentum equation is solved at all.

Select via `Params(; front_pressure = TruncatedDepthGradient())`.
"""
struct TruncatedDepthGradient <: AbstractFrontPressure end

# Weight `w` in the per-face gate `1 + w*(tmask_stag - 2)`, which is 1 on a
# fully-interior face (tmask_stag == 2) either way, and at a one-sided face
# (tmask_stag == 1) is 1 for FullDepthGradient / 0 for TruncatedDepthGradient.
# w = 0 multiplies the term by exactly 1.0, so the default is bit-identical to
# Python v1.x.
_front_pgf_weight(::FullDepthGradient, x) = zero(x)
_front_pgf_weight(::TruncatedDepthGradient, x) = one(x)

#############################
# Gaps BC
#############################

"""
Abstract supertype for the treatment of ice-shelf gaps — ice-free cells inside
the shelf domain, marked `4` in the mask.  Pass a concrete instance as
`BoundaryConditions(; gaps = ...)`: [`SinkGapsBC`](@ref) (the default, gaps act as sinks)
or [`ConnectedGapsBC`](@ref) (gaps stay dynamically connected).
"""
abstract type AbstractGapsBC end

"""
$(TYPEDEF)

Treat ice-shelf gaps — ice-free cells inside the ice-shelf domain, marked `4` in
the domain mask — as sinks of the meltwater layer (the default, and the behaviour
of LADDIE v1.x).  Gap cells are demoted to open ocean (`0`) at model construction,
so a meltwater current reaching a gap loses its heat and momentum to the ambient
ocean exactly as it would at the calving front, and cannot drive melting further
downstream.

Select via `BoundaryConditions(; gaps = SinkGapsBC())` (the default).

See also [`ConnectedGapsBC`](@ref).
"""
struct SinkGapsBC <: AbstractGapsBC end

"""
$(TYPEDEF)

Treat ice-shelf gaps as dynamically connected: the meltwater layer keeps being
integrated across ice-free cells inside the ice-shelf domain, so heat and momentum
are advected downstream and can drive melting on the far side of the gap.  No
melting occurs within a gap itself — there is no glacial ice to melt — so no
meltwater volume or buoyancy is added there.

Gap cells are those marked `4` in the domain mask.  Mark them either directly (a
coupled ice-sheet driver knows its own reference geometry) or from a reference ice
footprint with [`MarkGapsPreprocess`](@ref) in the `preprocess` list of [`Grid`](@ref).  This
boundary condition only decides what happens in the cells so marked.

Motivated by Jesse et al. (2026, https://doi.org/10.5194/egusphere-2026-4237),
who show that the choice between this treatment and [`SinkGapsBC`](@ref) can alter
coupled ice-volume loss by an amount comparable to 1 °C of ocean warming.  The two
are deliberate end-members; reality lies somewhere between them.

# Example

```julia
boundary = BoundaryConditions(; gaps = ConnectedGapsBC())
Model(Grid(mask, z_draft, dx, dy); forcing, boundary)   # gaps already marked as 4
grid = Grid(mask, z_draft, dx, dy; preprocess = [MarkGapsPreprocess(refgeo_H)])
Model(grid; forcing, boundary)                          # gaps from a footprint
```
"""
struct ConnectedGapsBC <: AbstractGapsBC end

# Resolve mask value 4 (gap) into the classification the selected treatment implies.
# Runs at model construction, before the Geometry is derived: SinkGapsBC demotes gaps
# to open ocean, so every mask derivation sees exactly the v1.x geometry;
# ConnectedGapsBC keeps them.  Marking gaps in the first place is
# `MarkGapsPreprocess`'s job.
_apply_gaps_bc(mask, ::SinkGapsBC) = ifelse.(mask .== 4, 0, mask)
_apply_gaps_bc(mask, ::ConnectedGapsBC) = mask
#############################
# Boundary-condition container
#############################

"""
$(TYPEDEF)

The boundary conditions of a [`Model`](@ref): what happens at the ice front, at
grounding-line and land walls, and in ice-shelf gaps.  Pass it as
`Model(...; boundary = BoundaryConditions(...))`.  Every condition has a default;
all but the wall slip reproduce LADDIE v1.x, whose single partial-slip factor is
available as [`PartialSlipGL`](@ref) and [`PartialSlipLand`](@ref).

```julia
BoundaryConditions(; land = FreeSlipLand(), gaps = ConnectedGapsBC())

# Python LADDIE v1.x walls
BoundaryConditions(; grounding_line = PartialSlipGL(1.0), land = PartialSlipLand(1.0))
```

# Fields
$(TYPEDFIELDS)
"""
struct BoundaryConditions{
    OB<:AbstractOpenOceanBC,
    GL<:AbstractGroundingLineBC,
    LB<:AbstractLandBC,
    GB<:AbstractGapsBC,
}
    "ice front: [`ZeroGradientInflow`](@ref) (default) or [`NoInflow`](@ref)"
    open_ocean::OB
    "grounding-line walls: [`NoSlipGL`](@ref) (default), [`FreeSlipGL`](@ref) or [`PartialSlipGL`](@ref)"
    grounding_line::GL
    "land walls: [`NoSlipLand`](@ref) (default), [`FreeSlipLand`](@ref) or [`PartialSlipLand`](@ref)"
    land::LB
    "ice-shelf gaps: [`SinkGapsBC`](@ref) (default) or [`ConnectedGapsBC`](@ref)"
    gaps::GB
end

BoundaryConditions(;
    open_ocean = ZeroGradientInflow(),
    grounding_line = NoSlipGL(),
    land = NoSlipLand(),
    gaps = SinkGapsBC(),
) = BoundaryConditions(open_ocean, grounding_line, land, gaps)

_bc_tuple(b::BoundaryConditions) = (b.open_ocean, b.grounding_line, b.land, b.gaps)
