# Wall slip

Grounding lines (mask value `2`) and land (mask value `1`: exposed bedrock, islands and
the outer border ring) are zero-flux walls. The condition on the tangential momentum
at each wall is one **slip factor** ``s`` on the faces it bounds, with the LADDIE v1
convention: ``s = 0`` is free slip, ``s = 2`` no slip, and anything in between partial
slip. The two kinds of wall are set independently, with
`BoundaryConditions(; grounding_line = ..., land = ...)`:

| | free slip, ``s = 0`` | partial slip | no slip, ``s = 2`` (default) |
|---|---|---|---|
| grounding line, [`AbstractGroundingLineBC`](@ref) | [`FreeSlipGL`](@ref) | [`PartialSlipGL`](@ref)`(s)` | [`NoSlipGL`](@ref) |
| land, [`AbstractLandBC`](@ref) | [`FreeSlipLand`](@ref) | [`PartialSlipLand`](@ref)`(s)` | [`NoSlipLand`](@ref) |

The slip factor enters the kernels exactly as in the LADDIE v1 stencils:

- in the **lateral viscosity**, the wall-face flux is ``-(1 + s)\,H u/\Delta^2``: the zero
  velocity stored beyond the wall, plus an extra wall drag ``s\,H u/\Delta^2``, which
  always scales with the global `Params.A_h` (see
  [Lateral viscosity](../physics/lateral_viscosity.md));
- in the **momentum advection**, through a ghost velocity ``(1 - s)\,u`` at the wall —
  unless [wall momentum advection](wall_advection.md) is switched off.

So ``s = 0`` removes the extra wall drag rather than making the wall exactly stress-free;
the naming follows the reference.

At a coastline corner whose stencil touches both grounded ice and exposed rock, the
grounding-line condition takes precedence, so each wall face gets exactly one slip
factor.

## Reproducing LADDIE v1 and v2

The no-slip default follows [LADDIE v2](@cite lambert_one-layer_2026), where a no-slip
grounding line improves melt patterns near the grounding line compared with
observations. LADDIE v1 applies one factor, ``s = 1``, to every wall:

```julia
# LADDIE v1
BoundaryConditions(; grounding_line = PartialSlipGL(1.0), land = PartialSlipLand(1.0))

# LADDIE v2: no slip in the viscosity, no momentum across walls
BoundaryConditions(; wall_advection = NoWallAdvection())
```

## API

```@docs
AbstractGroundingLineBC
NoSlipGL
FreeSlipGL
PartialSlipGL
AbstractLandBC
NoSlipLand
FreeSlipLand
PartialSlipLand
```
