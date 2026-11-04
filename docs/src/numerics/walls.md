# Wall slip

Grounding lines (mask value `2`) and land (mask value `1`: exposed bedrock, islands and
the outer border ring) are zero-flux walls. The condition on the tangential momentum
at each wall is one **slip factor** ``s`` on the faces it bounds, with the convention of
NEMO (`rn_shlat`): the velocity behind the wall is taken as a ghost ``(1 - s)\,u``. The
two kinds of wall are set independently, with
`BoundaryConditions(; grounding_line = ..., land = ...)`:

| | free slip, ``s = 0`` | partial slip | no slip, ``s = 2`` |
|---|---|---|---|
| grounding line, [`AbstractGroundingLineBC`](@ref) | [`FreeSlipGL`](@ref) | [`PartialSlipGL`](@ref)`(s)`, default `s = 1` | [`NoSlipGL`](@ref) |
| land, [`AbstractLandBC`](@ref) | [`FreeSlipLand`](@ref) | [`PartialSlipLand`](@ref)`(s)`, default `s = 1` | [`NoSlipLand`](@ref) |

## What ``s`` means

Take a ``u``-point next to a wall. On the C-grid the wall is the face between the two
rows of T-cells, half a grid spacing ``\Delta`` away; the next ``u``-point, a full
``\Delta`` away, is behind the wall. The viscous flux through the wall face is

```math
A\,H\,\frac{(1 - s)\,u - u}{\Delta^2} = -s\,A\,H\,\frac{u}{\Delta^2},
```

the same as a velocity gradient ``u/\delta`` with ``\delta = \Delta/s``. So ``s`` sets where
the velocity is zero:

| ``s`` | ``u = 0`` at | |
|---|---|---|
| 0 | nowhere | free slip: the wall exerts no stress |
| 1 | one ``\Delta`` behind the wall, at the next ``u``-point | partial slip; LADDIE v2's wall (the default) |
| 2 | the wall face, ``\Delta/2`` away | no slip |

The kernels form it from the velocity stored behind the wall, which is zero, so the
plain stencil already holds ``s = 1``; they add the remaining ``s - 1``. Under
[`NonlinearLateralViscosity`](@ref) the ghost enters the shear-scaled coefficient as
well, so its wall stress grows as ``s^2``.

The same ghost forms the momentum carried across a wall face by the advection, unless
[wall momentum advection](wall_advection.md) says otherwise.

At a coastline corner whose stencil touches both grounded ice and exposed rock, the
grounding-line condition takes precedence, so each wall face gets exactly one slip
factor.

## LADDIE v1 and v2

LADDIE v2 treats grounded and ice-free land neighbours as ordinary cells with zero
velocity, one grid spacing away (`laddie_velocity.f90:259-267`), and carries no momentum
across them. This is the default, ``s = 1`` with [`NoWallAdvection`](@ref). v2 calls it
"no slip" ([Lambert et al., 2026](@cite lambert_one-layer_2026)); in this convention it
is partial slip.

LADDIE v1 describes its walls as NEMO-like partial slip with a factor between 0 (free
slip) and 2 (no slip), and its published runs use 1 (Lambert et al., 2023). Its code,
however, adds the factor to a viscous stencil that already holds the stored zero, so
its viscous wall is one step stronger than labelled: ``s = 1`` is no slip there. In the
advection the factor is used as labelled. Reproducing v1 therefore takes
[`V1WallAdvection`](@ref):

```julia
# LADDIE v1 (slip = 1): no slip in the viscosity, partial slip in the advection
BoundaryConditions(; grounding_line = NoSlipGL(), land = NoSlipLand(),
                   wall_advection = V1WallAdvection())

# LADDIE v2: the defaults
BoundaryConditions()
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
