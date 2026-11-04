# Wall momentum advection

The [wall slip](walls.md) factor always sets the **viscous wall stress**. Whether it
also forms the **momentum carried across a wall face** by the advection, through a
ghost velocity ``(1 - \text{slip})\,u``, is selected with
`BoundaryConditions(; wall_advection = ...)`:

- [`NoWallAdvection`](@ref), the default: no ghost, the advection sees a factor of `0`
  at every wall. This is what LADDIE v2 does: both of its advection schemes skip
  grounded neighbours outright, and its wall is the viscous term alone.
- [`SlipScaledWallAdvection`](@ref): the advection uses the same factor as the
  viscosity.
- [`V1WallAdvection`](@ref): the advection uses the factor minus one, LADDIE v1's
  combination (see [Wall slip](walls.md#LADDIE-v1-and-v2)).

## Interaction with the advection scheme

This interacts with the [momentum-advection scheme](momentum_advection.md), and the
interaction is easy to trip over:

- Under [`CentredMomentumAdvection`](@ref) the two terms are independent, and
  `wall_advection` is a real choice. Switching it off weakens a wall that is
  otherwise braking the flow twice over.
- Under [`UpstreamMomentumAdvection`](@ref) walls carry no momentum *by
  construction*, since they carry no mass. `wall_advection` then has **no effect
  at all** — the settings give identical results — and the wall's entire
  contribution is the viscous one.

The practical consequence is that the pair (scheme, ``A_h``) has to be chosen
together, not the wall condition on its own. A donor-cell run on a C-grid can
need considerably more ``A_h`` than v2 uses on its mesh to stay stable, because
nothing else resists the flow at the wall: on the small-gap geometry of the
[ice-shelf gaps example](../generated/jesse-gaps.md), `UpstreamMomentumAdvection`
at v2's ``A_h = 10`` lets the layer run away at the grounding line, and needs
``A_h \approx 50`` to hold (measured with the earlier no-slip default, ``s = 2``).
Treat a large ``A_h`` there as the stability crutch it is.

## API

```@docs
AbstractWallAdvection
NoWallAdvection
SlipScaledWallAdvection
V1WallAdvection
```
