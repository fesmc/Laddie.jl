# Wall momentum advection

The [wall slip](walls.md) factor is not only a viscous condition. It scales

1. the **viscous wall drag** in the Laplacian, always; and
2. the **momentum carried across a wall face** by the advection, through a ghost
   velocity ``(1 - \text{slip})\,u`` — free slip leaves the flux at its interior
   value, no slip reverses it.

The second is v1's structure, and it is what
`BoundaryConditions(; wall_advection = ...)` selects:
[`SlipScaledWallAdvection`](@ref) (the default) keeps it,
[`NoWallAdvection`](@ref) drops it and leaves the slip factor in the viscous
drag alone, which is what LADDIE v2 does — both of v2's advection schemes skip
grounded neighbours outright, and its only no-slip term is the viscous one. With free
slip the choice makes no difference, since the factor is `0` in both terms.

## Interaction with the advection scheme

This interacts with the [momentum-advection scheme](momentum_advection.md), and the
interaction is easy to trip over:

- Under [`CentredMomentumAdvection`](@ref) the two terms are independent, and
  `wall_advection` is a real choice. Switching it off weakens a wall that is
  otherwise braking the flow twice over.
- Under [`UpstreamMomentumAdvection`](@ref) walls carry no momentum *by
  construction*, since they carry no mass. `wall_advection` then has **no effect
  at all** — the two settings give identical results — and the wall's entire
  contribution reduces to the viscous drag, i.e. to ``A_h``.

The practical consequence is that the pair (scheme, ``A_h``) has to be chosen
together, not the wall condition on its own. A donor-cell run on a C-grid can
need considerably more ``A_h`` than v2 uses on its mesh to stay stable, because
nothing else resists the flow at the wall: on the small-gap geometry of the
[ice-shelf gaps example](../generated/jesse-gaps.md), `UpstreamMomentumAdvection`
at v2's ``A_h = 10`` lets the layer run away at the grounding line, and needs
``A_h \approx 50`` to hold. That is a missing wall term rather than a physical
result — Laddie.jl has no C-grid counterpart of v2's own ``A_h H u / A``
wall stress — so treat a large ``A_h`` there as the stability crutch it is.

## API

```@docs
AbstractWallAdvection
SlipScaledWallAdvection
NoWallAdvection
```
