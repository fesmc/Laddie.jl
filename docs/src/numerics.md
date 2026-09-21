# Numerics

## Spatial discretisation

LADDIE uses an Arakawa **C-grid**: scalar fields (``D``, ``T``, ``S``) and all
diagnostic quantities live at cell centres, while ``U`` sits on the faces between
cells ``i`` and ``i+1`` and ``V`` on the faces between ``j`` and ``j+1``. Arrays are
stored `[x, y]`, and x and y are grid axes, not compass directions. Staggered
interpolation operators move quantities between the two locations as needed by
the pressure-gradient, Coriolis and drag terms. Boundary conditions are encoded in
floating-point mask arrays (`tmask`, `umask`, `vmask`, and the wall indicators)
rather than in branching logic, so the same kernels execute unchanged on CPU and
GPU.

## Spatial operators

Advective fluxes of volume, heat and salt are computed with a flux-form
**upwind** scheme (`upwind_advection_T`); viscous and diffusive fluxes use a
five-point **Laplacian** weighted by the layer thickness (`laplace_T`,
`laplace_U`, `laplace_V`). All operators are nearest-neighbour stencils written
as fused KernelAbstractions kernels, one pass per term, with the masks applied
inside.

**Momentum** advection (`upwind_advection_U`, `upwind_advection_V`) is a choice,
`Params(; momentum_advection)`:

- [`CentredMomentumAdvection`](@ref), the default, evaluates the advected
  velocity as the average of the two either side of the face, as Python LADDIE
  v1.x does. It adds no numerical viscosity of its own.
- [`UpstreamMomentumAdvection`](@ref) is flux-form donor cell, as LADDIE v2:
  the momentum rides on the thickness equation's own mass fluxes, averaged onto
  the staggered faces, and is taken from the upstream side. Its implicit
  viscosity is of order ``|u|\,\Delta/2`` — ≈ 50 m² s⁻¹ at 1 km and
  0.1 m s⁻¹, far above a typical ``A_h`` — so the layer is less energetic.

Only the momentum advection is affected; tracer advection is upwind either way.

## Time integration

The prognostic variables advance with a **leapfrog** scheme. Each variable
carries three time levels — past, present, and future — updated each step as

```math
q^{n+1} = q^{n-1} + 2\,\Delta t\; \mathcal{F}(q^n, q^{n-1}),
```

where advection, pressure, Coriolis, drag, entrainment and melt are evaluated at
the present level, while the Laplacian terms act on the **lagged** (past) level —
a diffusive term evaluated at the centre of a leapfrog step is unconditionally
unstable. After the leapfrog advance, a **Robert–Asselin filter** (strength
``\nu``, `Simulation(model; nu)`) damps the computational (``2\Delta t``) mode:

```math
q^n \;\leftarrow\; q^n + \frac{\nu}{2}\bigl(q^{n-1} + q^{n+1} - 2\,q^n\bigr).
```

The three levels are then cycled (past ← filtered present, present ← future)
before the next step. The very first step, and the first step after an adaptive
time-step change, is a forward step over ``\Delta t`` from identical levels.

The filter and the explicit Laplacian terms (``A_h``, ``K_h``) are what keep the
leapfrog computational mode in check alongside the upwind advection; weakening
either makes long runs prone to divergence.

## Domain, boundaries, and masks

The domain is padded with a **one-cell border** of land on all sides. This lets
all stencil operators use **periodic wrap** throughout — there are no special edge
cases — while the border cells are masked out and contribute nothing to the
dynamics. The physical boundaries are:

- **Grounding line and land**: zero-flux walls, each with its own slip factor
  (`BoundaryConditions(; grounding_line, land)`): 0 = free slip, 2 = no slip (the
  default), anything in between partial slip.
- **Ice front**: open boundary, with either zero-gradient inflow
  (`ZeroGradientInflow`) or no inflow of layer properties (`NoInflow`).
- **Ice-shelf gaps**: either sinks, like the ice front (`SinkGapsBC`), or part of
  the active domain (`ConnectedGapsBC`); see [Physics](physics.md).

### Wall slip enters two terms

The slip factor is not only a viscous condition. It scales

1. the **viscous wall drag** in the Laplacian, always; and
2. the **momentum carried across a wall face** by the advection, through a ghost
   velocity ``(1 - \text{slip})\,u`` — free slip leaves the flux at its interior
   value, no slip reverses it.

The second is v1's structure, and it is what
`BoundaryConditions(; wall_advection)` selects:
[`SlipScaledWallAdvection`](@ref) (the default) keeps it,
[`NoWallAdvection`](@ref) drops it and leaves the slip factor in the viscous
drag alone, which is what LADDIE v2 does — both of v2's advection schemes skip
grounded neighbours outright, and its only no-slip term is the viscous one.

This interacts with the scheme above, and the interaction is easy to trip over:

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
[ice-shelf gaps example](generated/jesse-gaps.md), `UpstreamMomentumAdvection`
at v2's ``A_h = 10`` lets the layer run away at the grounding line, and needs
``A_h \approx 50`` to hold. That is a missing wall term rather than a physical
result — Laddie.jl has no C-grid counterpart of v2's own ``A_h H u / A``
wall stress — so treat a large ``A_h`` there as the stability crutch it is.

## Stability safeguards

Several limiters guard against numerical blow-up in extreme conditions:

- A **minimum layer thickness** ``D_\min`` is enforced through extra entrainment:
  if a cell would thin below the threshold, the entrainment source is increased
  to restore it; a final clamp catches what remains.
- A **speed cutoff** ``v_\text{cut}`` scales both velocity components by one
  factor wherever the speed exceeds it, preserving the flow direction.
- **Tracer bounds** (`Params.T_min` … `S_max`) keep ``T`` and ``S`` in a plausible
  range; note that they also hide a diverging run from the blow-up check.
- An optional **maximum layer thickness** (`Params.max_layer_thickness`), off by
  default; read its warning before enabling it.

`run!` also checks the prognostic fields for non-finite values at a regular
cadence and aborts with an error instead of integrating NaNs.

## Vertical forcing lookup

The ambient profiles ``T_a(z)`` and ``S_a(z)`` are stored on a uniform depth grid
(1 m for profiles built with `OceanForcing1D`). At each time step the local
plume-base depth ``z_b - D`` indexes this grid and linear interpolation supplies
the ``T_a``, ``S_a`` values that enter the density, entrainment and convection
terms; depths beyond the profile take its end values.

The 1 m grid is the *storage*, not the profile's own resolution: `OceanForcing1D`
sorts, resamples and extrapolates flat from whatever samples it is handed. So the
levels a profile is built on are part of the forcing — handing it an analytic
function every metre and handing it the same function on a model's coarse ocean
levels are two different forcings wherever the profile is curved, such as across
a thermocline. Reproducing another model means sampling where it samples, not
only using its formula.
