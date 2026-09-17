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

Advective fluxes of volume, heat, salt and momentum are computed with a
flux-form **upwind** scheme (`upwind_advection_T`, `upwind_advection_U`,
`upwind_advection_V`); viscous and diffusive fluxes use a five-point
**Laplacian** weighted by the layer thickness (`laplace_T`, `laplace_U`,
`laplace_V`). All operators are nearest-neighbour stencils written as fused
KernelAbstractions kernels, one pass per term, with the masks applied inside.

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
