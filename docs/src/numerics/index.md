# Numerics

## Spatial discretisation

LADDIE uses an Arakawa **C-grid**: scalar fields (``H``, ``T``, ``S``) and all
diagnostic quantities live at cell centres, while ``U`` sits on the faces between
cells ``i`` and ``i+1`` and ``V`` on the faces between ``j`` and ``j+1``. Arrays are
stored `[x, y]`, and x and y are grid axes, not compass directions. Staggered
interpolation operators move quantities between the two locations as needed by
the pressure-gradient, Coriolis and drag terms. Boundary conditions are encoded in
floating-point mask arrays (`tmask`, `umask`, `vmask`, and the wall indicators)
rather than in branching logic, so the same kernels execute unchanged on CPU and
GPU.

Advective fluxes of volume, heat and salt are computed with a flux-form
**upwind** scheme; viscous and diffusive fluxes use a five-point **Laplacian**
weighted by the layer thickness. All operators are nearest-neighbour stencils written
as fused KernelAbstractions kernels, one pass per term, with the masks applied
inside. How momentum is advected, and in which form the tracer equations are stepped,
are choices of their own. The upwind switches, floors and caps can be smoothed for
gradient-based work; see [Kinks](kinks.md).

## Time integration

The prognostic variables advance with a leapfrog scheme and a Robert–Asselin filter, at
a constant step or one set by a CFL controller; see [Time stepping](time_stepping.md).

## Domain, boundaries, and masks

The domain is padded with a **one-cell border** of land on all sides. This lets
all stencil operators use **periodic wrap** throughout — there are no special edge
cases — while the border cells are masked out and contribute nothing to the
dynamics. The physical boundaries are:

- **Grounding line and land**: zero-flux walls, each with its own slip factor; see
  [Wall slip](walls.md) and [Wall momentum advection](wall_advection.md).
- **Ice front**: an open boundary; see [Open-ocean boundary](open_ocean.md) and
  [Ice-front pressure gradient](front_pressure.md).
- **Ice-shelf gaps**: either sinks, like the ice front, or part of the active domain;
  see [Ice-shelf gaps](../physics/gaps.md).

Before any of this, the input mask can be cleaned up by a list of
[preprocessing steps](preprocessing.md) and [cropped](domain_cropping.md) to the cavity,
and the [ice-base slope](ice_slope.md) is derived from the draft.

## Stability safeguards

Several limiters guard against numerical blow-up in extreme conditions:

- A **minimum layer thickness** ``H_\min`` is enforced through extra entrainment:
  if a cell would thin below the threshold, the entrainment source is increased
  to restore it (see [Entrainment](../physics/entrainment.md)); a final clamp catches
  what remains.
- A **speed cutoff** ``v_\text{cut}`` scales both velocity components by one
  factor wherever the speed exceeds it, preserving the flow direction.
- **Tracer bounds** (`Params.T_min` … `S_max`) keep ``T`` and ``S`` in a plausible
  range; note that they also hide a diverging run from the blow-up check.
- An optional [maximum layer thickness](max_layer_thickness.md), off by default,
  enforced the same way through detrainment of layer water at its own ``T``, ``S``
  and velocity; read its warning before enabling it.

`run!` also checks the prognostic fields for non-finite values at a regular
cadence and aborts with an error instead of integrating NaNs.

## Numerical choices

Each choice is a family of types under one abstract supertype. The pages of this section
describe one family each:

| page | abstract type | selected with | default |
|------|---------------|---------------|---------|
| [Momentum advection](momentum_advection.md) | [`AbstractMomentumAdvection`](@ref) | `Params(; momentum_advection)` | `CentredMomentumAdvection()` |
| [Tracer form](tracer_form.md) | [`AbstractTracerForm`](@ref) | `Params(; tracer_form)` | `FluxTracerForm()` |
| [Momentum form](momentum_form.md) | [`AbstractMomentumForm`](@ref) | `Params(; momentum_form)` | `AdvectiveMomentumForm()` |
| [Closure level](momentum_form.md) | [`AbstractClosureLevel`](@ref) | `Params(; closure_level)` | `PresentClosures()` |
| [Laplacian weights](laplacian_weights.md) | [`AbstractLaplacianWeights`](@ref) | `Params(; laplacian_weights)` | `PresentLaplacianWeights()` |
| [Time stepping](time_stepping.md) | [`AbstractTimeStepper`](@ref) | `Simulation(model; tstep)` | `FixedDt()` |
| [CFL number](cfl.md) | [`AbstractCFL`](@ref) | `Simulation(model; cfl)` | `ExactCFL()` |
| [Simulation end](simulation_end.md) | [`AbstractSimulationEnd`](@ref) | `Simulation(model; stop)`, `run!(sim; until)` | `FixedSimulationEnd()` |
| [Wall slip](walls.md) | [`AbstractGroundingLineBC`](@ref), [`AbstractLandBC`](@ref) | `BoundaryConditions(; grounding_line, land)` | `NoSlipGL()`, `NoSlipLand()` |
| [Wall momentum advection](wall_advection.md) | [`AbstractWallAdvection`](@ref) | `BoundaryConditions(; wall_advection)` | `SlipScaledWallAdvection()` |
| [Open-ocean boundary](open_ocean.md) | [`AbstractOpenOceanBC`](@ref) | `BoundaryConditions(; open_ocean)` | `ZeroGradientInflow()` |
| [Ice-front pressure gradient](front_pressure.md) | [`AbstractFrontPressure`](@ref) | `Params(; front_pressure)` | `FullDepthGradient()` |
| [Maximum layer thickness](max_layer_thickness.md) | [`AbstractMaxLayerThickness`](@ref) | `Params(; max_layer_thickness)` | `NoMaxLayerThickness()` |
| [Ice-base slope](ice_slope.md) | [`AbstractIceSlopeGradient`](@ref) | `Model(grid; gradient)` | `JlGradient()` |
| [Mask preprocessing](preprocessing.md) | [`AbstractPreprocess`](@ref) | `Grid(...; preprocess)` | none |
| [Domain cropping](domain_cropping.md) | [`AbstractDomainCropping`](@ref) | `Grid(...; domain_cropping)` | `MinRectangleDomainCropping()` |

How `run!` executes the steps — kernel by kernel, or compiled in batches — follows from
the backend; see [Reactant backend](@ref).
