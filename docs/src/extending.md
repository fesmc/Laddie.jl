# Adding a scheme

Every exchangeable part of Laddie.jl is a family of types under one abstract supertype, and
the families are open: a scheme defined in your own code takes part like a built-in one.
Subtype the abstract type, implement the methods of its family (table below) for your type,
and pass an instance where the built-in ones go, `Params(; melting = MyMelting())` or
`Model(grid; forcing = MyOcean())`.

A constant melt rate, for example:

```julia
using Laddie

struct ConstantMelt <: AbstractMelting
    rate::Float64   # m s⁻¹, freshwater equivalent
end

function Laddie.melt!(m, s::ConstantMelt)
    Laddie.ustar!(m)          # u★ also drives the entrainment
    m.melt .= s.rate .* m.imask
    m.Tb .= m.l1 .* m.S.present .+ m.l2 .+ m.l3 .* m.z_draft
    return nothing
end

sim = build_isomip(; params = Params(; melting = ConstantMelt(10 / 31_557_600)))
```

These methods are public and exported, so a scheme defined outside Laddie.jl adds a
method to them (`Laddie.melt!` above, or after `import Laddie: melt!`). Read an
existing scheme of the family first: the file named in the table holds its methods, with their signatures and the
fields of the model they read and write.

## What each family needs

| abstract type | slot | methods to implement for your type | file |
|---|---|---|---|
| [`AbstractEntrainment`](@ref) | `Params(; entrainment)` | `entrainment!(m, scheme)` | `physics/entrainment.jl` |
| [`AbstractMelting`](@ref) | `Params(; melting)` | `melt!(m, scheme)`, which also calls `ustar!(m)` | `physics/melting.jl` |
| [`AbstractConvection`](@ref) | `Params(; convection)` | `convection!(m, scheme)`; `convective_relaxation!` for a relaxation rate (`numerics/leapfrog.jl`) | `physics/convection.jl` |
| [`AbstractFrazilIce`](@ref) | `Params(; frazil_ice)` | `frazil!(m, scheme)` | `physics/frazil.jl` |
| [`AbstractIceConduction`](@ref) | `Params(; ice_conduction)` | `conduction_coefficients` | `physics/conduction.jl` |
| [`AbstractEOS`](@ref) | `Params(; equation_of_state)` | `eos_coefficients` | `physics/eos.jl` |
| [`AbstractMaxLayerThickness`](@ref) | `Params(; max_layer_thickness)` | `cap_layer_thickness` | `physics/max_layer_thickness.jl` |
| [`AbstractCoriolisParameter`](@ref) | `Params(; coriolis)` | `coriolis_field` | `physics/coriolis.jl` |
| [`AbstractTracerBounds`](@ref) | `Params(; tracer_bounds)` | `apply_tracer_bounds!(m, scheme, q, quantity)` | `numerics/tracer_bounds.jl` |
| [`AbstractLateralViscosity`](@ref) | `Params(; lateral_viscosity)` | `laplace_U`, `laplace_V` | `numerics/stencils.jl` |
| [`AbstractLaplacianWeights`](@ref) | `Params(; laplacian_weights)` | `laplacian_thickness` | `numerics/viscosity.jl` |
| [`AbstractMomentumAdvection`](@ref) | `Params(; momentum_advection)` | `advect_U`, `advect_V` (`numerics/stencils.jl`); `momentum_outflow` if it has an outflow (`numerics/leapfrog.jl`) | `numerics/advection.jl` |
| [`AbstractTracerForm`](@ref) | `Params(; tracer_form)` | `tracer_update`, `thickness_tendency`, `outflow_rate!` | `numerics/leapfrog.jl` |
| [`AbstractMomentumForm`](@ref) | `Params(; momentum_form)` | `momentum_update`, `momentum_thickness_tendency`, `momentum_flux_terms`, `explicit_sink` | `numerics/leapfrog.jl` |
| [`AbstractClosureLevel`](@ref) | `Params(; closure_level)` | `closure_field` | `numerics/advection.jl` |
| [`AbstractKinks`](@ref) | `Params(; kinks)` | `kink_width` and the smooth switches (`smooth_max`, `smooth_min`, `smooth_clamp`, …) | `numerics/kinks.jl` |
| [`AbstractIceSlope`](@ref) | `Params(; ice_slope)` | `icebase_slope` | `simulation/grid.jl` |
| [`AbstractFrontPressure`](@ref) | `Params(; front_pressure)` | `front_pgf_weight`, `front_drho_weight` | `bcs/open_ocean.jl` |
| [`AbstractOpenOceanBC`](@ref) | `BoundaryConditions(; open_ocean)` | `inflow_weight`; `clip_front_inflow!` for a valve | `bcs/open_ocean.jl` |
| [`AbstractSlipAtGL`](@ref) | `BoundaryConditions(; grounding_line_slip)` | `slip_factor` | `bcs/walls.jl` |
| [`AbstractSlipAtLand`](@ref) | `BoundaryConditions(; land_slip)` | `slip_factor` | `bcs/walls.jl` |
| [`AbstractWallAdvection`](@ref) | `BoundaryConditions(; wall_advection)` | `advection_slips` | `bcs/walls.jl` |
| [`AbstractGapsBC`](@ref) | `BoundaryConditions(; gaps)` | `gaps_mask` | `bcs/gaps.jl` |
| [`AbstractOceanForcing`](@ref) | `CavityForcing(ocean, ice)` | `ambient_fields!(m, forcing)`, writing `m.Ta` and `m.Sa` | `bcs/forcing.jl` |
| [`AbstractIceForcing`](@ref) | `CavityForcing(ocean, ice)` | a field `T_ice_farfield` | `bcs/forcing.jl` |
| [`AbstractTimeStepper`](@ref) | `Simulation(model; time_stepper)` | `init_adaptive_dt!`, `adapt_dt!` | `numerics/timestepping.jl` |
| [`AbstractCFL`](@ref) | `Simulation(model; cfl)` | `cfl_rate`, `cfl_reductions` | `numerics/timestepping.jl` |
| [`AbstractSimulationEnd`](@ref) | `Simulation(model; stop)` | a field `t_end` (days), or `end_seconds`; `needs_melt_sample` and `steady_reached` for a criterion on the mean melt | `simulation/simulation.jl` |
| [`AbstractPreprocess`](@ref) | `Grid(...; preprocess)` | `preprocess!` | `utils/preprocessing.jl` |
| [`AbstractDomainCropping`](@ref) | `Grid(...; domain_cropping)` | `crop_ranges` | `utils/preprocessing.jl` |

## Rules for a new scheme

- **Methods called inside kernels** (equation of state, conduction, kinks, the tracer and
  momentum forms, the advection stencils, the slip factors) must be `@inline`, type-stable,
  allocation-free and branch-free (`ifelse`), and read the numbers of the scheme struct
  through `Laddie._val`, as the built-in ones do.
- **Float fields** of a scheme are converted to the model's precision by `Params` and
  `BoundaryConditions` through its positional constructor; give the struct one that accepts
  its fields in order (the default constructor does).
- **Scheme-dependent work fields** (`m.gamT`, `m.gamS`, `m.conv2`, `m.frazil`) are full
  fields for a scheme Laddie.jl does not know, so your scheme can write them.
- **A forcing's fields are forwarded by the model**, like `m.T_ice_farfield`. Their names
  must differ from those of the model's other fields (`T`, `S`, `H`, …), or the model refuses
  to build.
- **Differentiability**: guard non-differentiable points with `Laddie._safe_sqrt` and
  `Laddie._safe_div`, and route a new `max`/`min` through the kink helpers of
  [`AbstractKinks`](@ref), so that the scheme keeps working with ForwardDiff and Reactant.

## Extension methods

The methods of the table, grouped by family, and the helpers a new scheme calls.

```@docs
entrainment!
melt!
ustar!
convection!
convective_relaxation!
frazil!
conduction_coefficients
eos_coefficients
cap_layer_thickness
coriolis_field
apply_tracer_bounds!
laplace_U
laplace_V
laplacian_thickness
advect_U
advect_V
momentum_outflow
tracer_update
thickness_tendency
outflow_rate!
momentum_update
momentum_thickness_tendency
momentum_flux_terms
explicit_sink
closure_field
kink_width
smooth_max
smooth_min
smooth_clamp
icebase_slope
front_pgf_weight
front_drho_weight
inflow_weight
clip_front_inflow!
slip_factor
advection_slips
gaps_mask
ambient_fields!
init_adaptive_dt!
adapt_dt!
cfl_reductions
cfl_rate
end_seconds
needs_melt_sample
steady_reached
preprocess!
crop_ranges
```
