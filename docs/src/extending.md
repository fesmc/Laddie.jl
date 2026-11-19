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

function Laddie.update_melt!(m, s::ConstantMelt)
    Laddie.update_ustar!(m)          # u★ also drives the entrainment
    m.melt .= s.rate .* m.imask
    m.Tb .= m.l1 .* m.S.present .+ m.l2 .+ m.l3 .* m.z_draft
    return nothing
end

sim = build_isomip(; params = Params(; melting = ConstantMelt(10 / 31_557_600)))
```

The methods have a leading underscore where they are internal helpers of the built-in
schemes; extending them is supported all the same. Read an existing scheme of the family
first: the file named in the table holds its methods, with their signatures and the
fields of the model they read and write.

## What each family needs

| abstract type | slot | methods to implement for your type | file |
|---|---|---|---|
| [`AbstractEntrainment`](@ref) | `Params(; entrainment)` | `_compute_entrainment!(m, scheme)` | `physics/entrainment.jl` |
| [`AbstractMelting`](@ref) | `Params(; melting)` | `update_melt!(m, scheme)`, which also calls `update_ustar!(m)` | `physics/melting.jl` |
| [`AbstractConvection`](@ref) | `Params(; convection)` | `update_convection!(m, scheme)`; `_update_conv2!` for a relaxation rate (`numerics/leapfrog.jl`) | `physics/convection.jl` |
| [`AbstractFrazilIce`](@ref) | `Params(; frazil_ice)` | `update_frazil!(m, scheme)` | `physics/frazil.jl` |
| [`AbstractIceConduction`](@ref) | `Params(; ice_conduction)` | `_conduction_coefficients` | `physics/conduction.jl` |
| [`AbstractEOS`](@ref) | `Params(; equation_of_state)` | `_eos_coefficients` | `physics/eos.jl` |
| [`AbstractMaxLayerThickness`](@ref) | `Params(; max_layer_thickness)` | `_max_layer_thickness` | `physics/max_layer_thickness.jl` |
| [`AbstractCoriolisParameter`](@ref) | `Params(; coriolis)` | `_coriolis_field` | `physics/coriolis.jl` |
| [`AbstractTracerBounds`](@ref) | `Params(; tracer_bounds)` | `apply_tracer_bounds!(m, scheme, q, quantity)` | `numerics/tracer_bounds.jl` |
| [`AbstractLateralViscosity`](@ref) | `Params(; lateral_viscosity)` | `laplace_U`, `laplace_V` | `numerics/stencils.jl` |
| [`AbstractLaplacianWeights`](@ref) | `Params(; laplacian_weights)` | `laplacian_thickness` | `numerics/viscosity.jl` |
| [`AbstractMomentumAdvection`](@ref) | `Params(; momentum_advection)` | `_advect_U`, `_advect_V` (`numerics/stencils.jl`); `_momentum_outflow` if it has an outflow (`numerics/leapfrog.jl`) | `numerics/advection.jl` |
| [`AbstractTracerForm`](@ref) | `Params(; tracer_form)` | `_tracer_update`, `_thickness_tendency`, `_outflow_rate!` | `numerics/leapfrog.jl` |
| [`AbstractMomentumForm`](@ref) | `Params(; momentum_form)` | `_momentum_update`, `_momentum_thickness_tendency`, `_momentum_flux_terms`, `_explicit_sink` | `numerics/leapfrog.jl` |
| [`AbstractClosureLevel`](@ref) | `Params(; closure_level)` | `_closure_level` | `numerics/advection.jl` |
| [`AbstractKinks`](@ref) | `Params(; kinks)` | `_kink_width` and the smooth switches (`_smax`, `_smin`, `_sclamp`, …) | `numerics/kinks.jl` |
| [`AbstractIceSlope`](@ref) | `Params(; ice_slope)` | `_icebase_slope` | `simulation/grid.jl` |
| [`AbstractFrontPressure`](@ref) | `Params(; front_pressure)` | `_front_pgf_weight` | `bcs/open_ocean.jl` |
| [`AbstractOpenOceanBC`](@ref) | `BoundaryConditions(; open_ocean)` | `_inflow_weight`; `clip_front_inflow!` for a valve | `bcs/open_ocean.jl` |
| [`AbstractSlipAtGL`](@ref) | `BoundaryConditions(; grounding_line)` | `_gl_slip` | `bcs/walls.jl` |
| [`AbstractSlipAtLand`](@ref) | `BoundaryConditions(; land)` | `_land_slip` | `bcs/walls.jl` |
| [`AbstractWallAdvection`](@ref) | `BoundaryConditions(; wall_advection)` | `_advection_slips` | `bcs/walls.jl` |
| [`AbstractGapsBC`](@ref) | `BoundaryConditions(; gaps)` | `_apply_gaps_bc` | `bcs/gaps.jl` |
| [`AbstractOceanForcing`](@ref) | `CavityForcing(ocean, ice)` | `_update_ambient_fields!(m, forcing)`, writing `m.Ta` and `m.Sa` | `bcs/forcing.jl` |
| [`AbstractIceForcing`](@ref) | `CavityForcing(ocean, ice)` | a field `T_ice_farfield` | `bcs/forcing.jl` |
| [`AbstractTimeStepper`](@ref) | `Simulation(model; tstep)` | `_init_adaptive_dt!`, `_maybe_adapt_dt!` | `numerics/timestepping.jl` |
| [`AbstractCFL`](@ref) | `Simulation(model; cfl)` | `_cfl_rate`, `_cfl_reductions` | `numerics/timestepping.jl` |
| [`AbstractSimulationEnd`](@ref) | `Simulation(model; stop)` | a field `t_end` (days), or `_end_seconds`; `_needs_melt_sample` and `_steady_reached` for a criterion on the mean melt | `simulation/simulation.jl` |
| [`AbstractPreprocess`](@ref) | `Grid(...; preprocess)` | `preprocess!` | `utils/preprocessing.jl` |
| [`AbstractDomainCropping`](@ref) | `Grid(...; domain_cropping)` | `_crop_ranges` | `utils/preprocessing.jl` |

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
