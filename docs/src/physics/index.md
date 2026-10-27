# Physics

LADDIE models the thin, **buoyant meltwater layer** that forms directly beneath
an ice shelf. Melting the ice base releases cold but *fresh* water; freshness
wins over coldness, so the meltwater is buoyant, pools against the sloping ice
draft, and drives a circulation — the **ice pump**. Rather than resolving the
full 3-D cavity, LADDIE collapses that layer into a single **depth-integrated
layer** and evolves five 2-D fields:

| field | meaning | C-grid location |
|-------|---------|-----------------|
| ``H`` | layer thickness (m) | cell centre |
| ``U, V`` | depth-averaged velocity (m/s) | cell faces |
| ``T, S`` | depth-averaged temperature (°C) / salinity (psu) | cell centre |

## Governing balances

The model integrates vertically-averaged shallow-water equations with
thermodynamics. Schematically:

```math
\frac{\partial H}{\partial t} = -\nabla\!\cdot(\mathbf{u}\,H) + \dot{m} + \dot{e}
```

```math
\frac{\partial (H\mathbf{u})}{\partial t} =
-\nabla\!\cdot(\mathbf{u}\,H\,\mathbf{u})
- f\,H\,\mathbf{u}^{\perp}
- g\,H\,\nabla(H\,\Delta\rho')
+ (\text{draft pressure})
- C_d\,|\mathbf{u}|\mathbf{u}
+ A_h\,\nabla^2(H\mathbf{u})
```

```math
\frac{\partial (HT)}{\partial t} = -\nabla\!\cdot(\mathbf{u}\,H\,T)
+ \dot{e}\,T_a + \dot{m}\,T_b - \gamma_T (T - T_b) + K_h\nabla^2 T
```

```math
\frac{\partial (HS)}{\partial t} = -\nabla\!\cdot(\mathbf{u}\,H\,S)
+ \dot{e}\,S_a + K_h\nabla^2 S
\qquad(\text{ice is salt-free})
```

The [equation reference](equations.md) gives every term as published in Lambert et al.
(2023) and as integrated by the kernels.

## Closures

Each closure is a family of types under one abstract supertype, and a run picks one
member by passing an instance to [`Params`](@ref), [`BoundaryConditions`](@ref) or
the forcing. The pages of this section describe one family each:

| page | abstract type | selected with | default |
|------|---------------|---------------|---------|
| [Equation of state](equation_of_state.md) | [`AbstractEquationOfState`](@ref) | `Params(; equation_of_state)` | `LinearEOS()` |
| [Melting](melting.md) | [`AbstractMelting`](@ref) | `Params(; melting)` | `FixedGamTMelting(1.8e-4)` |
| [Ice conduction](ice_conduction.md) | [`AbstractIceConduction`](@ref) | `Params(; ice_conduction)` | `HollandJenkinsConduction()` |
| [Entrainment](entrainment.md) | [`AbstractEntrainment`](@ref) | `Params(; entrainment)` | `GasparEntrainment(2.5)` |
| [Convection](convection.md) | [`AbstractConvectionScheme`](@ref) | `Params(; convection_scheme)` | `ResetToAmbient(0.005)` |
| [Frazil ice](frazil_ice.md) | [`AbstractFrazilIce`](@ref) | `Params(; frazil_ice)` | `NoFrazilIce()` |
| [Lateral viscosity](lateral_viscosity.md) | [`AbstractLateralViscosity`](@ref) | `Params(; lateral_viscosity)` | `PrescribedLateralViscosity()` |
| [Coriolis parameter](coriolis.md) | [`AbstractCoriolisParameter`](@ref) | `Params(; coriolis)` | `CoriolisParameter0D()` |
| [Ocean forcing](ocean_forcing.md) | [`AbstractOceanForcing`](@ref) | `Model(grid; forcing)` | — |
| [Ice-shelf gaps](gaps.md) | [`AbstractGapsBC`](@ref) | `BoundaryConditions(; gaps)` | `SinkGapsBC()` |

The defaults reproduce Python LADDIE v1.x, against which Laddie.jl is validated, except
for the no-slip walls (see [Wall slip](../numerics/walls.md)). Options marked
*experimental* log a warning when constructed.

The ice forcing has a single member, [`PrescribedIceForcing`](@ref), which supplies the
ice temperature ``T_i`` used by the [melting](melting.md) and
[ice conduction](ice_conduction.md) closures.
