# Physics

LADDIE models the thin, **buoyant meltwater layer** that forms directly beneath
an ice shelf. Melting the ice base releases cold but *fresh* water; freshness
wins over coldness, so the meltwater is buoyant, pools against the sloping ice
draft, and drives a circulation — the **ice pump**. Rather than resolving the
full 3-D cavity, LADDIE collapses that layer into a **depth-averaged** representation and
evolves five 2-D fields:

| field | meaning | C-grid location |
|-------|---------|-----------------|
| ``H`` | layer thickness (m) | cell centre |
| ``U, V`` | depth-averaged velocity (m/s) | cell faces |
| ``T, S`` | depth-averaged temperature (°C) / salinity (psu) | cell centre |

## Governing balances

The model integrates depth-averaged shallow-water equations with thermodynamics. Schematically:

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

The [equation reference](equations.md) gives every term as published in
[Lambert et al. (2023)](@cite lambert_modelling_2023) and as integrated by the kernels.

## Parameterisations

Each parameterisation is a family of types under one abstract supertype, and a run picks one
member by passing an instance to [`Params`](@ref), [`BoundaryConditions`](@ref) or
the forcing. The pages of this section describe one family each:

| abstract type | selected with | default |
|---------------|---------------|---------|
| [`AbstractEquationOfState`](@ref) | `Params(; equation_of_state)` | `LinearEOS()` |
| [`AbstractMelting`](@ref) | `Params(; melting)` | `FixedGamTMelting(1.8e-4)` |
| [`AbstractIceConduction`](@ref) | `Params(; ice_conduction)` | `HollandJenkinsConduction()` |
| [`AbstractEntrainment`](@ref) | `Params(; entrainment)` | `GasparEntrainment(2.5)` |
| [`AbstractConvectionScheme`](@ref) | `Params(; convection_scheme)` | `ResetToAmbient(0.005)` |
| [`AbstractFrazilIce`](@ref) | `Params(; frazil_ice)` | `NoFrazilIce()` |
| [`AbstractLateralViscosity`](@ref) | `Params(; lateral_viscosity)` | `PrescribedLateralViscosity()` |
| [`AbstractCoriolisParameter`](@ref) | `Params(; coriolis)` | `CoriolisParameter0D()` |
| [`AbstractOceanForcing`](@ref) | `Model(grid; forcing)` | — |
| [`AbstractGapsBC`](@ref) | `BoundaryConditions(; gaps)` | `SinkGapsBC()` |

Each default is the choice a plain run makes. The [LADDIE versions](../versions.md) page
gives, for every parameterisation and for the numerical choices, the option that v1 and v2
each use. Options marked *experimental* log a warning when constructed.

The ice forcing has a single member, [`PrescribedIceForcing`](@ref), which supplies the
ice temperature ``T_i`` used by the [melting](melting.md) and
[ice conduction](ice_conduction.md) parameterisations.
