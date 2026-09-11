# Physics

LADDIE models the thin, **buoyant meltwater layer** that forms directly beneath
an ice shelf. Melting the ice base releases cold but *fresh* water; freshness
wins over coldness, so the meltwater is buoyant, pools against the sloping ice
draft, and drives a circulation — the **ice pump**. Rather than resolving the
full 3-D cavity, LADDIE collapses that layer into a single **depth-integrated
layer** and evolves five 2-D fields:

| field | meaning | C-grid location |
|-------|---------|-----------------|
| ``D`` | layer thickness (m) | cell centre |
| ``U, V`` | depth-averaged velocity (m/s) | cell faces |
| ``T, S`` | depth-averaged temperature (°C) / salinity (psu) | cell centre |

## Governing balances

The model integrates vertically-averaged shallow-water equations with
thermodynamics. Schematically:

```math
\frac{\partial D}{\partial t} = -\nabla\!\cdot(\mathbf{u}\,D) + \dot{m} + \dot{e}
```

```math
\frac{\partial (D\mathbf{u})}{\partial t} =
-\nabla\!\cdot(\mathbf{u}\,D\,\mathbf{u})
- f\,D\,\mathbf{u}^{\perp}
- g\,D\,\nabla(D\,\Delta\rho')
+ (\text{draft pressure})
- C_d\,|\mathbf{u}|\mathbf{u}
+ A_h\,\nabla^2(D\mathbf{u})
```

```math
\frac{\partial (DT)}{\partial t} = -\nabla\!\cdot(\mathbf{u}\,D\,T)
+ \dot{e}\,T_a + \dot{m}\,T_b - \gamma_T (T - T_b) + K_h\nabla^2 T
```

```math
\frac{\partial (DS)}{\partial t} = -\nabla\!\cdot(\mathbf{u}\,D\,S)
+ \dot{e}\,S_a + K_h\nabla^2 S
\qquad(\text{ice is salt-free})
```

## The closures that make it physical

**Reduced gravity / buoyancy.** A linear equation of state gives the fractional
density contrast with the ambient cavity water,

```math
\Delta\rho' = \beta\,(S_a - S) - \alpha\,(T_a - T),
```

which sets the pressure gradients that push the light meltwater up the draft.

**Three-equation melt** ``\dot{m}`` (Jenkins 1991). Three constraints hold
simultaneously at the ice–ocean interface:

```math
c_p\,\gamma_T\,(T - T_b) = L_\text{eff}\,\dot{m}
\qquad\text{(heat balance)}
```

```math
\gamma_S\,(S - S_b) = S_b\,\dot{m}
\qquad\text{(salt balance; ice is salt-free)}
```

```math
T_b = \lambda_1 S_b + \lambda_2 + \lambda_3\,z_b
\qquad\text{(liquidus constraint)}
```

Here ``T_b`` and ``S_b`` are the boundary temperature and salinity at the ice
face, ``L_\text{eff} = L - c_i T_i`` is the effective latent heat adjusted for
basal ice temperature ``T_i``, and the turbulent exchange velocities
``\gamma_T, \gamma_S`` scale with the friction velocity

```math
u_\star = \sqrt{C_d^{\text{top}}\,(U^2 + V^2 + u_{\text{tide}}^2)}.
```

Eliminating ``T_b`` and ``S_b`` from the three equations above yields a
**quadratic in ``\dot{m}``**, solved pointwise at every grid cell:

```math
\dot{m}^2 + b\,\dot{m} + c = 0,
```

with coefficients (writing ``\tilde{T}_f = \lambda_2 + \lambda_3\,z_b`` for the
depth-dependent freezing temperature at zero salinity):

```math
b = \frac{c_p}{L_\text{eff}}\,\gamma_T\,(\tilde{T}_f - T)
  + \gamma_S\!\left[1 + \frac{c_i}{L_\text{eff}}
    \bigl(\tilde{T}_f + \lambda_1 S\bigr)\right],
```

```math
c = \frac{c_p}{L_\text{eff}}\,\gamma_T\,\gamma_S\,
    \bigl(\tilde{T}_f - T + \lambda_1 S\bigr).
```

The physical (melting) root is

```math
\dot{m} = \frac{-b + \sqrt{b^2 - 4c}}{2}.
```

For `FixedGamTMelting`, ``\gamma_T`` is a prescribed constant and
``\gamma_S = \gamma_T / 35``; for `TurbulentGamTMelting` both are computed from
``u_\star`` via a log-layer formulation (Holland & Jenkins 1999).

``T_i`` comes from the **ice forcing**, not the parameter set: `PrescribedIceForcing`
holds it as a scalar or as a 2D field, so a temperate ice front can sit next to a cold
interior. Colder ice absorbs more heat per unit melt — ``L_\text{eff}`` runs from
3.34e5 J kg⁻¹ at 0 °C to 3.84e5 at −25 °C — so it acts as a direct brake on melting.

**Entrainment** ``\dot e`` (Gaspar 1988 / Gladish 2012). Turbulence at the base
of the layer mixes warm ambient water *upward* into it. This supplies the heat
for melting and is a **positive feedback**: more shear/melt → more entrainment →
more heat → more melt. A small extra entrainment enforces a minimum layer
thickness, and detrainment removes water where the layer is over-thick.

**Ambient profiles** ``T_a, S_a``. Sampled from a prescribed background ocean
profile at the depth of the layer base ``z_b - D``. See the
[ISOMIP+ forcing](generated/forcing.md) example for the warm/cold profiles used
here.

**Coriolis** ``fD``. At ice-shelf scale, rotation steers the meltwater into
boundary currents rather than letting it flow straight up-slope — visible in the
flow-speed plot of the [ISOMIP+ run](generated/isomip_run.md).

## Gaps in the shelf: melt-through

Where melting thins a shelf all the way through, an ice-free **gap** opens inside the
domain. There is no ice base there, so no melt — but what happens to the meltwater layer
arriving from upstream is genuinely uncertain, and the two defensible answers bracket a
large range of coupled ice-sheet response (Jesse et al., 2026). LADDIE.jl offers both as
`Params.gaps_bc`, selected on the same input mask:

| | [`SinkGapsBC`](@ref) (default) | [`ConnectedGapsBC`](@ref) |
|---|---|---|
| gap cell is | open ocean | dynamically active, ice-free |
| ``D, T, S`` in the gap | not stepped | stepped |
| gap edge behaves like | the calving front | the shelf interior |
| heat and momentum arriving at the gap | leave the cavity | continue downstream |

Real gaps are full of sea ice, mélange and bergs, so neither end-member is right on its
own; running both is how the uncertainty is quantified.

A cell marked `4` is ice-free but *active*: the grid's `tmask` includes it, `imask`
(ice-covered cells) does not, and its draft sits at the sea surface, ``z_b = 0``. Three
consequences follow, and they are the whole of the physics change:

- **No melt.** ``\dot m = 0`` by mask, not by the three-equation solve.
- **No ice–ocean heat flux.** With ``\dot m = 0`` the three equations collapse to
  ``T_b = T`` exactly, so the ``\dot m\,T_b - \gamma_T (T - T_b)`` pair in the temperature
  equation vanishes identically — heat is advected across the gap rather than lost to an
  ice base that is not there.
- **Entrainment stays on.** It is a property of the layer's own shear and stratification,
  not of the ice above; its detrainment term, which scales with ``\dot m``, switches
  itself off.

**Convection in gaps.** The ambient profile in a gap is sampled at the sea surface, the
coldest and freshest water in the column, so the layer there reads as convectively
unstable almost unconditionally. [`ResetToAmbient`](@ref) and [`RelaxToAmbient`](@ref) are
therefore restricted to ice-covered cells: resetting a gap cell towards surface ambient
would erase the temperature and salinity anomaly the layer is carrying and quietly rebuild
the meltwater sink that `ConnectedGapsBC` exists to remove. [`ClampDensity`](@ref) is
*not* restricted — a buoyancy floor keeps the layer denser than ambient without
overwriting its tracers, and it is the one convection treatment the reference
implementation also applies over its whole active domain.
