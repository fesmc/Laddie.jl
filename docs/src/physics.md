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

## The closures that make it physical

**Reduced gravity / buoyancy.** A linear equation of state gives the fractional
density contrast with the ambient cavity water,

```math
\Delta\rho' = \beta\,(S_a - S) - \alpha\,(T_a - T),
```

which sets the pressure gradients that push the light meltwater up the draft.

The seawater equation of state is nonlinear, mostly in the thermal expansion ``\alpha``, which
falls towards the freezing point (**cabbeling**) and rises with depth (**thermobaricity**).
`Params(; equation_of_state = ...)` selects it:

| equation of state | ``\delta\rho`` | ``\alpha`` |
|---|---|---|
| `LinearEOS()` (default) | ``\beta\,\Delta S - \alpha\,\Delta T`` | constant, `Params.alpha` |
| `RoquetEOS()` | ``\beta\,\Delta S - (a_0 + a_T \bar T + a_Z Z)\,\Delta T`` | ``\bigl[C_b(\Theta - \Theta_0) + T_h Z\bigr]/\rho_0`` |

where ``\Delta X = X_a - X``, ``\bar T = (T_a + T)/2`` and ``Z`` is the depth of the layer
mid-point.  `RoquetEOS` is the simplified polynomial of Roquet et al. (2015, Eq. 17),
``\rho' = -\tfrac{C_b}{2}(\Theta - \Theta_0)^2 - T_h Z \Theta + b_0 S_A``, with the paper's
``C_b = 0.011`` kg m⁻³ K⁻², ``\Theta_0 = -4.5`` °C, ``T_h = 2.5\times10^{-5}`` kg m⁻⁴ K⁻¹ and
``b_0 = 0.77`` kg m⁻³ (g kg⁻¹)⁻¹.  Its thermal expansion is 0.029 kg m⁻³ K⁻¹ at the surface freezing
point, 0.041 at 500 m, and 0.068 at 0.5 °C and 500 m, against the constant 0.038 of the linear default,
so the buoyancy contrast of a plume against its ambient is weaker where it is warm or deep (by about 5 %
for a cold plume and 25 % for a warm one at 500 m).  The meltwater plume is the one place where the
EOS enters Laddie: through ``\delta\rho`` in the pressure gradient, the entrainment and the convective
stability.

!!! warning "Experimental"
    `RoquetEOS` is experimental and logs a warning when constructed.  Roquet et al. (2015) fit it to the
    open ocean in Conservative Temperature and Absolute Salinity, which Laddie takes as its temperature
    and salinity (only salinity differences enter, so the offset to practical salinity cancels), and say
    it serves understanding rather than realistic simulation (TEOS-10 does).
    Supercooled cavity water is an extrapolation.  `Params.alpha` and `Params.beta` are used by
    `LinearEOS` only, and an ambient profile built as neutral for them (as in the examples) is not neutral
    for another equation of state.

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
``u_\star`` via a log-layer formulation (Holland & Jenkins 1999).  `UStarGamTMelting` takes
``\gamma_T = \Gamma_T u_\star`` and ``\gamma_S = \gamma_T / r`` with ``r`` = 35 by default (an
experimental `ratio`, e.g. 70, is possible; McPhee et al. 2008 give 35–70 for a melting base).

Jenkins et al. (2010) tested these parameterizations against the thinning of the Ronne Ice
Shelf measured by radar, and give ``C_d^{1/2}\Gamma_T = 1.1\times10^{-3}`` and
``C_d^{1/2}\Gamma_S = 3.1\times10^{-5}`` for the three-equation form.  Since ``\Gamma`` multiplies
the ``u_\star`` of this model, which uses `C_d_top`, divide by ``\sqrt{C_d^\text{top}}``: that gives
``\Gamma_T \approx 3.3\times10^{-2}`` for the default `C_d_top = 1.1e-3`, close to the
`UStarGamTMelting` default of ``3\times10^{-2}``.  The same data are fit equally well by the
**two-equation** form (`TwoEquationMelting`; McPhee 1992), which drops the salt balance and sets
the interface at the freezing point of the layer salinity, ``T_b = T_f(S, z_b)``:

```math
c_p\,\gamma_{TS}\,(T - T_b) = \dot{m}\,L_\text{eff} + q_\text{cond}
\;\Rightarrow\;
\dot m = \frac{c_p\,\gamma_{TS}\,(T - T_f) - q_\text{cond}}{L - c_i\,(T_i - T_f)},
```

linear in ``\dot m``, with ``\gamma_{TS} = \Gamma_{TS}\,u_\star`` and
``C_d^{1/2}\Gamma_{TS} = 5.9\times10^{-4}`` (default ``\Gamma_{TS} = 0.0178``).  The two forms agree
when ``\gamma_S \to \infty``, and differ under strong thermal forcing, where the interface salinity
departs from the layer's.  Jenkins et al. expect the three-equation form to be the more general.

!!! warning "Experimental"
    `TwoEquationMelting` and a `UStarGamTMelting` ratio other than 35 are experimental and
    log a warning when constructed.  The two-equation scheme leaves the interface salinity at
    the layer salinity, so the melt correction of the buoyancy-flux entrainment sees no
    salinity contrast, and `gamS` is not used.

``T_i`` comes from the **ice forcing**, not the parameter set: `PrescribedIceForcing`
holds it as a scalar or as a 2D field, so a temperate ice front can sit next to a cold
interior. Colder ice absorbs more heat per unit melt — ``L_\text{eff}`` runs from
3.34e5 J kg⁻¹ at 0 °C to 3.84e5 at −25 °C — so it acts as a direct brake on melting.

The term ``-c_i (T_i - T_b)\,\dot m`` in ``L_\text{eff}`` is the heat conducted into the
ice, in the linearisation of Holland & Jenkins (1999), which assumes that the ice
velocity is proportional to the melt rate.  `Params(; ice_conduction = ...)` selects
other closures from Wiskandt & Jourdain (2025):

| closure | heat conducted into the ice | when it is reasonable |
|---|---|---|
| `HollandJenkinsConduction()` (default) | ``\dot m\,c_i\,(T_b - T_i)``, melting and refreezing | melting; the v1/v2 behaviour |
| `MeltOnlyConduction()` | the same for ``\dot m > 0``, zero otherwise | as above, and a refreezing base of marine ice near the freezing point |
| `NoConduction()` | none, ``L_\text{eff} = L`` | marine ice only; overestimates melt by ``c_i (T_s - T_f)/L`` (≈ 11 % for ``T_s = -20`` °C) |
| `LinearProfileConduction()` | ``k_i (T_f - T_s)/H_i``, independent of ``\dot m`` | thick, slow ice shelves in thermal steady state (e.g. Ross) |

The switch of `MeltOnlyConduction` is the sign of ``T - T_f(S, z_b)``, which is the sign of
``\dot m``, so the melt equations are solved once.  `LinearProfileConduction` takes ``T_f``
at the layer salinity so that the flux does not depend on ``\dot m`` and the equation
stays quadratic, and the ice thickness from the draft by hydrostatic equilibrium.

!!! warning "Experimental"
    Every closure but the default is experimental, and constructing one logs a warning.
    The `T_ice_base` of the ice forcing is used as the ice temperature ``T_i`` by all of
    them. In Holland & Jenkins (1999) and Wiskandt & Jourdain (2025) this is the ice-shelf
    *surface* temperature ``T_s`` for every closure (the ice at the base itself is at the
    freezing point); Wiskandt & Jourdain use the 1980–2019 mean snow-surface temperature.

**Entrainment** ``\dot e`` (Gaspar 1988 / Gladish 2012). Turbulence at the base
of the layer mixes warm ambient water *upward* into it. This supplies the heat
for melting and is a **positive feedback**: more shear/melt → more entrainment →
more heat → more melt. A small extra entrainment enforces a minimum layer
thickness. Where the buoyancy supplied by melting outweighs the turbulent
production, the net rate turns negative and the layer detrains instead.

**Ambient profiles** ``T_a, S_a``. Sampled from a prescribed background ocean
profile at the depth of the layer base ``z_b - H``. See the
[ISOMIP+ example](generated/isomip.md#Forcing) for the warm/cold profiles used
here.

**Coriolis** ``fH``. At ice-shelf scale, rotation steers the meltwater into
boundary currents rather than letting it flow straight up-slope — visible in the
flow-speed plot of the [ISOMIP+ warm run](generated/isomip.md#Warm-run).

## Gaps in the shelf: melt-through

Where melting thins a shelf all the way through, an ice-free **gap** opens inside the
domain. There is no ice base there, so no melt — but what happens to the meltwater layer
arriving from upstream is genuinely uncertain, and the two defensible answers bracket a
large range of coupled ice-sheet response (Jesse et al., 2026). LADDIE.jl offers both as
`BoundaryConditions(; gaps = ...)`, selected on the same input mask:

| | [`SinkGapsBC`](@ref) (default) | [`ConnectedGapsBC`](@ref) |
|---|---|---|
| gap cell is | open ocean | dynamically active, ice-free |
| ``H, T, S`` in the gap | not stepped | stepped |
| gap edge behaves like | the calving front | the shelf interior |
| heat and momentum arriving at the gap | leave the cavity | continue downstream |

Real gaps are full of sea ice, mélange and bergs, so neither end-member is right on its
own; running both is how the uncertainty is quantified. The
[ice-shelf gaps example](generated/jesse-gaps.md) runs both on the paper's own geometries
and compares them with LADDIE v2.

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
