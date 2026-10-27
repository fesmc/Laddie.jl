# Governing equations

This page is the **canonical equation reference** for Laddie.jl. Equation
numbers `(1)`–`(14)` follow the description paper

> Lambert, E., Jüling, A., van de Wal, R. S. W., and Holland, P. R. (2023):
> Modelling Antarctic ice shelf basal melt patterns using the one-layer
> Eulerian ocean model LADDIE, *The Cryosphere*, 17, 3203–3228,
> [doi:10.5194/tc-17-3203-2023](https://doi.org/10.5194/tc-17-3203-2023).

For each equation we give the form **as published** and, where the kernels in
[`src/numerics.jl`](https://github.com/fesmc/Laddie.jl) and
[`src/physics.jl`](https://github.com/fesmc/Laddie.jl) differ algebraically
(e.g. expanded out of flux form, or extended), the **form actually integrated**.
The readable per-term reference implementation lives in
`test/equation_terms.jl`, one function per term; the fused time-loop
kernels in `src/numerics.jl` are asserted equal to it by the test suite.

!!! note "Reduced (dimensionless) density"
    The paper carries the *dimensional* density anomaly
    ``\Delta\rho_a = \rho_0\!\left[-\alpha(T_a-T) + \beta(S_a-S)\right]``
    (Eq. 7). The code stores the **dimensionless** reduced density
    ``\delta\rho \equiv \Delta\rho_a/\rho_0 = \beta(S_a-S) - \alpha(T_a-T)``
    (field `m.drho`). The reduced gravity (Eq. 6) is therefore
    ``g_a' = g\,\Delta\rho_a/\rho_0 = g\,\delta\rho``, and the gravity-wave
    speed used by the CFL monitor is ``c = \sqrt{g\,\delta\rho\,H}``. Whenever a
    paper term has ``\Delta\rho_a/\rho_0`` the code has the single field
    `m.drho`; whenever the paper has ``g_a'`` the code has `m.g * m.drho`.

## Notation

The layer thickness is ``H``, as in LADDIE v2 (Lambert et al. 2026). Lambert et al.
(2023) and the Python v1 code call it ``D``; the equations below are written with ``H``.

| symbol | code | meaning | units |
|--------|------|---------|-------|
| ``H`` | `m.H` | layer thickness (``D`` in Lambert et al. 2023) | m |
| ``U, V`` | `m.U`, `m.V` | depth-averaged velocity (C-grid faces) | m s⁻¹ |
| ``T, S`` | `m.T`, `m.S` | depth-averaged temperature / salinity | °C / psu |
| ``\dot m`` | `m.melt` | basal melt rate (``>0`` = melting) | m s⁻¹ |
| ``\dot e`` | `m.nentr` | net entrainment ``= \mathrm{entr} + \mathrm{ent2} - \mathrm{detr}`` | m s⁻¹ |
| ``T_a, S_a`` | `m.Ta`, `m.Sa` | ambient T/S at layer base ``z_b-H`` | °C / psu |
| ``T_b, S_b`` | `m.Tb` (``S_b`` is not stored) | ice–ocean interface (boundary) T/S | °C / psu |
| ``\Delta\rho_a`` | ``\rho_0\,```m.drho` | dimensional plume–ambient density anomaly | kg m⁻³ |
| ``\delta\rho`` | `m.drho` | dimensionless reduced density ``\Delta\rho_a/\rho_0`` | – |
| ``g_a'`` | `m.g * m.drho` | reduced gravity, Eq. (6) | m s⁻² |
| ``z_b`` | `m.z_draft` | ice-base depth (negative below sea level) | m |
| ``f`` | `m.f` | Coriolis parameter | s⁻¹ |
| ``g`` | `m.g` | gravitational acceleration | m s⁻² |
| ``C_d`` | `m.C_d` | quadratic drag coefficient (momentum) | – |
| ``C_d^{\text{top}}`` | `m.C_d_top` | drag coefficient (friction velocity) | – |
| ``A_h, K_h`` | `m.A_h`, `m.K_h` | horizontal viscosity / diffusivity | m² s⁻¹ |

!!! note "``A_h`` is the coefficient only under the default viscosity scheme"
    The momentum equations below write lateral viscosity as ``A_h\nabla^2\mathbf{u}``,
    which is [`PrescribedLateralViscosity`](@ref), the default. Under
    [`NonlinearLateralViscosity`](@ref) the coefficient becomes the shear-dependent
    ``(C_\mathrm{visc}/100)\,\Delta\,|\delta u|`` instead, and `m.A_h` retains only its
    wall-drag role in the grounding-line/land slip terms (it also always sets the
    entrainment length scale in [`HollandEntrainment`](@ref), independent of either
    choice). Note that `dt` control (`_cfl_number`) accounts for advection and gravity
    waves only — there is no viscous stability constraint, so a large viscosity on a
    fine grid is not caught automatically.
| ``\gamma_T, \gamma_S`` | `m.gamT`, `m.gamS` | turbulent exchange velocities | m s⁻¹ |
| ``u_\star`` | `m.ustar` | friction velocity | m s⁻¹ |
| ``\lambda_1,\lambda_2,\lambda_3`` | `m.l1,m.l2,m.l3` | linear-liquidus coefficients | – |

## Prognostic equations (1–5)

The model evolves five depth-integrated 2-D fields. Each kernel in
`src/numerics.jl` expands the flux-form left-hand side
``\partial(H\,q)/\partial t = q\,\partial H/\partial t + H\,\partial q/\partial t``
and divides the momentum/tracer right-hand sides by the (face-interpolated)
thickness, so the prognostic variable stepped is ``q`` itself, not ``Hq``.

### (1) Layer thickness — continuity

```math
\frac{\partial H}{\partial t} + \nabla\!\cdot(H\mathbf{U}) = \dot m + \dot e
\tag{1}
```

Kernel `_step_thickness_kernel!`: ``H^{+} = H^{0} + (\mathrm{convH} + \dot m + \dot e - \dot f)\,\Delta t``,
where `convH` ``= -\nabla\!\cdot(H\mathbf{U})`` (advection), `m.melt` ``=\dot m``,
`m.nentr` ``=\dot e`` and `m.frazil` ``=\dot f``, the frazil-ice formation (zero unless
`frazil_ice = RelaxedFrazilIce`, see [Frazil ice](@ref)). The cap detrainment `detr_cap`
is a sink of the same kind.

### (2) ``U``-momentum

```math
\frac{\partial HU}{\partial t} + \nabla\!\cdot(H\mathbf{U}U) - fHV =
-\frac{gH^2}{2\rho_0}\frac{\partial \Delta\rho_a}{\partial x}
+ g_a'H\,\frac{\partial (z_b - H)}{\partial x}
- C_d\,|\mathbf{U}|\,U
+ \nabla\!\cdot(A_h H\nabla U)
\tag{2}
```

Kernel `_step_u_momentum_kernel!` integrates the **expanded** form (per unit
``\bar H``); term by term, the right-hand side `rhs` is:

```math
\underbrace{-U\,\partial_t \bar H}_{\text{thickness coupling}}
\;+\; \underbrace{\mathrm{cU}}_{-\nabla\cdot(H\mathbf{U}U)}
\;-\; \underbrace{g\,\overline{H\delta\rho}\,\frac{H_{x-1}-H}{\Delta x}}_{g_a'H\,\partial_x(z_b-H)\ \text{(depth part)}}
\;+\; \underbrace{g\,\overline{H\delta\rho\,\partial_x z_b}}_{\text{ice-base slope part}}
\;-\; \underbrace{\tfrac{1}{2}g\,\bar H^2\,\frac{\partial \delta\rho}{\partial x}}_{-\,gH^2/(2\rho_0)\,\partial_x\Delta\rho_a}
\;+\; \underbrace{f\,\overline{H V}}_{\text{Coriolis}}
\;-\; \underbrace{C_d\,U|\mathbf{U}|}_{\text{drag}}
\;+\; \underbrace{A_h\nabla^2 U}_{\text{diffusion}}
\;-\; \underbrace{\mathrm{detr}\cdot U}_{\text{detrainment loss}}
```

then ``U^{+} = U^{p} + (\mathrm{rhs}/\bar H)\,\Delta t``. Overbars denote C-grid
face interpolation to the ``U``-node. The detrainment-momentum-loss term is a
Laddie.jl addition (see [Entrainment](@ref)).

At a **one-sided face** — the ice front, or a `SinkGapsBC` gap-sink edge — the neighbour
needed for ``H_{x-1}`` is not part of the active domain, and masking pins its stored
thickness to `0`. The depth-gradient term ``g\,\overline{H\delta\rho}\,(H_{x-1}-H)/\Delta x``
is then a full one-sided gradient, as if ``H`` collapsed to zero across one grid cell.
The two reference implementations disagree on what to do about this, so
[`AbstractFrontPressure`](@ref) makes it a choice:

- [`FullDepthGradient`](@ref) (**default**) keeps the term, reproducing Python LADDIE
  v1.x, which Laddie.jl is ported from and validated against.
- [`TruncatedDepthGradient`](@ref) drops it, as the LADDIE v2 Fortran reference does at
  calving-front faces. Measured on a warm ISOMIP+ run, the term is ~150× larger at the
  ice front than in the interior under the default.
- [`ZeroGradientFront`](@ref) drops it and the density-gradient term
  ``\tfrac12 g\,\bar H^2\,(\delta\rho_{x-1}-\delta\rho)/\Delta x`` as well, whose
  neighbour ``\delta\rho`` is likewise masked to `0` there. This is the zero lateral
  gradient at the calving front stated in the LADDIE v2 description paper; the v2
  Fortran keeps the density term.

Neither affects the grounding line, where no momentum equation is solved.

### (3) ``V``-momentum

```math
\frac{\partial HV}{\partial t} + \nabla\!\cdot(H\mathbf{U}V) + fHU =
-\frac{gH^2}{2\rho_0}\frac{\partial \Delta\rho_a}{\partial y}
+ g_a'H\,\frac{\partial (z_b - H)}{\partial y}
- C_d\,|\mathbf{U}|\,V
+ \nabla\!\cdot(A_h H\nabla V)
\tag{3}
```

Kernel `_step_v_momentum_kernel!`, structurally identical to (2) with
``x\to y`` and the Coriolis sign flipped (``-f\,\overline{H U}``).

### (4) Heat

```math
\frac{\partial HT}{\partial t} + \nabla\!\cdot(H\mathbf{U}T) =
\dot e\,T_a + \dot m\,T_b - \gamma_T (T - T_b) + \nabla\!\cdot(K_h H\nabla T)
\tag{4}
```

Kernel `_step_temperature_kernel!` (and `mat_*` variants for array-valued
``\gamma_T``/`conv2`). The implemented `rhs` adds two Laddie.jl terms beyond the
paper: the convective relaxation ``-(T^{p}-T_a)\,\mathrm{conv2}`` (zero unless
`convection_scheme = RelaxToAmbient`, see [Convective relaxation](@ref)), and the frazil
term ``\dot f\,(L/c_p - T)`` (zero unless `frazil_ice = RelaxedFrazilIce`, see
[Frazil ice](@ref)):

```math
\mathrm{rhs} = -T\,\partial_t H + \mathrm{cT} + \dot e\,T_a + \dot m\,T_b
- \gamma_T(T - T_b) + K_h\nabla^2 T - (T^{p}-T_a)\,\mathrm{conv2}
+ \dot f\,(L/c_p - T),
```

with `cT` ``=-\nabla\!\cdot(H\mathbf{U}T)``, then ``T^{+} = T^{p} + (\mathrm{rhs}/H)\,\Delta t``.
That is the default [`AdvectiveTracerForm`](@ref). Under [`FluxTracerForm`](@ref)
the ``-T\,\partial_t H`` term is left out of `rhs` and the content is stepped
instead, ``T^{+} = (H^{p}T^{p} + \mathrm{rhs}\,\Delta t)/H^{+}``.

### (5) Salt

```math
\frac{\partial HS}{\partial t} + \nabla\!\cdot(H\mathbf{U}S) =
\dot e\,S_a + \nabla\!\cdot(K_h H\nabla S)
\tag{5}
```

Kernel `_step_salinity_kernel!`. Identical structure to (4) but with **no**
ice–ocean exchange term (ice is salt-free), plus the same optional `conv2`
relaxation toward ``S_a``. Frazil adds no term: the ice is salt-free, so its salt stays in
the layer, and ``-S\,\partial_t H`` concentrates it as the layer thins.

## Closures

### (6) Reduced gravity & (7) equation of state

```math
g_a' = \frac{g\,\Delta\rho_a}{\rho_0}, \tag{6}
\qquad
\Delta\rho_a = \rho_0\!\left[-\alpha(T_a - T) + \beta(S_a - S)\right]. \tag{7}
```

`update_density!` / `_density_kernel!` computes the dimensionless
``\delta\rho = \beta(S_a-S) - \alpha(T_a-T)`` into `m.drho` (masked).

### (8–10) Three-equation melt parameterisation

```math
c_p\,\gamma_T\,(T - T_b) = \dot m\,L + \dot m\,c_i\,(T_b - T_i) \tag{8}
```
```math
\gamma_S\,(S - S_b) = \dot m\,S_b \tag{9}
```
```math
T_b = \lambda_1 S_b + \lambda_2 + \lambda_3 z_b \tag{10}
```

Equation (10) is the linear liquidus.
Define the **effective latent heat** ``L_\text{eff} = L - c_i T_i``, where the basal
ice temperature ``T_i`` is supplied per cell by the ice forcing. Eliminating
``T_b, S_b`` gives a quadratic in ``\dot m``, solved pointwise in
`_three_eq_melt_kernel!` with ``\tilde T_f = \lambda_2 + \lambda_3 z_b``:

```math
\dot m = \frac{-b + \sqrt{\max(0,\;b^2 - 4c)}}{2},
```
```math
b = \frac{c_p}{L_\text{eff}}\gamma_T(\tilde T_f - T)
  + \gamma_S\!\left[1 + \frac{c_i}{L_\text{eff}}(\tilde T_f + \lambda_1 S)\right],
\qquad
c = \frac{c_p}{L_\text{eff}}\gamma_T\gamma_S(\tilde T_f - T + \lambda_1 S).
```

The interface temperature is then
``T_b = (c_p\gamma_T\,T/L_\text{eff} - \dot m)/(c_p\gamma_T/L_\text{eff} + c_i\dot m/L_\text{eff})``.

The ``c_i`` terms are the conduction into the ice of `ice_conduction =
HollandJenkinsConduction` (the default). With `ice_conduction` set to another
[`AbstractIceConduction`](@ref), the kernel replaces ``c_i`` per cell by
``c_{i,\text{eff}} = c_i\,w`` with ``w \in \{0, 1\}`` set by the sign of ``T - T_f(S, z_b)``
(`MeltOnlyConduction`: 1 where melting, 0 otherwise; `NoConduction`, `LinearProfileConduction`: 0),
and adds the fixed flux ``q = k_i(T_f - T_s)/(\rho_\text{fw} H_i)`` of `LinearProfileConduction`
to the heat balance (8), which amounts to evaluating the equations above at
``T - q/(c_p\gamma_T)`` in place of ``T``.

### (11–12) Turbulent exchange coefficients

```math
\gamma_T = \frac{u_\star}{2.12\,\ln(u_\star H/\nu_0) + 12.5\,\mathrm{Pr}^{2/3} - 8.68} \tag{11}
```
```math
\gamma_S = \frac{u_\star}{2.12\,\ln(u_\star H/\nu_0) + 12.5\,\mathrm{Sc}^{2/3} - 8.68} \tag{12}
```

Computed by `_compute_turbulent_transfer_coefficients!` for
`melting = TurbulentGamTMelting` (the log argument carries a `+1e-12` floor for
``H\to 0``). For `melting = FixedGamTMelting`, ``\gamma_T`` is a prescribed constant
and ``\gamma_S = \gamma_T/35``.

### (13) Friction velocity

```math
u_\star = \sqrt{C_d^{\text{top}}\,(U^2 + V^2 + u_{\text{tide}}^2)} \tag{13}
```

Kernel `_ustar_kernel!`, with ``U,V`` interpolated from the C-grid faces to the
T-point: ``U \to \tfrac12(U_{i,j}+U_{i,j-1})``, ``V \to \tfrac12(V_{i,j}+V_{i-1,j})``.

### (14) Entrainment / detrainment

The paper's mechanical-energy balance (Gaspar 1988; Gladish et al. 2012) is

```math
\frac{H}{2}\, g_b'\,\dot m + \frac{H}{2}\, g_a'\,\dot e = \mu\,u_\star^3, \tag{14}
```

(Eq. 13 in Lambert et al. 2026), where ``g_b' = g\,\delta\rho_b`` uses the **plume–interface** density contrast
``\delta\rho_b = \beta(S - S_b) - \alpha(T - T_b)`` (computed in `_buoyancy_entrainment_kernel!`, not stored).
With ``\delta\rho^{+} = \max(\mathrm{drho\_floor}, \delta\rho)`` this is solved for ``\dot e``
(`GasparEntrainment`, the default — the reference LADDIE v1 `'Gaspar'` and v2 `'Gaspar1988'`):

```math
\dot e
= \frac{2\mu}{g}\,\frac{u_\star^3}{H\,\delta\rho^{+}}
\;-\; \frac{\delta\rho_b}{\delta\rho^{+}}\,\dot m,
```

and split into ``\mathrm{entr} = \max(\dot e,0)``,
``\mathrm{detr} = \min(\mathrm{max\_detrainment}, \max(-\dot e,0))``.

!!! note "Reading Eq. (14)"
    The ``H/2`` factors are easy to misread as ``H^2`` in the typeset PDF. The ``2\mu``
    prefactor and single power of ``H`` above follow from ``H/2``, and match both
    reference implementations bit-for-bit.

**Alternative (`entrainment = HollandEntrainment`)** — shear entrainment after
Holland & Jenkins (1999), *not* a Lambert et al. equation:

```math
\mathrm{entr} = \frac{c_l K_h}{A_h^2}\,
\sqrt{\max\!\Big(0,\;|\mathbf{u}|^2 - g_a'\,\frac{K_h}{A_h}\,H\Big)},
\qquad \mathrm{detr} = 0,
```

kernel `_holland_entrainment_kernel!`.

## Laddie.jl extensions (not in the paper)

These terms are zero/identity in the configuration that reproduces the paper, so
the Python-LADDIE verification stays valid.

### Minimum-thickness correction

`update_entrainment!` adds a non-negative `ent2` so ``H`` cannot fall below
`H_min`, and assembles the net entrainment used in Eq. (1):

```math
\mathrm{ent2} = \max\!\Big(0,\;\frac{\mathrm{H_min} - H^{\text{past}}}{2\Delta t}
- (\mathrm{convH} + \dot m + \mathrm{entr} - \mathrm{detr})\Big),
\qquad
\dot e = \mathrm{entr} + \mathrm{ent2} - \mathrm{detr}.
```

### Convective relaxation

For `convection_scheme = RelaxToAmbient`, unstable cells (``\delta\rho < 0``) relax T/S
toward ambient over `convection_time` via the `conv2` field appearing in (4)–(5):

```math
\mathrm{conv2} = \frac{[\delta\rho < 0]\,H}{\mathrm{convection_time}}.
```

`ClampDensity` (clamp ``\delta\rho`` to a positive floor) and `ResetToAmbient`
(reset unstable-cell T/S to ambient) set `conv2 = 0` and act directly on the
fields instead.

### Frazil ice

For `frazil_ice = RelaxedFrazilIce`, a supercooled layer forms frazil ice that is
deposited on the ice base at once. Under the ice cover, the rate is

```math
\dot f = \frac{c_p}{L}\,\frac{H\,\max(T_f - T^{p}, 0)}{\tau},
\qquad T_f = l_1 S^{p} + l_2 + l_3 z_b,
```

with ``\tau`` = `relaxation_time`. The frozen water leaves the layer at its own ``T`` and
velocity: it is subtracted in (1), and the momentum equations carry
``-\overline{\dot f}\,U`` next to the cap detrainment, which cancels its share of the
thickness-tendency correction. Its latent heat ``L\dot f`` enters (4), so ``T`` relaxes
toward ``T_f`` at the rate ``1/\tau``. The salt stays in the layer, see (5).

## Equation → code map

| eq | quantity | source of truth |
|----|----------|-----------------|
| (1) | thickness | `_step_thickness_kernel!` (`numerics.jl`) |
| (2) | ``U``-momentum | `_step_u_momentum_kernel!`; terms: `u_*` fns (`test/equation_terms.jl`) |
| (3) | ``V``-momentum | `_step_v_momentum_kernel!`; terms: `v_*` fns (`test/equation_terms.jl`) |
| (4) | heat | `_step_temperature_kernel!` + `mat_*` variants |
| (5) | salt | `_step_salinity_kernel!` + `mat_*` variant |
| (6)/(7) | reduced gravity / EOS | `update_density!`, `_density_kernel!` |
| (8)–(10) | three-eq melt + liquidus | `_three_eq_melt_kernel!` |
| — | conduction into the ice | `_ice_conduction`, `_conduction_coefficients` |
| — | two-equation melt (`TwoEquationMelting`) | `_two_eq_melt_kernel!`, `_ustar_gamT_kernel!` |
| (11)/(12) | ``\gamma_T,\gamma_S`` | `_compute_turbulent_transfer_coefficients!` |
| (13) | ``u_\star`` | `_ustar_kernel!` |
| (14) | entrainment | `GasparEntrainment` → `_buoyancy_entrainment_kernel!` (default); Holland → `_holland_entrainment_kernel!` |
| — | min-thickness / net ``\dot e`` | `update_entrainment!` |
| — | convective relaxation | `update_convection!`, `_update_conv2!` |
| — | frazil ice | `update_frazil!`, `_frazil_kernel!` |
