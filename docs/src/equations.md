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
The readable per-term reference implementation lives in the
`*_terms` functions at the bottom of `src/physics.jl`; the fused time-loop
kernels in `src/numerics.jl` are asserted equal to it by the test suite.

!!! note "Reduced (dimensionless) density"
    The paper carries the *dimensional* density anomaly
    ``\Delta\rho_a = \rho_0\!\left[-\alpha(T_a-T) + \beta(S_a-S)\right]``
    (Eq. 7). The code stores the **dimensionless** reduced density
    ``\delta\rho \equiv \Delta\rho_a/\rho_0 = \beta(S_a-S) - \alpha(T_a-T)``
    (field `m.drho`). The reduced gravity (Eq. 6) is therefore
    ``g_a' = g\,\Delta\rho_a/\rho_0 = g\,\delta\rho``, and the gravity-wave
    speed used by the CFL monitor is ``c = \sqrt{g\,\delta\rho\,D}``. Whenever a
    paper term has ``\Delta\rho_a/\rho_0`` the code has the single field
    `m.drho`; whenever the paper has ``g_a'`` the code has `m.g * m.drho`.

## Notation

| symbol | code | meaning | units |
|--------|------|---------|-------|
| ``D`` | `m.D` | layer thickness | m |
| ``U, V`` | `m.U`, `m.V` | depth-averaged velocity (C-grid faces) | m s⁻¹ |
| ``T, S`` | `m.T`, `m.S` | depth-averaged temperature / salinity | °C / psu |
| ``\dot m`` | `m.melt` | basal melt rate (``>0`` = melting) | m s⁻¹ |
| ``\dot e`` | `m.nentr` | net entrainment ``= \mathrm{entr} + \mathrm{ent2} - \mathrm{detr}`` | m s⁻¹ |
| ``T_a, S_a`` | `m.Ta`, `m.Sa` | ambient T/S at layer base ``z_b-D`` | °C / psu |
| ``T_b, S_b`` | `m.Tb`, `m.Sb` | ice–ocean interface (boundary) T/S | °C / psu |
| ``\Delta\rho_a`` | ``\rho_0\,```m.drho` | dimensional plume–ambient density anomaly | kg m⁻³ |
| ``\delta\rho`` | `m.drho` | dimensionless reduced density ``\Delta\rho_a/\rho_0`` | – |
| ``g_a'`` | `m.g * m.drho` | reduced gravity, Eq. (6) | m s⁻² |
| ``z_b`` | `m.zb` | ice-base depth (negative below sea level) | m |
| ``f`` | `m.f` | Coriolis parameter | s⁻¹ |
| ``g`` | `m.g` | gravitational acceleration | m s⁻² |
| ``C_d`` | `m.Cd` | quadratic drag coefficient (momentum) | – |
| ``C_d^{\text{top}}`` | `m.Cdtop` | drag coefficient (friction velocity) | – |
| ``A_h, K_h`` | `m.Ah`, `m.Kh` | horizontal viscosity / diffusivity | m² s⁻¹ |
| ``\gamma_T, \gamma_S`` | `m.gamT`, `m.gamS` | turbulent exchange velocities | m s⁻¹ |
| ``u_\star`` | `m.ustar` | friction velocity | m s⁻¹ |
| ``\lambda_1,\lambda_2,\lambda_3`` | `m.l1,m.l2,m.l3` | linear-liquidus coefficients | – |

## Prognostic equations (1–5)

The model evolves five depth-integrated 2-D fields. Each kernel in
`src/numerics.jl` expands the flux-form left-hand side
``\partial(D\,q)/\partial t = q\,\partial D/\partial t + D\,\partial q/\partial t``
and divides the momentum/tracer right-hand sides by the (face-interpolated)
thickness, so the prognostic variable stepped is ``q`` itself, not ``Dq``.

### (1) Layer thickness — continuity

```math
\frac{\partial D}{\partial t} + \nabla\!\cdot(D\mathbf{U}) = \dot m + \dot e
\tag{1}
```

Kernel `_step_thickness_kernel!`: ``D^{+} = D^{0} + (\mathrm{convD} + \dot m + \dot e)\,\Delta t``,
where `convD` ``= -\nabla\!\cdot(D\mathbf{U})`` (advection), `m.melt` ``=\dot m``,
`m.nentr` ``=\dot e``.

### (2) ``U``-momentum

```math
\frac{\partial DU}{\partial t} + \nabla\!\cdot(D\mathbf{U}U) - fDV =
-\frac{gD^2}{2\rho_0}\frac{\partial \Delta\rho_a}{\partial x}
+ g_a'D\,\frac{\partial (z_b - D)}{\partial x}
- C_d\,|\mathbf{U}|\,U
+ \nabla\!\cdot(A_h D\nabla U)
\tag{2}
```

Kernel `_step_u_momentum_kernel!` integrates the **expanded** form (per unit
``\bar D``); term by term, the right-hand side `rhs` is:

```math
\underbrace{-U\,\partial_t \bar D}_{\text{thickness coupling}}
\;+\; \underbrace{\mathrm{cU}}_{-\nabla\cdot(D\mathbf{U}U)}
\;-\; \underbrace{g\,\overline{D\delta\rho}\,\frac{D_{x-1}-D}{\Delta x}}_{g_a'D\,\partial_x(z_b-D)\ \text{(depth part)}}
\;+\; \underbrace{g\,\overline{D\delta\rho\,\partial_x z_b}}_{\text{ice-base slope part}}
\;-\; \underbrace{\tfrac{1}{2}g\,\bar D^2\,\frac{\partial \delta\rho}{\partial x}}_{-\,gD^2/(2\rho_0)\,\partial_x\Delta\rho_a}
\;+\; \underbrace{f\,\overline{D V}}_{\text{Coriolis}}
\;-\; \underbrace{C_d\,U|\mathbf{U}|}_{\text{drag}}
\;+\; \underbrace{A_h\nabla^2 U}_{\text{diffusion}}
\;-\; \underbrace{\mathrm{detr}\cdot U}_{\text{detrainment loss}}
```

then ``U^{+} = U^{p} + (\mathrm{rhs}/\bar D)\,\Delta t``. Overbars denote C-grid
face interpolation to the ``U``-node. The detrainment-momentum-loss term is a
Laddie.jl addition (see [Entrainment](@ref)).

### (3) ``V``-momentum

```math
\frac{\partial DV}{\partial t} + \nabla\!\cdot(D\mathbf{U}V) + fDU =
-\frac{gD^2}{2\rho_0}\frac{\partial \Delta\rho_a}{\partial y}
+ g_a'D\,\frac{\partial (z_b - D)}{\partial y}
- C_d\,|\mathbf{U}|\,V
+ \nabla\!\cdot(A_h D\nabla V)
\tag{3}
```

Kernel `_step_v_momentum_kernel!`, structurally identical to (2) with
``x\to y`` and the Coriolis sign flipped (``-f\,\overline{D U}``).

### (4) Heat

```math
\frac{\partial DT}{\partial t} + \nabla\!\cdot(D\mathbf{U}T) =
\dot e\,T_a + \dot m\,T_b - \gamma_T (T - T_b) + \nabla\!\cdot(K_h D\nabla T)
\tag{4}
```

Kernel `_step_temperature_kernel!` (and `mat_*` variants for array-valued
``\gamma_T``/`conv2`). The implemented `rhs` adds one Laddie.jl term beyond the
paper — the convective relaxation ``-(T^{p}-T_a)\,\mathrm{conv2}`` (zero unless
`convpar = RelaxToAmbient`, see [Convective relaxation](@ref)):

```math
\mathrm{rhs} = -T\,\partial_t D + \mathrm{cT} + \dot e\,T_a + \dot m\,T_b
- \gamma_T(T - T_b) + K_h\nabla^2 T - (T^{p}-T_a)\,\mathrm{conv2},
```

with `cT` ``=-\nabla\!\cdot(D\mathbf{U}T)``, then ``T^{+} = T^{p} + (\mathrm{rhs}/D)\,\Delta t``.

### (5) Salt

```math
\frac{\partial DS}{\partial t} + \nabla\!\cdot(D\mathbf{U}S) =
\dot e\,S_a + \nabla\!\cdot(K_h D\nabla S)
\tag{5}
```

Kernel `_step_salinity_kernel!`. Identical structure to (4) but with **no**
ice–ocean exchange term (ice is salt-free), plus the same optional `conv2`
relaxation toward ``S_a``.

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

Equation (10) is the linear liquidus; `update_freezing_temperature!` also uses
it for the plume freezing point ``T_f = \lambda_1 S + \lambda_2 + \lambda_3 z_b``.
Define the **effective latent heat** ``L_\text{eff} = L - c_i T_i``. Eliminating
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

### (11–12) Turbulent exchange coefficients

```math
\gamma_T = \frac{u_\star}{2.12\,\ln(u_\star D/\nu_0) + 12.5\,\mathrm{Pr}^{2/3} - 8.68} \tag{11}
```
```math
\gamma_S = \frac{u_\star}{2.12\,\ln(u_\star D/\nu_0) + 12.5\,\mathrm{Sc}^{2/3} - 8.68} \tag{12}
```

Computed by `_compute_turbulent_transfer_coefficients!` for
`meltpar = TurbulentGamT` (the log argument carries a `+1e-12` floor for
``D\to 0``). For `meltpar = FixedGamT`, ``\gamma_T`` is a prescribed constant
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
D^2 g_b'\,\dot m + D^2 g_a'\,\dot e = \mu\,u_\star^3, \tag{14}
```

where ``g_b' = g\,\delta\rho_b`` uses the **plume–interface** density contrast
``\delta\rho_b = \beta(S - S_b) - \alpha(T - T_b)`` (field `m.drhob`). The melt
and detrainment term is the same in every variant below; with
``\delta\rho^{+} = \max(10^{-4}, \delta\rho)`` and
``\mathrm{entr} = \max(\dot e,0)``,
``\mathrm{detr} = \min(\mathrm{maxdetr}, \max(-\dot e,0))``.

Laddie.jl exposes **two** buoyancy-flux schemes that differ *only* in the
production term — pick via `entpar`:

`LambertEntrainment` **(default)** — `_lambert_entrainment_kernel!`, the form the
reference Python LADDIE actually integrates and which the verification test
reproduces bit-for-bit:

```math
\dot e
= \frac{2\mu}{g}\,\frac{u_\star^3}{D\,\delta\rho^{+}}
\;-\; \frac{\delta\rho_b}{\delta\rho^{+}}\,\dot m
\qquad (\text{single } D,\ \text{prefactor } 2\mu).
```

`GasparEntrainment` — `_gaspar_entrainment_kernel!`, the **literal** solution of
Eq. (14) for ``\dot e`` (``\dot e = \mu u_\star^3/(D^2 g_a') - (g_b'/g_a')\dot m``,
with ``g_a' = g\,\delta\rho``):

```math
\dot e
= \frac{\mu}{g}\,\frac{u_\star^3}{D^2\,\delta\rho^{+}}
\;-\; \frac{\delta\rho_b}{\delta\rho^{+}}\,\dot m
\qquad (D^2,\ \text{prefactor } \mu).
```

!!! note "Why two schemes"
    The melt/detrainment term ``-(\delta\rho_b/\delta\rho)\dot m`` matches Eq. (14)
    exactly in both (``g_b'/g_a' = \delta\rho_b/\delta\rho``). The **production**
    terms differ: a literal reading of Eq. (14) gives ``\mu u_\star^3/(D^2 g_a')``
    (`GasparEntrainment`), whereas the reference LADDIE code integrates
    ``2\mu u_\star^3/(D\,g_a')`` (`LambertEntrainment`). The default is
    `LambertEntrainment` so the Python verification stays valid; `GasparEntrainment`
    is provided for users who want the textbook Eq. (14).

**Alternative (`entpar = HollandEntrainment`)** — shear entrainment after
Holland & Jenkins (1999), *not* a Lambert et al. equation:

```math
\mathrm{entr} = \frac{c_l K_h}{A_h^2}\,
\sqrt{\max\!\Big(0,\;|\mathbf{u}|^2 - g_a'\,\frac{K_h}{A_h}\,D\Big)},
\qquad \mathrm{detr} = 0,
```

kernel `_holland_entrainment_kernel!`.

## Laddie.jl extensions (not in the paper)

These terms are zero/identity in the configuration that reproduces the paper, so
the Python-LADDIE verification stays valid.

### Minimum-thickness correction

`update_entrainment!` adds a non-negative `ent2` so ``D`` cannot fall below
`minD`, and assembles the net entrainment used in Eq. (1):

```math
\mathrm{ent2} = \max\!\Big(0,\;\frac{\mathrm{minD} - D^{\text{past}}}{2\Delta t}
- (\mathrm{convD} + \dot m + \mathrm{entr} - \mathrm{detr})\Big),
\qquad
\dot e = \mathrm{entr} + \mathrm{ent2} - \mathrm{detr}.
```

### Convective relaxation

For `convpar = RelaxToAmbient`, unstable cells (``\delta\rho < 0``) relax T/S
toward ambient over `convtime` via the `conv2` field appearing in (4)–(5):

```math
\mathrm{conv2} = \frac{[\delta\rho < 0]\,D}{\mathrm{convtime}}.
```

`ClampDensity` (clamp ``\delta\rho`` to a positive floor) and `ResetToAmbient`
(reset unstable-cell T/S to ambient) set `conv2 = 0` and act directly on the
fields instead.

## Equation → code map

| eq | quantity | source of truth |
|----|----------|-----------------|
| (1) | thickness | `_step_thickness_kernel!` (`numerics.jl`) |
| (2) | ``U``-momentum | `_step_u_momentum_kernel!`; terms: `u_*` fns (`physics.jl`) |
| (3) | ``V``-momentum | `_step_v_momentum_kernel!`; terms: `v_*` fns (`physics.jl`) |
| (4) | heat | `_step_temperature_kernel!` + `mat_*` variants |
| (5) | salt | `_step_salinity_kernel!` + `mat_*` variant |
| (6)/(7) | reduced gravity / EOS | `update_density!`, `_density_kernel!` |
| (8)–(10) | three-eq melt + liquidus | `_three_eq_melt_kernel!`, `update_freezing_temperature!` |
| (11)/(12) | ``\gamma_T,\gamma_S`` | `_compute_turbulent_transfer_coefficients!` |
| (13) | ``u_\star`` | `_ustar_kernel!` |
| (14) | entrainment | `LambertEntrainment` → `_lambert_entrainment_kernel!` (default); `GasparEntrainment` → `_gaspar_entrainment_kernel!` (literal Eq. 14); Holland → `_holland_entrainment_kernel!` |
| — | min-thickness / net ``\dot e`` | `update_entrainment!` |
| — | convective relaxation | `update_convection!`, `_update_conv2!` |
