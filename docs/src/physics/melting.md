# Melting

The melt rate ``\dot m`` (m s⁻¹ of freshwater, positive for melting) is computed on
ice-covered cells only; ice-free [gap](gaps.md) cells never melt. Five schemes are
available through `Params(; melting = ...)`:

| scheme | equations | transfer coefficients |
|---|---|---|
| [`FixedGamTMelting`](@ref) (default) | three-equation | ``\gamma_T`` constant, ``\gamma_S = \gamma_T/35`` |
| [`TurbulentGamTMelting`](@ref) | three-equation | log-layer functions of ``u_\star`` (Holland & Jenkins 1999) |
| [`UStarGamTMelting`](@ref) | three-equation | ``\gamma_T = \Gamma_T u_\star``, ``\gamma_S = \gamma_T/r`` |
| [`TwoEquationMelting`](@ref) | two-equation, experimental | ``\gamma_{TS} = \Gamma_{TS} u_\star`` |
| [`PrescribedMelting`](@ref) | none, ``\dot m`` given | equivalent ``\gamma_T`` diagnosed |

## Three-equation melt

Three constraints hold simultaneously at the ice–ocean interface (Jenkins 1991):

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
face, ``L_\text{eff} = L - c_i (T_i - T_b)`` is the effective latent heat adjusted for
the heat conducted into the ice (see [Ice temperature](@ref)), and the turbulent exchange
velocities ``\gamma_T, \gamma_S`` scale with the friction velocity

```math
u_\star = \sqrt{C_d^{\text{top}}\,(U^2 + V^2 + u_{\text{tide}}^2)}.
```

Eliminating ``T_b`` and ``S_b`` yields a **quadratic in ``\dot{m}``**, solved pointwise
at every grid cell:

```math
\dot{m}^2 + b\,\dot{m} + c = 0,
```

with coefficients (writing ``\tilde{T}_f = \lambda_2 + \lambda_3\,z_b`` for the
depth-dependent freezing temperature at zero salinity, and ``L_\text{eff} = L - c_i T_i``):

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

The three schemes that solve it differ only in ``\gamma_T`` and ``\gamma_S``:

- [`FixedGamTMelting`](@ref): ``\gamma_T`` is a prescribed constant and
  ``\gamma_S = \gamma_T / 35``. The ISOMIP+ default.
- [`TurbulentGamTMelting`](@ref): both are computed from ``u_\star`` via a log-layer
  formulation (Holland & Jenkins 1999; Lambert et al. 2023, Eqs. 11–12).
- [`UStarGamTMelting`](@ref): ``\gamma_T = \Gamma_T u_\star`` and ``\gamma_S = \gamma_T / r``
  with ``r`` = 35 by default. This is LADDIE v2's `'uniform'` option and what the ISOMIP+
  protocol tunes. A `ratio` other than 35 (e.g. 70) is experimental; McPhee et al. (2008)
  give 35–70 for a melting base.

Jenkins et al. (2010) tested these parameterizations against the thinning of the Ronne Ice
Shelf measured by radar, and give ``C_d^{1/2}\Gamma_T = 1.1\times10^{-3}`` and
``C_d^{1/2}\Gamma_S = 3.1\times10^{-5}`` for the three-equation form. Since ``\Gamma`` multiplies
the ``u_\star`` of this model, which uses `C_d_top`, divide by ``\sqrt{C_d^\text{top}}``: that gives
``\Gamma_T \approx 3.3\times10^{-2}`` for the default `C_d_top = 1.1e-3`, close to the
`UStarGamTMelting` default of ``3\times10^{-2}``.

## Two-equation melt

The same data are fit equally well by the **two-equation** form
([`TwoEquationMelting`](@ref); McPhee 1992), which drops the salt balance and sets the
interface at the freezing point of the layer salinity, ``T_b = T_f(S, z_b)``:

```math
c_p\,\gamma_{TS}\,(T - T_b) = \dot{m}\,L_\text{eff} + q_\text{cond}
\;\Rightarrow\;
\dot m = \frac{c_p\,\gamma_{TS}\,(T - T_f) - q_\text{cond}}{L - c_i\,(T_i - T_f)},
```

linear in ``\dot m``, with ``\gamma_{TS} = \Gamma_{TS}\,u_\star`` and
``C_d^{1/2}\Gamma_{TS} = 5.9\times10^{-4}`` (default ``\Gamma_{TS} = 0.0178``, for the default
`C_d_top`; change it along with `C_d_top`). The two forms agree when ``\gamma_S \to \infty``,
and differ under strong thermal forcing, where the interface salinity departs from the layer's.
Jenkins et al. expect the three-equation form to be the more general.

!!! warning "Experimental"
    `TwoEquationMelting` and a `UStarGamTMelting` ratio other than 35 are experimental and
    log a warning when constructed. The two-equation scheme leaves the interface salinity at
    the layer salinity, so the melt correction of the buoyancy-flux entrainment sees no
    salinity contrast, and `gamS` is not used.

## Prescribed melt

[`PrescribedMelting`](@ref) takes the melt rate as given, in m yr⁻¹ (the units of the
`melt` output), as a scalar or as a full-domain matrix that `Model` crops with the mask.
Negative values freeze. The melt still acts on the layer as it would in the
three-equation model: the interface is at the local freezing point of the layer
salinity, the layer loses the heat that melting requires (reported as an equivalent
``\gamma_T`` in `m.gamT`), and ``u_\star`` is computed as usual, since entrainment needs
it. This is the way to drive the layer circulation with an observed or modelled melt
pattern.

## Ice temperature

``T_i`` comes from the **ice forcing**, not the parameter set: [`PrescribedIceForcing`](@ref)
holds it (as `T_ice_base`) as a scalar or as a 2D field, so a temperate ice front can sit
next to a cold interior. Despite the field name, it is not the temperature at the ice
base: the base is at ``T_b``, the freezing point, and ``T_i`` is the temperature of the
ice above the thin thermal boundary layer at the base — the surface temperature for a
steady ice shelf, or the temperature 50–100 m above the base from an ice sheet model.
Colder ice absorbs more heat per unit melt — ``L_\text{eff}`` runs from
3.34e5 J kg⁻¹ at 0 °C to 3.84e5 at −25 °C — so it acts as a direct brake on melting.
How that heat enters the balance is the [ice-conduction](ice_conduction.md) closure.

## API

```@docs
AbstractMelting
FixedGamTMelting
TurbulentGamTMelting
UStarGamTMelting
TwoEquationMelting
PrescribedMelting
```
