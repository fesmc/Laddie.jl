# Convection

A one-layer model has no vertical structure to overturn. Where the layer becomes denser
than the ambient water beneath it, ``\delta\rho < 0``, it is convectively unstable and
would in reality mix away; the model has to say what happens instead. The treatment is
selected with `Params(; convection_scheme = ...)`:

| scheme | action | where |
|---|---|---|
| [`ResetToAmbient`](@ref) (default) | ``T, S`` reset to ambient at once, with ``S`` lowered just enough for a contrast of `d_rho_min` | ice-covered cells |
| [`ClampDensity`](@ref) | ``\delta\rho`` floored at `d_rho_min`, ``T`` and ``S`` left alone | the whole active domain |
| [`RelaxToAmbient`](@ref) | ``T, S`` relaxed toward ambient over `convection_time` | ice-covered cells |

The thresholds `d_rho_min` are dimensional contrasts ``\rho_a - \rho`` in kg m⁻³, divided
by `Params.rho0_seawater` to compare with the reduced contrast ``\delta\rho``: the default
0.005 kg m⁻³ is ``\delta\rho \approx 4.9\times10^{-6}``, the buoyancy floor of LADDIE v2.
`ClampDensity` and `ResetToAmbient` act wherever ``\delta\rho`` is below that threshold,
so on marginally stable cells as well as unstable ones. Only the unstable cells,
``\delta\rho < 0``, are flagged in `model.convection`. The salinity offset of the reset
makes the contrast exactly the threshold under any [equation of state](equation_of_state.md),
because ``T = T_a`` removes the thermal term.

[`RelaxToAmbient`](@ref) acts through the tracer equations, as a sink

```math
-(q^{p} - q_a)\,\mathrm{conv2}, \qquad
\mathrm{conv2} = \frac{[\delta\rho < 0]\,H}{\mathrm{convection\_time}},
```

evaluated at the past leapfrog level (see the [equation reference](equations.md)). The two
others act on the fields directly, once at the start of each step, before the melt and
entrainment are computed, and again after the Robert–Asselin filter.

## Convection in gaps

The ambient profile in an ice-free [gap](gaps.md) is sampled at the sea surface, the
coldest and freshest water in the column, so the layer there reads as convectively
unstable almost unconditionally. [`ResetToAmbient`](@ref) and [`RelaxToAmbient`](@ref) are
therefore restricted to ice-covered cells: resetting a gap cell towards surface ambient
would erase the temperature and salinity anomaly the layer is carrying and quietly rebuild
the meltwater sink that [`ConnectedGapsBC`](@ref) exists to remove. [`ClampDensity`](@ref) is
*not* restricted — a buoyancy floor keeps the layer lighter than ambient without
overwriting its tracers, and it is the one convection treatment the reference
implementation also applies over its whole active domain.

## API

```@docs
AbstractConvectionScheme
ResetToAmbient
ClampDensity
RelaxToAmbient
```
