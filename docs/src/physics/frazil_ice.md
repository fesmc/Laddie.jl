# Frazil ice

Where the layer is supercooled, ``T < T_f(S, z_b)``, ice crystals grow in the water
column. The three-equation melt can already freeze directly at the interface, but in real
Ice Shelf Water plumes most of the supercooling is released by frazil instead (Jenkins &
Bombosch 1995; [Smedsrud & Jenkins 2004](@cite smedsrud_frazil_2004)). The treatment is selected with
`Params(; frazil_ice = ...)`:

- [`NoFrazilIce`](@ref), the default and the behaviour of [LADDIE v1 and v2](../versions.md): supercooling
  is released only by basal freezing in the melt parameterization.
- [`RelaxedFrazilIce`](@ref): the supercooling relaxes toward zero over a time ``\tau`` by
  forming frazil ice, which is deposited on the ice base at once.

## Relaxed frazil formation

Under the ice cover, the formation rate is

```math
\dot f = \frac{c_p}{L}\,\frac{H\,\max(T_f - T^{p}, 0)}{\tau},
\qquad T_f = \lambda_1 S^{p} + \lambda_2 + \lambda_3 z_b,
```

a freshwater-equivalent rate in m s⁻¹ like the melt, available as `model.frazil`. The
water that freezes leaves the layer at its own ``T`` and velocity, its salt stays behind,
and its latent heat ``L \dot f`` warms the rest. So ``T`` relaxes toward ``T_f`` at the
rate ``1/\tau``, ``S`` rises by brine rejection, and momentum per unit mass is unchanged.
The terms in the thickness, momentum and heat equations are given in the
[equation reference](equations.md).

``L`` is used, not the ``L_\text{eff}`` of the melt: the crystals form at the freezing
point inside the water, and cooling the deposited ice to the shelf temperature is a matter
for the ice, not the layer. Gap cells form no frazil; the ice they would form is sea ice.

The supercooling is read at the past leapfrog level, like the convective relaxation of
[`RelaxToAmbient`](@ref), so the step is stable for ``\tau > \Delta t`` and does not
overshoot ``T_f`` for ``\tau > 2\Delta t``.

The default ``\tau`` = 1 h is an order-of-magnitude choice; the literature has no
relaxation time for a depth-integrated plume. [Rees Jones & Wells (2018)](@cite rees_jones_frazil-ice_2018) find that frazil
adjusts to its forcing over a few hundred seconds in a 1 m layer, a time proportional to
the layer depth, which gives about an hour for a 10 m layer. [Smedsrud & Jenkins (2004)](@cite smedsrud_frazil_2004)
find a residual supercooling of ~0.02 °C that frazil growth leaves in place for up to a
day. Ocean models often remove the supercooling within one step (``\tau \to \Delta t``).

## API

```@docs
AbstractFrazilIce
NoFrazilIce
RelaxedFrazilIce
```
