# Ice forcing

The ice forcing supplies the far-field ice temperature `T_ice_farfield`: the temperature of the ice
above the basal boundary layer, typically the surface temperature. It enters the heat conducted into
the ice at the base, through the [ice-conduction](../physics/ice_conduction.md) scheme, and so the
[melting](../physics/melting.md) parameterisations.

`T_ice_farfield` is the temperature of the ice away from the base, not at it. The base is at
the interface temperature ``T_b``, the freezing point, which the melt solve computes every
step; were ``T_i`` the basal temperature, ``T_i = T_b`` and it would have no effect. The ice
column is not resolved: the heat conducted into the ice is parameterised, and ``T_i`` is the
temperature above the thin thermal boundary layer at the base, into which the ice is advected
as it melts.

- For a steady ice column, as assumed by [`HollandJenkinsIceConduction`](@ref)
  ([Holland & Jenkins 1999](@cite holland_modeling_1999)) and the default
  [`MeltOnlyIceConduction`](@ref), it is the ice-shelf surface temperature; Wiskandt &
  Jourdain (2025) use the 1980–2019 mean snow-surface temperature.
- With an ice-sheet model at hand, the ice temperature 50–100 m above the base, as a 2D
  field, carries the transient and upstream-advected state of the ice instead.

Colder ice absorbs more heat per unit melt, so a lower `T_ice_farfield` gives less melt for
the same thermal driving: the effective latent heat runs from 3.34e5 J kg⁻¹ at 0 °C to
3.84e5 J kg⁻¹ at −25 °C, a 15 % spread.

`T_ice_farfield` is a scalar (uniform ice) or a matrix the size of `mask`, letting a colder
interior sit next to a temperate ice front; `Model` converts either to a field on the grid,
cropped with the mask.

A model takes it as the second half of its [`CavityForcing`](@ref). Passing an ocean forcing on its own
to `Model` pairs it with a uniform `PrescribedIceForcing(-25.0)`.

## API

```@docs
AbstractIceForcing
PrescribedIceForcing
```
