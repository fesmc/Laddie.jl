# Ice forcing

The ice forcing supplies the far-field ice temperature `T_ice_farfield`: the temperature of the ice
above the basal boundary layer, typically the surface temperature. It enters the heat conducted into
the ice at the base, through the [ice-conduction](../physics/ice_conduction.md) scheme, and so the
[melting](../physics/melting.md) parameterisations.

A model takes it as the second half of its [`CavityForcing`](@ref). Passing an ocean forcing on its own
to `Model` pairs it with a uniform `PrescribedIceForcing(-25.0)`.

## API

```@docs
AbstractIceForcing
PrescribedIceForcing
```
