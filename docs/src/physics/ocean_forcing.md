# Ocean forcing

The ocean forcing supplies the ambient temperature ``T_a`` and salinity ``S_a`` that the
layer entrains and that set its buoyancy. They are sampled from vertical profiles at the
depth of the layer base, ``z_b - H``. A model is driven by a [`CavityForcing`](@ref),
which pairs an ocean forcing with an ice forcing; passing an ocean forcing on its own
to `Model` pairs it with the default `PrescribedIceForcing(-25.0)`.

| forcing | profiles | use |
|---|---|---|
| [`OceanForcing1D`](@ref) | one, for the whole domain | idealised setups and single cavities |
| [`PerBasinOceanForcing1D`](@ref) | one per basin, with a basin index for every cell | domains spanning several basins (e.g. pan-Antarctic) |
| [`Laddie.OceanForcing2D`](@ref) | one per grid cell | laterally varying forcing; not exported, not on the Reactant backend |

[`ISOMIPForcing`](@ref) builds the warm and cold ISOMIP+ profiles as an `OceanForcing1D`;
see the [ISOMIP+ example](../generated/isomip.md#Forcing) for plots. For ocean models
without ice-shelf cavities, [`front_ocean_forcing`](@ref) builds a
`PerBasinOceanForcing1D` with one profile per ice-front water column, shared by the cavity
cells behind it.

The basin map of `PerBasinOceanForcing1D` and the per-cell profiles of `OceanForcing2D`
cover the full domain including the border ring, like a 2D ice temperature, and `Model`
crops them with the grid. Every cell of the basin map, land and ocean included, must hold
a valid basin index.

## Vertical lookup

The profiles are stored on a uniform depth grid (1 m for profiles built with
`OceanForcing1D`). At each time step the local plume-base depth ``z_b - H`` indexes this
grid and linear interpolation supplies the ``T_a``, ``S_a`` values that enter the density,
entrainment and convection terms; depths beyond the profile take its end values.

The 1 m grid is the *storage*, not the profile's own resolution: `OceanForcing1D`
sorts, resamples and extrapolates flat from whatever samples it is handed. So the
levels a profile is built on are part of the forcing — handing it an analytic
function every metre and handing it the same function on a model's coarse ocean
levels are two different forcings wherever the profile is curved, such as across
a thermocline. Reproducing another model means sampling where it samples, not
only using its formula.

## API

```@docs
AbstractOceanForcing
OceanForcing1D
PerBasinOceanForcing1D
Laddie.OceanForcing2D
ISOMIPForcing
```
