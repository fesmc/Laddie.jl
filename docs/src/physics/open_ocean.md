# Open-ocean boundary

At the ice front, and at the edges of gaps under [`SinkGapsBC`](@ref), the layer meets
the open ocean. Layer properties leave freely; what enters is set by
`BoundaryConditions(; open_ocean = ...)`:

- [`ZeroGradientInflow`](@ref), the default: all fields are extrapolated with zero
  gradient, so inflow from the ambient ocean carries the properties of the cell it enters.
- [`NoInflow`](@ref): where the flow enters through an ice-front face, the thickness and
  tracer fluxes carry the (zero) value stored in the open-ocean cell, so no volume, heat
  or salt is advected in. The velocities themselves are not clipped, and the momentum
  advection is unaffected.
- [`OutflowOnly`](@ref): a one-way valve. After each momentum step the new face-normal
  velocity at every ice-front face is clipped to point out of the cavity, so the layer
  can leave but never enter; the fluxes carry nothing in, as under `NoInflow`, while an
  inward start fades from the filtered level. Under the other two, an
  inflow face either recirculates the front cell's own water (thickening the front row)
  or keeps an inward velocity that carries nothing (thinning it).

The pressure gradient across the same faces is a separate choice; see
[Ice-front pressure gradient](front_pressure.md).

## API

```@docs
AbstractOpenOceanBC
ZeroGradientInflow
NoInflow
OutflowOnly
```
