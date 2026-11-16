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

The pressure gradient across the same faces is a separate choice; see
[Ice-front pressure gradient](front_pressure.md).

## API

```@docs
AbstractOpenOceanBC
ZeroGradientInflow
NoInflow
```
