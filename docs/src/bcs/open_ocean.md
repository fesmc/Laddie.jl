# Open-ocean boundary

At the ice front, and at the edges of gaps under [`SinkGapsBC`](@ref), the layer meets
the open ocean. Layer properties leave freely; what enters is set by
`BoundaryConditions(; open_ocean = ...)`:

- [`ZeroGradientOpenOceanBC`](@ref), LADDIE v1 and v2: all fields are extrapolated with zero
  gradient, so inflow from the ambient ocean carries the properties of the cell it enters.
- [`NoInflowOpenOceanBC`](@ref): where the flow enters through an ice-front face, the thickness and
  tracer fluxes carry the (zero) value stored in the open-ocean cell, so no volume, heat
  or salt is advected in. The velocities themselves are not clipped, and the momentum
  advection is unaffected.
- [`OutflowOnlyOpenOceanBC`](@ref): a one-way valve. After each momentum step the new face-normal
  velocity at every ice-front face is clipped to point out of the cavity, so the layer
  can leave but never enter; the fluxes carry nothing in, as under `NoInflowOpenOceanBC`, while an
  inward start fades from the filtered level. Under the other two, an
  inflow face either recirculates the front cell's own water (thickening and cooling the
  front row) or keeps an inward velocity that carries nothing (thinning it). Under the valve
  the layer is fed only by entrainment and melt; ambient water that really enters a cavity
  does so below the plume.

The valve is the default, with [`ZeroGradientFrontPressure`](@ref). On a warm ISOMIP+ cavity
at 4 km over 20 days, against zero-gradient inflow with [`FullDepthFrontPressure`](@ref):

| front pressure + open ocean | mean melt (m/yr) | total (Gt/yr) | max speed (m/s) | front/interior melt | H front (m) |
|---|---|---|---|---|---|
| `FullDepthFrontPressure` + `ZeroGradientOpenOceanBC` (former default) | 1.59 | 56.41 | 0.538 | 1.87 | 6.3 |
| `ZeroGradientFrontPressure` + `OutflowOnlyOpenOceanBC` (default) | 1.58 | 56.21 | 0.340 | 0.93 | 11.1 |
| `TruncatedDepthFrontPressure` + `OutflowOnlyOpenOceanBC` | 1.59 | 56.32 | 0.429 | 1.27 | 8.4 |

The front row is the last ice cell of each row, the interior three cells upstream.

The pressure gradient across the same faces is a separate choice; see
[Ice-front pressure gradient](front_pressure.md).

## API

```@docs
AbstractOpenOceanBC
ZeroGradientOpenOceanBC
NoInflowOpenOceanBC
OutflowOnlyOpenOceanBC
```
