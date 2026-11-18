# Tracer bounds

After every tracer step, the layer temperature and salinity can be clamped to a fixed range
inside the active domain, `Params(; tracer_bounds = ...)`:

| option | effect |
|---|---|
| [`TracerBounds`](@ref) | ``T \in [T_\min, T_\max]``, ``S \in [S_\min, S_\max]`` (defaults −5 … 5 °C, 32 … 36 psu) |
| [`NoTracerBounds`](@ref) (default) | none, as in LADDIE v2 |

The bounds are a stability safeguard, not physics. With the tracers pinned the melt rate
stays finite, so a diverging run is hidden from the blow-up check. Widen them for fresher
settings such as Greenland fjords; an infinite bound switches that side off. Under
[`SmoothKinks`](@ref) the corners of the clamp are smoothed over `ε_T` and `ε_S`
(see [Kinks](kinks.md)).

## API

```@docs
AbstractTracerBounds
NoTracerBounds
TracerBounds
```
