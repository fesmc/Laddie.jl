# Maximum layer thickness

The layer thickness is bounded from below by `Params.H_min`, through the
[minimum-thickness entrainment](../physics/entrainment.md). An upper bound is optional,
`Params(; max_layer_thickness = ...)`:

| option | cap on ``H`` | needs a bed |
|---|---|---|
| [`NoMaxLayerThickness`](@ref) (default) | none | no |
| [`TopographicMaxLayerThickness`](@ref) | the water column, ``z_b - z_\text{bed}`` | yes |
| [`AbsoluteMaxLayerThickness`](@ref) | a fixed `H_max` | no |
| [`RelativeMaxLayerThickness`](@ref) | ``\max(f\,(z_b - z_\text{bed}),\ H_\text{floor})`` | yes |

The topographic and relative caps bind only when a bed elevation `z_bed` was given to
[`Grid`](@ref); without one the bed is at ``-\infty`` and the cap never binds.

The cap is enforced before the thickness step, as LADDIE v2 does with its `Hmax`: where
the first guess of the stepped ``H`` exceeds it, the excess is detrained (`model.detr_cap`,
output as `:detr_cap`). The detrained water leaves at the layer's own ``T``, ``S`` and
velocity, so the cap thins the layer without changing its tracers or momentum. A clamp
after the step remains as a safety net; under the [flux tracer form](tracer_form.md), the
volume it removes also leaves at the layer's own tracers.

`H_floor` of `RelativeMaxLayerThickness` is for cells where the draft nearly meets the
bed. There, the cell-mean water column underestimates how thick a plume can be, and a cap
of a few metres leaves the layer almost no room between `H_min` and the cap, so the cap
detrainment and the `H_min` top-up have to pass all the water that neighbours advect into
the cell through a layer of about a metre.

Without a cap the layer can exceed the local water column where its volume budget takes
it — as it does in the published Crosson–Dotson run, in 10 % of the shelf cells. That is
a known limitation of a depth-averaged model.

!!! warning "A cap still removes water, and that can cost more than it looks"
    Until 2026-10 the cap was purely corrective: it clamped `H` *after* the step and
    kept the clipped volume's tracer content. Wherever it bound, T and S then
    drifted towards the incoming meltwater and entrainment without dilution, which
    is what pinned pan-Antarctic runs at the `T_min` and `S_max` bounds. Capping a
    [Crosson–Dotson](../generated/crosson-dotson.md) run at the water column also
    collapsed the mean melt rate from 9.9 to 1.1 m yr⁻¹, including in cells where the
    cap never binds. Both findings predate the detrainment form and have not been
    re-measured with it.

    LADDIE v1 has no upper bound at all (its `maxD = 3000 m` never binds),
    and LADDIE v2's Antarctic setup uses a fixed `Hmax` of 10 000 m that never binds
    either. Prefer the default, and reach for a cap only as a stability measure.

## API

```@docs
AbstractMaxLayerThickness
NoMaxLayerThickness
TopographicMaxLayerThickness
AbsoluteMaxLayerThickness
RelativeMaxLayerThickness
```
