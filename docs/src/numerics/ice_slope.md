# Ice-base slope

The slope of the ice base, ``\partial_x z_b`` and ``\partial_y z_b``, drives the layer up
the draft through the pressure gradient. It is part of the model's geometry, not of the
[`Grid`](@ref), because it depends on which cells are active, and is computed when the
model is built, by `Model(grid; gradient = ...)`:

- [`JlGradient`](@ref), the default, is mask-aware. At each shelf cell it uses only
  shelf-cell neighbours: centred differences where both neighbours are shelf, one-sided
  at ice fronts and grounding lines, zero where neither is. This avoids spurious slopes
  of order 0.2 from the drafts stored at ocean cells (``z_b = 0``) and grounded cells
  (``z_b`` = bed), which have nothing to do with the ice base.
- [`PyGradient`](@ref) is the centred difference ``(z_{b,i+1} - z_{b,i-1})/(2\Delta x)``
  over all cells with no mask awareness, the stencil of Python LADDIE v1.1. Use it to
  compare against the Python reference.

## API

```@docs
AbstractIceSlopeGradient
JlGradient
PyGradient
```
