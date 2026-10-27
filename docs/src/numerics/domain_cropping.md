# Domain cropping

A realistic mask is mostly land and open ocean, which the solver steps through for
nothing. [`Grid`](@ref) therefore crops its inputs by default, with
`Grid(...; domain_cropping = ...)`:

- [`MinRectangleDomainCropping`](@ref), the default: crop the mask, draft, bed,
  coordinates, and any full-domain field a model is given later (a 2D ice temperature,
  latitude, basin map or prescribed melt), to the smallest rectangle containing all
  active cells — floating shelf and gaps — expanded by `margin` cells (4 by default)
  on every side.
- [`NoDomainCropping`](@ref): keep the full input arrays.

`margin` must be at least 2: the stencils skip the outermost ring, so an active cell must
not border an ocean cell of that ring. A margin larger than 2 leaves some context around
the cavity, which mostly matters for plotting. The crop is clipped to the input array, so
the padding can end up asymmetric; the info message reports the margin achieved on each
side.

`multiple` rounds the cropped size up to a multiple of it, per axis, so that a grid
sharded over a device mesh ([`ReactantBackend`](@ref)) splits evenly.

## API

```@docs
AbstractDomainCropping
MinRectangleDomainCropping
NoDomainCropping
```
