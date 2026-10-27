# Coriolis parameter

At ice-shelf scale, rotation steers the meltwater into boundary currents rather than
letting it flow straight up-slope — visible in the flow-speed plot of the
[ISOMIP+ warm run](../generated/isomip.md#Warm-run).

The Coriolis parameter ``f`` enters the momentum equations only as the scalar in
``-f\,\hat{\mathbf{k}} \times \mathbf{u}``, which does not change when the horizontal
axes are rotated. So the orientation of the grid relative to true north never matters;
only the latitude does, through ``f = 2\Omega \sin\varphi``. It is selected with
`Params(; coriolis = ...)`:

- [`CoriolisParameter0D`](@ref), the default: an f-plane with ``f`` given directly,
  ``-1.37\times10^{-4}`` s⁻¹ (the ISOMIP+ value, 69.95 °S) by default.
- [`CoriolisParameter2D`](@ref): ``f`` from latitude. A scalar latitude is still an
  f-plane, stated geographically; its default reproduces the f-plane default to the last
  bit. A full-domain latitude matrix, the size of the mask, gives a genuine β-plane, and
  `Model` crops it with the mask.

Across the Antarctic shelves ``f`` runs from ``-1.32\times10^{-4}`` s⁻¹ at 65 °S to
``-1.44\times10^{-4}`` at 80 °S, roughly a 9 % spread. The Rossby radius scales as
``1/f``, so a large domain feels it even though the range looks narrow.

## API

```@docs
AbstractCoriolisParameter
CoriolisParameter0D
CoriolisParameter2D
```
