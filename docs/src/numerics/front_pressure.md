# Ice-front pressure gradient

The pressure-gradient force in the momentum equations has three parts: the ice-base
slope, the layer-thickness gradient and the density gradient (see the
[equation reference](../physics/equations.md)). The last two difference a cell against
its neighbour. At a **one-sided face** — the ice front, or a [`SinkGapsBC`](@ref)
gap edge — that neighbour is not part of the active domain, and masking pins its stored
``H`` and ``\delta\rho`` to `0`. The thickness-gradient term
``g\,\overline{H\delta\rho}\,(H_{x-1}-H)/\Delta x`` is then a full one-sided gradient, as
if ``H`` collapsed to zero across one grid cell.

The two reference implementations disagree on what to do about this, so
`Params(; front_pressure = ...)` makes it a choice:

| option | thickness gradient | density gradient | matches |
|---|---|---|---|
| [`FullDepthGradient`](@ref) (default) | kept | kept | Python LADDIE v1.x |
| [`TruncatedDepthGradient`](@ref) | dropped | kept | LADDIE v2 Fortran |
| [`ZeroGradientFront`](@ref) | dropped | dropped | LADDIE v2 description paper |

- `FullDepthGradient` reproduces Python LADDIE v1.x, which Laddie.jl is ported from and
  validated against. Measured on a warm ISOMIP+ run, the thickness-gradient term is
  ~150× larger at the ice front than in the interior under it.
- `TruncatedDepthGradient` drops the thickness-gradient term, as the LADDIE v2 Fortran
  does at calving-front faces.
- `ZeroGradientFront` also drops the density-gradient term
  ``\tfrac12 g\,\bar H^2\,(\delta\rho_{x-1}-\delta\rho)/\Delta x``. This is the zero
  lateral gradient at the calving front stated in the LADDIE v2 description paper
  (Lambert et al. 2026); the v2 Fortran keeps the density term.

None of them affects the grounding line, where no momentum equation is solved.

## API

```@docs
AbstractFrontPressure
FullDepthGradient
TruncatedDepthGradient
ZeroGradientFront
```
