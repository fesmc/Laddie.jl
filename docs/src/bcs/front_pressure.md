# Ice-front pressure gradient

The pressure-gradient force in the momentum equations has three parts: the ice-base
slope, the layer-thickness gradient and the density gradient (see the
[equation reference](../physics/equations.md)). The last two are the product-rule split of
``-\nabla P``, with ``P = \tfrac12 g\,\delta\rho\,H^2`` the depth-integrated reduced pressure of
the layer, and difference a cell against its neighbour. At a **one-sided face** — the ice
front, or a [`SinkGapsBC`](@ref) gap edge — that neighbour is not part of the active domain:
what it holds is a condition on ``P`` beyond the front, which `Params(; front_pressure = ...)`
selects. With the front cell's ``H`` and ``\delta\rho``:

| option | thickness / density term | outward tendency of ``HU`` at the face | condition beyond the front | matches |
|---|---|---|---|---|
| [`ZeroGradientFrontPressure`](@ref) (default) | dropped / dropped | 0 | ``\partial P/\partial n = 0`` | [LADDIE v2 description paper](@cite lambert_one-layer_2026) |
| [`ZeroExteriorFrontPressure`](@ref) | dropped / kept | ``\tfrac12 g\,\delta\rho\,H^2/\Delta x`` | ``P = 0`` one cell out | LADDIE v2 code |
| [`V1FrontPressure`](@ref) | kept / kept | ``\tfrac32 g\,\delta\rho\,H^2/\Delta x`` | ``P = 0`` one cell out, counted 3× | LADDIE v1 |

- `ZeroExteriorFrontPressure` is the one-sided difference of ``P`` against ``H = \delta\rho = 0``
  in the neighbour, the values masking stores there. The LADDIE v2 code drops the thickness
  term at calving-front faces and keeps the density term, which gives the same force.
- `V1FrontPressure` keeps both terms. The face averages of the split collapse to the front
  cell's values, so it sums to 3× the one-sided difference of ``P``: the thickness term alone
  is ~150× its interior value on a warm ISOMIP+ run. It reproduces LADDIE v1, which Laddie.jl
  is ported from and validated against.
- `ZeroGradientFrontPressure` drops both, as the zero lateral gradient at the calving front
  stated by [Lambert et al. (2026)](@cite lambert_one-layer_2026). It is the default, with
  [`OutflowOnlyOpenOceanBC`](@ref); see [Open-ocean boundary](open_ocean.md) for the comparison.

## Why zero gradient

Beyond the front, ``P`` decays over some length ``L``: a Robin condition
``\partial P/\partial n = -P/L``, of which `ZeroExteriorFrontPressure` is ``L = \Delta x`` and
`ZeroGradientFrontPressure` is ``L \to \infty``. Observed outflows leave the front as buoyant,
geostrophic currents (Dotson) and keep rising beyond it (Pine Island), so ``L`` is of the order
of the Rossby radius (≈ 10 km), not ``\Delta x``. At ``\Delta x`` = 1–4 km zero gradient is the
closer limit, and the only one whose front-row force does not grow as ``1/\Delta x`` under
refinement.

None of the options affects the grounding line, where no momentum equation is solved.

## API

```@docs
AbstractFrontPressure
ZeroGradientFrontPressure
ZeroExteriorFrontPressure
V1FrontPressure
```
