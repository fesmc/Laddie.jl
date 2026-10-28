# Laplacian weights

All lateral Laplacians — tracer diffusion and momentum viscosity — are weighted by the
layer thickness ``H``. The diffused field itself is always taken at the lagged level
`past`, as leapfrog stability requires (see [Time stepping](time_stepping.md)). Which
time level of ``H`` multiplies it is a choice, `Params(; laplacian_weights = ...)`:

- [`PresentLaplacianWeights`](@ref), the default: `H.present`.
- [`PastLaplacianWeights`](@ref): the lagged, filtered `H.past`, so the weight sits on
  the same time level as the diffused field. This is what LADDIE v1 does.

The difference is small. On the [Crosson–Dotson](../generated/crosson-dotson.md)
validation case the two options give the same mean melt to 0.01 % and differ locally by
at most 1.4 m in ``H``.

## API

```@docs
AbstractLaplacianWeights
PresentLaplacianWeights
PastLaplacianWeights
```
