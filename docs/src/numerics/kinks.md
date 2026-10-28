# Kinks

The model is piecewise smooth. Floors and caps (`max`, `min`, `clamp`), the upwind
choice of the donor side, the speed cap and `√` at zero all have a derivative that
jumps, or grows without bound, at one point. The derivative is exact everywhere else,
so a gradient from automatic differentiation is correct almost everywhere. Still,
gradient-based optimisers can stall on a kink, and over a long window an adjoint that
multiplies through such switches every step can grow much faster than the
finite-perturbation response does.

`Params(; kinks = ...)` chooses how they are treated:

- [`ExactKinks`](@ref), the default, keeps every kink. The model is bit-identical to
  one without a kink treatment.
- [`SmoothKinks`](@ref) replaces each kink by a ``C^1`` blend over a band of half-width
  ε, with one ε per physical quantity, and is exact outside the band. `max(x, 0)`
  becomes

  ```math
  \max{}_\varepsilon(x) = \begin{cases} 0, & x \le -\varepsilon \\ (x + \varepsilon)^2 / 4\varepsilon, & |x| < \varepsilon \\ x, & x \ge \varepsilon, \end{cases}
  ```

  and `max(a, b)`, `min(a, b)`, `clamp`, `|u|` and the upwind split
  ``u^\pm`` follow from it (``u^+ + u^- = u`` holds exactly, so the upwind fluxes
  stay conservative). `√max(x, 0)` becomes the quadratic
  ``(x + 3\varepsilon^2)^2 / 16\varepsilon^3`` below ``x = \varepsilon^2``, whose slope
  stays below ``1/2\varepsilon``. The step of the conductive ``c_i`` at the freezing
  point is blended with a cubic smoothstep.

An ε of zero keeps that group exact, so the groups can be switched on one at a time to
find the ones that matter:

```julia
Params(; kinks = SmoothKinks())                                           # all on
Params(; kinks = SmoothKinks(; H = 0.0, u = 0.0, uH = 0.0, w = 0.0,
                               drho = 0.0, T = 0.0, S = 0.0, log = 0.1))   # γ_T only
```

Kinks that the state sits on exactly, rather than crosses, are left exact. A blend
there would shift the result by about ε/4 in every step. This covers the
`max(0, ⋅)` guard of the cap detrainment, the clamp of `H` after the thickness step
and the share of the flux-form tracer update that follows it: the minimum-thickness
entrainment and the cap detrainment bring the stepped `H` exactly to its bounds.

On a 3-day ISOMIP+ run with the LADDIE v2 schemes, the default widths move the
mean melt by less than 0.1 %. In a cold setup where the `T_max` bound binds, the
`T` group moves it by about 1 %.

## API

```@docs
AbstractKinks
ExactKinks
SmoothKinks
```
