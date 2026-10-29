# Momentum form and closure level

The momentum budget is in flux form, ``\partial(HU)/\partial t = -\nabla\cdot(HU\mathbf{u}) + \dots``,
and can be stepped in two forms, `Params(; momentum_form = ...)`, the momentum counterpart of
the [tracer form](tracer_form.md):

- [`AdvectiveMomentumForm`](@ref), the default, steps ``U`` itself with the product-rule term
  ``-U\,\partial_t H`` and every term at the present level, as Python LADDIE v1.x does:
  ```math
  U^{+} = U^{-} + 2\Delta t\,\frac{\mathrm{rhs} - U\,\partial_t H - s\,U}{\bar H^{n}}.
  ```
- [`FluxMomentumForm`](@ref) steps the transport ``\bar H U`` and divides by the new face
  thickness, as LADDIE v2 does, with the terms that remove momentum at the layer's own
  ``U`` taken at the new level:
  ```math
  U^{+} = \frac{\bar H^{-} U^{-} + 2\Delta t\,\mathrm{rhs}}{\bar H^{+} + 2\Delta t\, s},
  \qquad s = d + d_\mathrm{cap} + f + C_d\,|\mathbf{u}|,
  ```
  with ``d`` the detrainment, ``d_\mathrm{cap}`` the cap detrainment and ``f`` the frazil formation,
  averaged onto the face.

Separately, `Params(; closure_level = ...)` sets the time level the closures read:
the friction velocity ``u_\star`` (hence melt and entrainment), the thickness in the
entrainment rate and the drag speed ``|\mathbf{u}|``, either [`PresentClosures`](@ref)
(the default) or [`PastClosures`](@ref).

## Why

A term that damps the state it is computed from is unstable in the leapfrog at the centred
level once its rate ``\lambda`` is large: with the Robert–Asselin filter at ``\nu = 0.8``, the
limit is ``\lambda\Delta t \approx 0.57``, and beyond it the computational mode flips the
state every step.  Entrainment of water at rest dilutes ``U`` at the rate ``e/H``, and since
Gaspar's rate grows with ``u_\star^3 \propto |U|^3`` the linearised rate is ``4e/H``.  In thin,
strongly entraining layers it exceeds the limit (``\lambda\Delta t \approx 0.6`` in the wall
corners of the 2 km pan-Antarctic domain at ``\Delta t = 413`` s); quadratic drag, at
``2C_d|\mathbf{u}|/H``, follows at ``H \approx 1`` m.  The primal stays bounded, held by the
floors and the speed cap, but the alternation biases the mean entrainment and melt of the
cell and its tangent grows without bound.

The two switches remove the two parts of that rate.  Under [`FluxMomentumForm`](@ref) the
dilution acts through ``\bar H^{+}``, the factor ``\bar H^{-}/\bar H^{+} \le 1``, and drag and
detrainment through the denominator: unconditionally stable for a frozen rate.  Under
[`PastClosures`](@ref) the dependence of the rates on the state, ``e(u_\star, H)`` and
``|\mathbf{u}|``, comes from the lagged level, as Python LADDIE v1.x lags the diffusion and
the convective relaxation.  For the scalar model ``y' = -\lambda y`` with the rate split as in
``e(U)\,U``, the stability limit moves from ``\lambda\Delta t = 0.57`` to ``1.15`` under the flux form
alone, and beyond ``20`` with both.

The same holds for advection: under [`UpstreamMomentumAdvection`](@ref) the momentum that
leaves a face's control volume, ``M_\mathrm{out}\,U``, is divided by ``\bar H``, and a thin face
next to thick donors loses more than it holds in one step.  The flux form takes that outflow
at the new level too, as [`FluxTracerForm`](@ref) does for the tracers, so ``U^{+}`` is a weighted
mean of ``U^{-}`` and the momentum coming in.

In the 2 km pan-Antarctic domain at ``\Delta t = 413`` s, the flux form removes the alternation
at both wall corners where it occurred, one from each mechanism, and the cells settle where
a run at ``\Delta t/2`` puts them.

Both defaults reproduce the earlier integration bit for bit.  On warm ISOMIP+ all four
combinations give the same mean melt to 0.1 %.

## API

```@docs
AbstractMomentumForm
AdvectiveMomentumForm
FluxMomentumForm
AbstractClosureLevel
PresentClosures
PastClosures
```
