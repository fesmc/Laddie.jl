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

## Equations

The continuous system is the same under both forms.  Melt ``\dot m`` and entrainment
``\dot e`` bring in water at rest, so they carry no momentum and only dilute ``U``
through ``H``.  Detrainment ``\dot d``, cap detrainment ``\dot d_\mathrm{cap}`` and frazil
formation ``\dot f`` remove water at the layer's own velocity.  Together with the quadratic
drag they form one rate ``s`` acting on ``U``:
```math
\begin{aligned}
&\frac{\partial H}{\partial t} + \nabla\cdot(H\mathbf{u})
  = \dot m + \dot e - \dot d - \dot d_\mathrm{cap} - \dot f, \\
&\frac{\partial (HU)}{\partial t} + \nabla\cdot(H\mathbf{u}U) - fHV
  = P_x + \nabla\cdot(A_h H\nabla U) - s\,U, \\
&\frac{\partial (HV)}{\partial t} + \nabla\cdot(H\mathbf{u}V) + fHU
  = P_y + \nabla\cdot(A_h H\nabla V) - s\,V, \\
&s = \dot d + \dot d_\mathrm{cap} + \dot f + C_d\,|\mathbf{u}|,
\end{aligned}
```
with the pressure gradient
``P_x = -\frac{gH^2}{2\rho_0}\partial_x\Delta\rho_a + g_a' H\,\partial_x(H_{ib} - H)``.
In the code, ``\dot e - \dot d`` is `nentr`.

In the leapfrog, ``n-1`` is the filtered past level.  Let ``R^n`` collect the advection,
Coriolis, pressure and viscosity terms at the present level, and let ``\bar H`` be the
face average of ``H``.  The advective form applies the product rule and takes every
term at level ``n``:
```math
U^{n+1} = U^{n-1} + \frac{2\Delta t}{\bar H^{n}}
  \Big[R^{n} - U^{n}\,\overline{\partial_t H}^{\,n} - s^{n}\,U^{n}\Big],
\qquad \partial_t H^{\,n} = \frac{H^{n+1} - H^{n-1}}{2\Delta t}.
```
The flux form steps the transport, and the sinks act on ``U^{n+1}``:
```math
\bar H_\ast^{\,n+1}\,U^{n+1} = \bar H^{\,n-1}\,U^{n-1} + 2\Delta t\,\big[R^{n} - s\,U^{n+1}\big]
\quad\Longrightarrow\quad
U^{n+1} = \frac{\bar H^{\,n-1}\,U^{n-1} + 2\Delta t\,R^{n}}{\bar H_\ast^{\,n+1} + 2\Delta t\,s},
```
where ``\bar H_\ast^{\,n+1}`` is the face average of ``\max(H^{n+1}, H^\ast)`` and ``H^\ast``
is the thickness before the clamp.  Under [`UpstreamMomentumAdvection`](@ref), the
advection in ``R^n`` contains the face's own outflow ``-o\,U^n``.  It is moved to the new
level as well:
```math
U^{n+1} = \frac{\bar H^{\,n-1}\,U^{n-1} + 2\Delta t\,\big(R^{n} + o\,U^{n}\big)}
               {\bar H_\ast^{\,n+1} + 2\Delta t\,(s + o)}.
```
Since ``R^n + o\,U^n`` holds only the inflow, ``U^{n+1}`` is a weighted mean of ``U^{n-1}``
and the momentum coming in.  Under [`PastClosures`](@ref), ``s`` and the ``u_\star`` in
``\dot m`` and ``\dot e`` are evaluated at ``n-1``.

The advective form takes ``\dot d`` at the tracer point ``(i, j)`` and does not average it
onto the face, as Python LADDIE v1.x does (`-detr * U` in `intU`/`intV`).  The term is
therefore shifted half a cell: a ``U`` face sees only the detrainment of the cell to its
west, and a ``V`` face only that of the cell to its south.  It is kept that way so that the default stays bit-identical to v1.x.  The flux
form averages ``\dot d`` onto the face, as it does ``\dot d_\mathrm{cap}`` and ``\dot f``
under both forms.

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
