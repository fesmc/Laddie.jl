# Tracer form

The temperature and salinity budgets are in flux form,
``\partial(Hq)/\partial t = -\nabla\cdot(H\mathbf{u}q) + \dots``, and can be stepped in
two forms, `Params(; tracer_form = ...)`:

- [`AdvectiveTracerForm`](@ref), the default, steps ``T`` and ``S`` themselves, with the
  product-rule term ``-q\,\partial_t H`` taken from the thickness step just made, as
  Python LADDIE v1.x does:
  ```math
  q^{+} = q^{-} + 2\Delta t\,\frac{\mathrm{rhs}}{H^{n}}.
  ```
- [`FluxTracerForm`](@ref) steps the contents ``HT`` and ``HS`` and divides by the new
  thickness, as LADDIE v2 does:
  ```math
  (Hq)^{+} = H^{-} q^{-} + 2\Delta t\,\mathrm{rhs}, \qquad q^{+} = \frac{(Hq)^{+}}{H^{+}},
  ```
  with `rhs` the same right-hand side without ``-q\,\partial_t H``.

The two differ when the discrete ``\partial_t H`` does not match the volume sources of the
same step, which happens wherever the thickness step is clamped or restarted. In the
advective form that mismatch is divided by ``H``, and in thin layers it can drive ``q``
far outside the range of the waters that make it up. In the flux form the sources enter
the tracer only in proportion to the volume they bring, so the mismatch cannot be
amplified.

Where the clamp after the thickness step moves ``H^{+}`` off the stepped value ``H^{*}``,
the flux form gives the moved water its matching tracer, as LADDIE v2's preventive bounds
do: volume cut at the upper bound leaves at the layer's own ``q``, volume added by the
`H_min` floor comes in at the ambient ``q_a``,

```math
q^{+} = \frac{(Hq)^{+} + \max(H^{+} - H^{*}, 0)\,q_a}{\max(H^{+}, H^{*})}.
```

The clamp binds mostly in the bootstrap step, where the minimum-thickness entrainment and
the cap detrainment, sized for a ``2\Delta t`` step, do only half their work.

## API

```@docs
AbstractTracerForm
AdvectiveTracerForm
FluxTracerForm
```
