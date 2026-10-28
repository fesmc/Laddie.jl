# Tracer form

The temperature and salinity budgets are in flux form,
``\partial(Hq)/\partial t = -\nabla\cdot(H\mathbf{u}q) + \dots``, and can be stepped in
two forms, `Params(; tracer_form = ...)`:

- [`FluxTracerForm`](@ref), the default, steps the contents ``HT`` and ``HS`` and
  divides by the new thickness, as [LADDIE v2](../versions.md) does:
  ```math
  (Hq)^{+} = H^{-} q^{-} + 2\Delta t\,\mathrm{rhs}, \qquad q^{+} = \frac{(Hq)^{+}}{H^{+}}.
  ```
- [`AdvectiveTracerForm`](@ref) steps ``T`` and ``S`` themselves, with the product-rule
  term ``-q\,\partial_t H`` taken from the thickness step just made, as LADDIE v1 does:
  ```math
  q^{+} = q^{-} + 2\Delta t\,\frac{\mathrm{rhs} - q\,\partial_t H}{H^{n}},
  ```
  with `rhs` the same right-hand side. Select it explicitly to reproduce LADDIE v1; the
  verification against the v1 code and the [Crosson–Dotson](../generated/crosson-dotson.md)
  example do.

The flux form is the default because of how the two differ. They differ when the discrete ``\partial_t H`` does not match the volume sources of the
same step, which happens wherever the thickness step is clamped or restarted. In the
advective form that mismatch is divided by ``H``, and in thin layers it can drive ``q``
far outside the range of the waters that make it up. In the flux form the sources enter
the tracer only in proportion to the volume they bring, so the mismatch cannot be
amplified. The advective form also carries a larger time-step error: on a 2-day warm
ISOMIP+ run, cutting the step from 210 s to 52.5 s moves the mean melt by 7.3 % under the
advective form and by 0.4 % under the flux form.

Where the clamp after the thickness step moves ``H^{+}`` off the stepped value ``H^{*}``,
the flux form gives the moved water its matching tracer, as the preventive bounds of LADDIE v2
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
