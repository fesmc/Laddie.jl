# Entrainment

Turbulence at the base of the layer mixes warm ambient water *upward* into it. This
supplies the heat for melting and is a **positive feedback**: more shear and melt mean
more entrainment, more heat and more melt. Where the buoyancy supplied by melting
outweighs the turbulent production, the net rate turns negative and the layer detrains
instead. The scheme is selected with `Params(; entrainment = ...)`.

## Buoyancy-flux entrainment (Gaspar 1988)

[`GasparEntrainment`](@ref), the default, solves the mechanical-energy balance of the
layer (Gaspar 1988; Gladish et al. 2012; Lambert et al. 2023, Eq. 14)

```math
\frac{H}{2}\, g_b'\,\dot m + \frac{H}{2}\, g_a'\,\dot e = \mu\,u_\star^3
```

for the net rate ``\dot e``:

```math
\dot e
= \frac{2\mu}{g}\,\frac{u_\star^3}{H\,\delta\rho^{+}}
\;-\; \frac{\delta\rho_b}{\delta\rho^{+}}\,\dot m,
\qquad \delta\rho^{+} = \max(\mathrm{drho\_floor}, \delta\rho).
```

Here ``g_a' = g\,\delta\rho`` is the reduced gravity against the ambient water and
``g_b' = g\,\delta\rho_b`` that against the interface water, at ``T_b`` and ``S_b``. The
positive part is entrainment, the negative part detrainment, bounded by
`Params.max_detrainment` (unbounded by default). This is the reference LADDIE v1
`'Gaspar'` and v2 `'Gaspar1988'` scheme, verified bit-for-bit against the Python
reference.

## Shear entrainment (Holland & Jenkins 1999)

[`HollandEntrainment`](@ref) is driven by the shear alone, with no detrainment:

```math
\dot e = \frac{c_l K_h}{A_h^2}\,
\sqrt{\max\!\Big(0,\;|\mathbf{u}|^2 - g\,\delta\rho\,\frac{K_h}{A_h}\,H\Big)}.
```

``A_h`` and ``K_h`` are `Params.A_h` and `Params.K_h` under either
[lateral viscosity](lateral_viscosity.md) scheme.

## Minimum thickness

Both schemes are topped up by a non-negative extra entrainment `ent2` that keeps the
layer at least `Params.H_min` thick (1 m by default), sized so that the thickness step
would land on `H_min`:

```math
\mathrm{ent2} = \max\!\Big(0,\;\frac{H_\text{min} - H^{\text{past}}}{2\Delta t}
- (\mathrm{convH} + \dot m + \mathrm{entr} - \mathrm{detr})\Big),
\qquad
\dot e = \mathrm{entr} + \mathrm{ent2} - \mathrm{detr}.
```

The ambient water it brings in is at ``T_a, S_a``, like the rest of the entrainment.
A [maximum layer thickness](../numerics/max_layer_thickness.md), if one is set, is the
mirror image: a cap detrainment at the layer's own properties.

## API

```@docs
AbstractEntrainment
GasparEntrainment
HollandEntrainment
```
