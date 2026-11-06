# Ice-shelf gaps

Where melting thins a shelf all the way through, an ice-free **gap** opens inside the
domain. There is no ice base there, so no melt — but what happens to the meltwater layer
arriving from upstream is genuinely uncertain, and the two defensible answers bracket a
large range of coupled ice-sheet response ([Jesse et al. (2026)](@cite jesse_ice-sheet_2026)).
Laddie.jl offers both, and a family in between, as
`BoundaryConditions(; gaps = ...)`, selected on the same input mask:

| | [`SinkGapsBC`](@ref) (default) | [`LeakyGapsBC`](@ref)`(α)` | [`ConnectedGapsBC`](@ref) |
|---|---|---|---|
| gap cell is | open ocean | dynamically active, ice-free | dynamically active, ice-free |
| ``H, T, S`` in the gap | not stepped | stepped, relaxed | stepped |
| gap edge behaves like | the calving front | the shelf interior | the shelf interior |
| heat and momentum arriving at the gap | leave the cavity | a fraction α continues | continue downstream |

Real gaps are full of sea ice, mélange and bergs, so neither end-member is right on its
own; running both is how the uncertainty is quantified, and [`LeakyGapsBC`](@ref) spans
the range between them with one parameter. The
[ice-shelf gaps example](../generated/jesse-gaps.md) runs both on the paper's own geometries
and compares them with [LADDIE v2](../versions.md).

Gap cells are those marked `4` in the mask. Mark them directly, or from a reference ice
footprint with [`MarkGapsPreprocess`](@ref) (see
[Mask preprocessing](../numerics/preprocessing.md)). The boundary condition only decides
what happens in the cells so marked. Under `SinkGapsBC` they are demoted to open ocean
when the model is built, so the gap edges are one-sided faces like the ice front, with the
same [open-ocean condition](../numerics/open_ocean.md) and
[front pressure gradient](../numerics/front_pressure.md).

## Connected gaps

A cell marked `4` is ice-free but *active* under `ConnectedGapsBC`: the grid's `tmask`
includes it, `imask` (ice-covered cells) does not, and its draft sits at the sea surface,
``z_b = 0``. Three consequences follow, and they are the whole of the physics change:

- **No melt.** ``\dot m = 0`` by mask, not by the three-equation solve.
- **No ice–ocean heat flux.** With ``\dot m = 0`` the three equations collapse to
  ``T_b = T`` exactly, so the ``\dot m\,T_b - \gamma_T (T - T_b)`` pair in the temperature
  equation vanishes identically — heat is advected across the gap rather than lost to an
  ice base that is not there.
- **Entrainment stays on.** It is a property of the layer's own shear and stratification,
  not of the ice above; its detrainment term, which scales with ``\dot m``, switches
  itself off.

The ambient profile in a gap is sampled at the sea surface, so the layer there reads as
convectively unstable almost unconditionally; see [Convection](convection.md) for how each
scheme handles that. Frazil ice does not form in gaps either.

## Leaky gaps

[`LeakyGapsBC`](@ref) keeps the gaps active as `ConnectedGapsBC` does, with everything in
the section above. In addition, the layer dissolves into the water around it while
it crosses a gap. With ``q`` standing for ``H``, ``T``, ``U`` and ``V`` and ``q_a`` for
``H_\mathrm{min}``, ``T_a``, ``0`` and ``0``, each gap cell is relaxed after its update by

```math
\frac{\partial q}{\partial t} = -r\,(q - q_a), \qquad
r = \kappa\,|\mathbf u|, \qquad \kappa = -\frac{\ln \alpha}{L},
```

so the anomaly ``q - q_a`` falls by ``\alpha^{s/L}`` along a path of length ``s``
through the gap, whatever the speed. ``L`` is the gap's crossing length: the hydraulic
diameter ``4A/P`` of each gap, or one length for all of them. The transmission
``\alpha`` is the fraction of the layer's thickness, heat and momentum anomaly that
survives one crossing. Salinity is not relaxed: the water detrained leaves at the
layer's own ``T`` and ``S``. The step is backward Euler, so it is stable for any ``r``.

The family is continuous at both ends. ``\alpha = 1`` is `ConnectedGapsBC` to the last
bit, and ``\alpha = 0`` is `SinkGapsBC` itself; the limit ``\alpha \to 0^+`` drains the
gap and reproduces the sink's downstream melt. The response in between is not linear.
On the ISOMIP+ channel of the test suite (a 12 × 24 km gap across the boundary
current, 20 days, downstream melt summed over the wall rows):

| | sink | 0.1 | 0.25 | 0.5 | 0.75 | **0.8** | 0.9 | connected |
|---|---|---|---|---|---|---|---|---|
| downstream melt (m yr⁻¹, summed) | 104.3 | 104.0 | 105.6 | 102.8 | 129.8 | **129.1** | 155.2 | 166.9 |

A boundary current that loses half its momentum at each crossing mostly stalls, so the
melt stays close to the sink up to ``\alpha \approx 0.5`` and recovers over the last
half of the range. On the two year-200 geometries of the
[ice-shelf gaps example](../generated/jesse-gaps.md) (total melt, days 20–30):

| geometry | sink | ``\alpha = 0.8`` | connected |
|---|---|---|---|
| small-gap (five gap cells at the grounding line) | 38.7 | 46.3 | 68.6 Gt yr⁻¹ |
| large-gap (northern margin open over 300 km) | 52.0 | 53.5 | 65.6 Gt yr⁻¹ |

These tables justify ``\alpha = 0.8`` as the default: it is past the flat part of the
response, so the gap lets a real share of the connected signal through, and it lands
between the end-members on all three geometries, at 40 % (channel), 25 % (small-gap)
and 11 % (large-gap) of the way from the sink to the connected melt. The leak is
weaker on long gaps, where a current running along the gap crosses many times its
crossing length ``L``.

## API

```@docs
AbstractGapsBC
SinkGapsBC
ConnectedGapsBC
LeakyGapsBC
```
