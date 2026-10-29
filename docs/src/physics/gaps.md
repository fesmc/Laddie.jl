# Ice-shelf gaps

Where melting thins a shelf all the way through, an ice-free **gap** opens inside the
domain. There is no ice base there, so no melt — but what happens to the meltwater layer
arriving from upstream is genuinely uncertain, and the two defensible answers bracket a
large range of coupled ice-sheet response ([Jesse et al. (2026)](@cite jesse_ice-sheet_2026)).
Laddie.jl offers both as
`BoundaryConditions(; gaps = ...)`, selected on the same input mask:

| | [`SinkGapsBC`](@ref) (default) | [`ConnectedGapsBC`](@ref) |
|---|---|---|
| gap cell is | open ocean | dynamically active, ice-free |
| ``H, T, S`` in the gap | not stepped | stepped |
| gap edge behaves like | the calving front | the shelf interior |
| heat and momentum arriving at the gap | leave the cavity | continue downstream |

Real gaps are full of sea ice, mélange and bergs, so neither end-member is right on its
own; running both is how the uncertainty is quantified. The
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

## API

```@docs
AbstractGapsBC
SinkGapsBC
ConnectedGapsBC
```
