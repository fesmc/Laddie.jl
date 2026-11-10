# Ice-shelf gaps

Where melting thins a shelf all the way through, an ice-free **gap** opens inside the
domain. There is no ice base there, so no melt — but what happens to the meltwater layer
arriving from upstream is genuinely uncertain, and the two defensible answers bracket a
large range of coupled ice-sheet response ([Jesse et al. (2026)](@cite jesse_ice-sheet_2026)).
Laddie.jl offers both, and a physically based variant of the connected gap, as
`BoundaryConditions(; gaps = ...)`, selected on the same input mask:

| | [`SinkGapsBC`](@ref) (default) | [`ConnectedGapsBC`](@ref) | [`LeakyGapsBC`](@ref)`(α, Q)` |
|---|---|---|---|
| gap cell is | open ocean | dynamically active, ice-free | dynamically active, ice-free |
| ``H, T, S`` in the gap | not stepped | stepped | stepped, plus surface heat loss and convection |
| ice–ocean drag in the gap | – | full | scaled by the mélange cover α |
| gap edge behaves like | the calving front | the shelf interior | the shelf interior |
| heat and momentum arriving at the gap | leave the cavity | continue downstream | continue downstream, less what the open water takes |

Real gaps are full of sea ice, mélange and bergs. [`LeakyGapsBC`](@ref) represents them
by their cover fraction α: the covered part acts on the layer as the ice shelf does, the
open part loses heat to the atmosphere. Running the sink and the connected gap is how the
uncertainty of the treatment is quantified; [`LeakyGapsBC`](@ref) shows where the
processes in an enclosed gap put the answer between them. The
[ice-shelf gaps example](../generated/jesse-gaps.md) runs the sink and the connected gap on
the paper's own geometries and compares them with [LADDIE v2](../versions.md).

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
the section above, and covers a fraction α of each gap with mélange and sea ice (default
``\alpha = 0.9``). The layer's thickness is never relaxed: it changes only by transport,
entrainment, the sea ice the open water forms and convection. Three processes act in gap
cells.

**Drag.** The ice–ocean drag coefficient ``C_d`` is scaled by α in gap cells, by the mean
cover of the two cells of a velocity face. `ConnectedGapsBC` applies the full drag in its
gaps, as if they were covered; open water (``\alpha = 0``) applies none.

**Open-water heat loss.** The open fraction loses ``(1 - \alpha)\,Q`` to the atmosphere
(``Q`` = `surface_heat_loss`, W m⁻², default 0). After the tracer step it first cools the
layer, and once the layer reaches the surface freezing point ``T_f(S, 0)``, the rest forms
sea ice of salinity ``S_i`` (`ice_salinity`, default 5 psu):

```math
\delta T = \frac{(1 - \alpha)\,Q\,\Delta t}{\rho_0\,c_p\,H}, \qquad
h = \frac{c_p\,\delta T_f\,H}{L}, \qquad
H' = H - h, \qquad S' = \frac{S H - S_i h}{H'},
```

with ``\delta T_f`` the part of the cooling below the freezing point, which the latent
heat holds the layer at. The frozen water leaves the layer and its salt beyond ``S_i``
stays: brine rejection, with the bookkeeping of [frazil ice](frazil_ice.md).

**Convection.** Where the layer is denser than the water beneath it
(``\delta\rho < 0``), its ``T`` and ``S`` relax towards ``T_a`` and ``S_a`` on the time
scale τ (`convection_time`, default 10⁴ s), as under [`RelaxToAmbient`](@ref), and its
thickness relaxes towards ``H_\mathrm{min}`` at the same rate, so the convected water
leaves the layer. This runs in every gap cell, whatever the cover and whatever the
model's convection scheme, which leaves gap cells alone (see
[Convection in gaps](convection.md#Convection-in-gaps)).

With ``\alpha = 1`` there is no surface exchange and the full drag: this is
`ConnectedGapsBC`, plus convection, and identical to it wherever no gap cell turns
unstable. ``\alpha = 0`` is open water. It is not the sink: the sink assumes the layer
leaves the cavity at the gap, and none of the three processes removes it nearly as fast.

On the ISOMIP+ channel of the test suite (a 12 × 24 km gap across the boundary current,
downstream melt summed over the wall rows, mean over days 18–20; sink 103.9, connected
164.2):

| α | ``Q = 0`` | ``Q = 300`` W m⁻² | ``Q = 600`` W m⁻² |
|---|---|---|---|
| 0 (open water) | 209.6 | 183.3 | 174.2 |
| 0.25 | 196.8 | 178.1 | 165.0 |
| 0.5 | 184.9 | 173.1 | 160.8 |
| 0.8 | 171.9 | 167.6 | 163.1 |
| **0.9** | **168.0** | **165.8** | **163.7** |
| 1 (full cover) | 164.2 | 164.2 | 164.2 |

Open water removes the drag, so the current crosses the gap faster and melts more
downstream than under the connected gap; the heat loss takes part of that back. The gap
layer stays buoyant here, so convection hardly acts. The layer in this gap is 230–245 m
thick: it fills the upturned bowl the ice base forms around the gap, and is renewed
within about two weeks, faster than a heat loss of a few hundred W m⁻² can cool and
salt it to neutral buoyancy.

On the two year-200 geometries of the
[ice-shelf gaps example](../generated/jesse-gaps.md) (total melt, Gt yr⁻¹, days 20–30):

| geometry | sink | ``\alpha = 0`` | ``\alpha = 0``, ``Q = 300`` | **``\alpha = 0.9``** | ``\alpha = 0.9``, ``Q = 300`` | connected |
|---|---|---|---|---|---|---|
| small-gap (five gap cells at the grounding line) | 38.7 | 68.6 | 68.4 | **68.6** | 68.6 | 68.6 |
| large-gap (northern margin open over 300 km) | 52.0 | 65.4 | 64.3 | **65.6** | 65.5 | 65.6 |

Within the processes represented, an enclosed gap behaves much like a connected one. At
the default cover the leaky gap is within 2.5 % of `ConnectedGapsBC` on every geometry
tested, and even open water stays on the connected side: up to 28 % more melt on the
channel, at most 2 % less on the real geometries, against 20–45 % less under the sink. The default ``\alpha = 0.9`` describes a gap mostly
filled with mélange and sea ice, and ``Q = 0`` leaves the heat loss off until a value for
the region is chosen; estimates for coastal polynyas reach a few hundred W m⁻² in winter.
The sink remains the end-member for a gap through which the layer can leave the cavity.

## API

```@docs
AbstractGapsBC
SinkGapsBC
ConnectedGapsBC
LeakyGapsBC
```
