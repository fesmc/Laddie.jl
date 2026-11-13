# Relative layer-thickness cap

- **Type:** dead end
- **Status:** code in main (`RelativeMaxLayerThickness(f, H_floor)`, opt-in); not used in reference runs since 2026-11-04, which take v2's `AbsoluteMaxLayerThickness(10 000)`
- **Commits:** 45b4d3e (enforced by detrainment), 5d38792 (`H_floor`)
- **Code:** `RelativeMaxLayerThickness` in `src/max_layer_thickness.jl`
- **Docs:** `docs/src/numerics/max_layer_thickness.md`

## Idea

In a one-layer model the plume should not be thicker than the water column.
Cap H at f × water column (f = 0.8 leaves ambient water beneath the layer), enforced by detrainment (`preventive-thickness-cap.md`).
LADDIE v2 uses a fixed 10 000 m, which in practice is no cap.

## Why it is a dead end

The detrainment that holds the cap is itself unphysical. It suppresses refreezing and the strong boundary currents.

Pan-Antarctic 2 km, BedMachine v4, v2 physics, ISMIP6 J20 front-cell ocean, flux tracers + flux momentum, 300 d from rest; Gt/yr ([laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin)):

| | v2 | 0.8 cap | no cap (10 000 m) |
|---|---|---|---|
| basal refreezing | 38.9 | 4.0 | 38.2 |
| Ronne refreezing | 22.0 | 1.3 | 22.0 |
| Amery refreezing | 6.3 | 0.1 | 5.1 |
| speed mean / p99 (m/s) | 0.038 / 0.262 | 0.044 / 0.216 | 0.050 / 0.250 |
| 42 shelves net / RMSE vs obs. | 1480 / 36.6 | 1683 / 36.2 | 1620 / 34.0 |
| final dt (s) | 90 (fixed) | 463 | 322 |

- Removing the cap recovers v2's refreezing shelf by shelf. No other switch tried (EOS, conduction, convection, front inflow, BedMachine version, frazil) moves refreezing above 4.2 Gt/yr or p99 speed above 0.225 m/s.
- Without the cap, the run refreezes where Paolo et al. (2023) do (central and eastern Ronne, Amery, Ross, Filchner, Brunt).
- This also closes the FRIS question of the missing central-Ronne freezing body (1 km FRIS runs with the cap: 0.97 % of the shelf freezing, against ~21 % in the ocean-model ensemble).
- Lowering C_d_top 4× and raising Pr 5× had fitted FRIS melt under the cap, but at the price of almost all refreezing.
  It was a compensation.

## What would reopen it

- Most of v2's refreezing is where the layer fits: 9.5 of 37.0 Gt/yr lies where H > water column, and 4.2 where H > 2×.
  So a cap at 1× would touch about a quarter of the refreezing, and one at 2× about a tenth.
  A cap at 2–5× could remove the runaway cells and recover dt (`open-questions/layer-thicker-than-water-column.md`).
- `H_floor` and the subgrid water column (`flux-form-tracers.md`) only matter under a relative cap.
