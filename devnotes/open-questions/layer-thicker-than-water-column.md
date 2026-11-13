# Layer thicker than its water column

- **Type:** open question
- **Status:** accepted for reference runs since 2026-11-04 (`AbsoluteMaxLayerThickness(10 000)`, as v2)
- **Code:** `AbsoluteMaxLayerThickness`, `RelativeMaxLayerThickness` in `src/max_layer_thickness.jl`

## Question

Without a relative cap, the one-layer plume grows thicker than the water column in 5–7 % of the shelf, and a few cells run away to H_max.
How can those cells and the dt they cost be removed without the refreezing loss of a relative cap (`deadends/relative-thickness-cap.md`)?

## What we know

Pan-Antarctic 2 km, BedMachine v4, v2 physics, ISMIP6 J20, day 300 ([laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin)).
The water column is that of the deepest 500 m subcell:

| H / water column | v2 (312 964 vertices) | no cap (379 196 shelf cells) | 0.8 cap |
|---|---|---|---|
| > 1 | 22 683 (7.2 %) | 20 572 (5.4 %) | 52 |
| > 2 | 12 040 | 10 339 | 1 |
| > 10 | 2 200 | 2 189 | 0 |

- Removing the cap reproduces v2's layer, impossible cells included. v2's are in the big cold cavities (Ronne, Ross, Filchner, Amery).
- Runaway cells set the gravity-wave CFL: dt 463 → 322 s. LADDIE v2 runs at a fixed 90 s and does not feel this.
- With the nearest-front ocean, 99 % of the cells with H > 2000 m sat on a forcing border. `FrontBlend` removes most of them (`improvements/front-blend.md`).
- Under warm (Mathiot) forcing, the cells at H_max are isolated one-cell spikes in deep Ronne, set off by the switch to warm forcing. dt and ν do not change them.
- Removing dead-end cells from the mask removed the worst runaway (a 32 m/s one-cell pocket), and dt tripled (26 → 84 s).
- The cold state is not in equilibrium in H: from day 400 to 600, mean H grows 46.5 → 50.3 m (+8 %) and refreezing 47.0 → 51.8 Gt/yr, while melt changes by −0.5 %.

## Candidate explanations

- Degenerate geometry (dead ends, one-cell pockets, 1–2 m water columns at grounding lines) traps buoyant water with no exit.
- Inverted basins of the ice base fill to their spill depth, as in the gap channel (`improvements/leaky-gaps.md`). That would be physical, and a one-layer model cannot represent the water underneath.

## Next test

- Relative caps at 1×, 2× and 5× the water column (`F_HMAX` in laddie-perbasin, no code change): the lowest one that keeps v2's refreezing, and how much dt each recovers.
- An absolute cap of ~1000–2000 m as the alternative.
- Locate where the H_max cells pile up under warm forcing (679 cells after the dead ends were removed).
