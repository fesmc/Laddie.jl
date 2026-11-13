# Relaxed frazil ice

- **Type:** dead end
- **Status:** code in main, default off (`NoFrazilIce()`); not used in reference runs since 2026-11-04
- **Commits:** fed5db0 (feature), 9f6b8d7 (merge)
- **Code:** `RelaxedFrazilIce` in `src/frazil.jl`; `update_frazil!` in `src/physics.jl`; output `:frazil`

## Idea

Laddie under-refreezes beneath Ronne compared with ocean models.
Real ISW plumes release most of their supercooling as frazil (Jenkins & Bombosch 1995), and Laddie only has basal freezing.
Closure: `ḟ = c_p H max(T_f − T, 0) / (L τ)`, which relaxes T to T_f over τ.
Salt stays in the layer, and the ice leaves at the layer's T and velocity.
There is no published τ for a depth-integrated plume. The default of 1 h comes from Rees Jones & Wells (2018).

## Why it is a dead end

Pan-Antarctic, BedMachine v4, v2 physics, ISMIP6 J20 front-cell ocean, day 300, Gt/yr ([laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin)).
LADDIE v2 has no frazil and refreezes 38.9 Gt/yr.

| setup | basal refreezing | frazil | net melt, 42 shelves |
|---|---|---|---|
| 1 km, 0.8 cap, first-guess parameters, τ 1 h | ~1 | 23 | +8 |
| 2 km, 0.8 cap, τ 1 h | 4.0 → 0.1 | 90 | 1683 → 1706 |
| 2 km, absolute cap (reference spin-up), τ 24 h | 38.2 → 3.5 | 241 | 1620 → 1704 |

- Frazil **replaces** basal refreezing instead of adding to it. It is 2.3–6× v2's total refreezing.
- The pattern is wrong: Paolo et al. (2023) freezing areas (central/eastern Ronne, Amery, Ross, Filchner, Brunt) turn to weak melt.
- The actual cause of missing refreezing was the relative layer-thickness cap.
  Without it, v2's refreezing comes back shelf by shelf (Ronne 22.0 vs 22.0).

## What would reopen it

- Checking how much of the frazil sits in cells thicker than their water column (not done).
- A prognostic frazil concentration with settling (Smedsrud & Jenkins 2004), instead of instant deposition.
