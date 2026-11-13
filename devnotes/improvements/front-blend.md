# Blended front ocean

- **Type:** improvement
- **Status:** opt-in since 2026-11-04 (`front_ocean_forcing(...; blend = FrontBlend(...))`); used in the pan-Antarctic reference runs
- **Commits:** 89420ae (feature), acda2a3 (blend over `front_length`, default 64 km; source columns cut at their seabed)
- **Code:** `front_ocean_forcing`, `FrontBlend`, `_front_seeds` in `src/ocean_sources.jl`
- **Docs:** `docs/src/physics/ocean_forcing.md`

## Summary

With the nearest-front ocean, each cavity cell takes the column outside its nearest front, by distance through the cavity.
Neighbouring cells can take different columns, so the ambient steps in one cell.
`FrontBlend` gives each cell a weighted mean of the source columns of the fronts near its nearest one.
A source enters and leaves a blend at zero weight, so no step remains.

## Evidence

Pan-Antarctic 2 km, BedMachine v4, v2 physics, ISMIP6 J20, no relative cap, 300 d from rest ([laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin)).

With the nearest-front ocean, the steps line up one-cell-wide artefacts:

- Ross_East has straight high-melt lines along x ≈ 148 km.
  Across x = 148 | 150 km the ice-base ambient is 0.24 °C warmer and 0.20 psu fresher on the east side.
  Water converges into one column (H 637–1151 m over a ~390 m water column) fed by thin, fast columns (6–7 m, 0.1–0.2 m/s) on both sides.
- 99 % of the cells with H > 2000 m sit on a source border (79 % for H > 500 m).
  On average 31.8 % of shelf cells do.

Steps in the ambient T to a 4-neighbour at the same depth (the cell's draft), blend width 20 km:

| | nearest front | blend 20 km |
|---|---|---|
| cells with a step > 0.1 / > 0.05 °C | 2.31 / 3.98 % | 0.05 / 0.52 % |
| step p99 / max (°C) | 0.206 / 1.63 | 0.038 / 0.57 |
| Ross_East x 148 \| 150 km, ΔT at the ice base | 0.244 | 0.006 |

## Parts that were needed

- The result is still a `PerBasinOceanForcing1D`, with a row per source column and per blended cavity cell.
  At 2 km that is 384 015 × 30 (0.12 GB at Float32), against 1.4 GB for an `OceanForcing2D` (23 GB at 500 m).
- It builds in ~10 s at 2 km.

## Caveats

- Where the fronts are dense, the blend radius is set by the K-th source (`kmax`, default 64), not by the width.
- LADDIE v2 has 36 vertices above 2000 m with J20 in every cell, so not all runaway cells are this artefact: see `open-questions/layer-thicker-than-water-column.md`.
