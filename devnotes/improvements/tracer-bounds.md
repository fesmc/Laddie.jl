# T/S bounds off by default

- **Type:** improvement
- **Status:** default off since 2026-11-18 (`NoTracerBounds()`); `TracerBounds()` opt-in, pinned by the reproduction aliases
- **Commits:** 18142ef (`tracer_bounds` slot), 6fdbc74 (default flip)
- **Code:** `AbstractTracerBounds`, `TracerBounds`, `NoTracerBounds`, `apply_tracer_bounds!` in `src/numerics/tracer_bounds.jl`; blow-up check in `run!` (c0acc3f)
- **Docs:** `docs/src/numerics/tracer_bounds.md`

## Summary

LADDIE v2 has no T/S bounds.
Laddie.jl needed them because of the corrective thickness clamp and the advective tracer form; the coherent clamp, implicit outflow and the flux tracer form removed that need (`flux-form-tracers.md`).
The bounds became a scheme slot, off by default, and stay available as a safeguard.

## Evidence

- With the flux form, implicit outflow and no relative cap, pan-Antarctic 1 km and 2 km runs go 200–300 days without bounds ([laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin)).
  At day 30 (1 km), no cell left even [−5, 5] °C / [32, 36] psu: T −3.51…1.31 °C, S 33.02…34.79 psu.
- The reproduction pins (`REPRODUCTION_MEAN_MELT`) are unchanged: the aliases select `TracerBounds()`.

## Caveats

- Bounds hide a diverging run from the blow-up check, because bounded tracers keep the melt finite. The check covers T and S and runs at least daily.
- The `TracerBounds` defaults (−5…5 °C, 32…36 psu) are tight for fresher settings such as Greenland fjords, and for warm (Mathiot) forcing, whose ambient is ~32.4 psu at Shackleton.
- `TracerBounds()` in the aliases keeps Laddie.jl's earlier behaviour, not that of the published codes.
