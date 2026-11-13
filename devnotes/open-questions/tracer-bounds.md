# T/S bounds as a default

- **Type:** open question
- **Status:** bounds on by default (`T_min = −5`, `T_max = 5` °C, `S_min = 32`, `S_max = 36` psu); off in the pan-Antarctic reference runs
- **Code:** `T_min`, `T_max`, `S_min`, `S_max` in `src/params.jl`; blow-up check in `run!` (c0acc3f)

## Question

LADDIE v2 has no T/S bounds. Laddie.jl needed them because of the corrective clamp and the advective tracer form.
Both are fixed now. Should the bounds be off by default (`−Inf`/`Inf`), or removed?

## What we know

- With the flux form, implicit outflow and no relative cap, pan-Antarctic 1 km and 2 km runs go 200–300 days without bounds.
  At day 30 (1 km), no cell left even [−5, 5] °C / [32, 36] psu: T −3.51…1.31 °C, S 33.02…34.79 psu.
- Bounds hide a diverging run from the blow-up check, because bounded tracers keep melt finite.
  The check covers T and S and runs at least daily since c0acc3f.
- The defaults are tight for fresher settings such as Greenland fjords.
- Under warm (Mathiot) forcing the ambient itself is ~32.4 psu at Shackleton, below `S_min`.

## Candidate explanations

None needed: this is a decision, not a diagnosis.

## Next test

- ISOMIP+, Crosson–Dotson and the v2 reproductions without bounds: does any of them hit them today?
- If not, set the defaults to ±Inf and keep the bounds as an opt-in safeguard. Update the `Params` docs.
