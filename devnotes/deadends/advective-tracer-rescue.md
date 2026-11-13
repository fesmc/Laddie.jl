# Rescuing the advective tracer form

- **Type:** dead end
- **Status:** not implemented (coherent clamp) or tested and dropped (cap floor); `AdvectiveTracerForm()` remains only for v1 reproduction
- **Commits:** none for the clamp; the cap floor `H_floor` of `RelativeMaxLayerThickness` is in 5d38792
- **Code:** `AdvectiveTracerForm` in `src/advection.jl`

## Idea

Keep stepping T and S themselves (v1's advective form) and fix its failures where they appear:

1. A coherent clamp: volume clipped at the cap leaves at the layer's T and S, and volume added at the floor enters at T_a and S_a.
2. A floor on the relative cap, `max(f·wc, H_floor)`, so thin water columns are not squeezed to H_min.

## Why it is a dead end

- The advective form carries a time-step error that the flux form does not.
  Warm ISOMIP+ 40 × 20, mean melt at day 2: quartering dt moves it by 9 % (advective) and 0.5 % (flux).
  A coherent clamp would not remove that error, so it was not built for the advective form.
- A 10 m cap floor does not rescue it.
  Larsen C regional run (domain + 30 km, 8 d, v2 physics, 0.8 cap, no T/S bounds):

| variant | shelf S (psu) | cells out of input-water range |
|---|---|---|
| flux form, implicit outflow | 33.05…34.65 | 0 |
| advective form + 10 m cap floor | 30…332 | 5 000–22 000 |
| advective form | −120…190 | 7 000–20 000 |

## What would reopen it

Nothing foreseen. The fix is the flux form (`improvements/flux-form-tracers.md`).
