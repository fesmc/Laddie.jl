# Resolution dependence of melt near the ice front

- **Type:** open question
- **Status:** open since 2026-11-20. Under the default front (`OutflowOnlyOpenOceanBC` + `ZeroGradientFrontPressure`) the warm pan-Antarctic melt still drops by 5 % from 2 to 1 km
- **Code:** `OutflowOnlyOpenOceanBC`, `clip_front_inflow!`, `ZeroGradientFrontPressure` (`src/bcs/open_ocean.jl`); `PerBasinOceanForcing1D` (`src/bcs/forcing.jl`)
- **Docs:** `docs/src/bcs/open_ocean.md`, `docs/src/bcs/front_pressure.md`

## Question

Integrated melt depends on resolution, and in warm runs most of the dependence sits within 10 km of the ice front.
The default front removed the front-row pile-up (`improvements/ice-front-defaults.md`) and cut the 2 to 1 km gap from 8 % to 5 %.
What remains is a speed-up of the plume over its last two cells before the front, so its width scales with the cell size.
Is that a discretization artefact, and can it be removed without a tuning knob?

## What we know

All numbers: pan-Antarctic, BedMachine v4, the fitted v2 setup of [laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin) (front-cell forcing blended over 64 km, no slip).
Melt is the mean over the last 10 days, in Gt/yr.

**Resolution series** under the former default (`ZeroGradientOpenOceanBC` + `ZeroExteriorFrontPressure`). 8 and 4 km spun up from rest; 1 and 0.5 km refined from the 2 km state. Each run is flat over its last 30 days:

| cell size | 8 km | 4 km | 2 km | 1 km | 0.5 km |
|---|---|---|---|---|---|
| cold (present-day) | 1018 | 1110 | 1229 | 1270 | 1215 |
| warm (Mathiot) | 12938 | 13470 | 13635 | 12603 | 12456 |

- Cold melt peaks at 1 km, warm at 2 km. From 1 to 0.5 km the change is 1–4 %.
- Right after refinement, with the same state, the 1 km grid melts 7 % less than the 2 km grid on a 1 % smaller shelf, so part of the drop is geometry.
- In cold runs the front row is fast (0.12 against 0.05 m/s inside) but cold (TF 0.18 °C), so it melts no more than the interior.

**Front options**: warm, 50-day continuations of the converged 2 km (day 1500) and 1 km (day 1600) states:

| open ocean + front pressure | 2 km | 1 km | gap |
|---|---|---|---|
| `ZeroGradientOpenOceanBC` + `ZeroExteriorFrontPressure` (former default) | 13636 | 12604 | 1032 (8 %) |
| `ZeroGradientOpenOceanBC` + `V1FrontPressure` | 13517 | 12519 | 998 |
| `ZeroGradientOpenOceanBC` + `ZeroGradientFrontPressure` | 13862 | 13183 | 679, still falling at day 50 |
| `NoInflowOpenOceanBC` + `ZeroExteriorFrontPressure` | 16515 | 13770 | 2745 (17 %) |
| `OutflowOnlyOpenOceanBC` + `ZeroExteriorFrontPressure` | 13123 | 12466 | 657 (5 %) |
| `OutflowOnlyOpenOceanBC` + `ZeroGradientFrontPressure` (default) | 13122 | 12465 | 657 (5 %) |

- With the valve, the front pressure is irrelevant (±1 Gt/yr). Without it, the choice moved the total by up to 580 Gt/yr.
- The front row is slower at 1 km (0.107 against 0.127 m/s) with or without the density-gradient term. The 1/Δx scaling of that one-sided term is therefore not what slows it.
- Same default, cold, 1 km, 50 days from day 1200: 1253 against 1270 under the former default (−1.3 %). The melt pattern is unchanged.

**The ambient forcing does not depend on resolution.** Ambient TF = T_a − T_f(S_a, z_draft) from the forcing table, mean per distance band from the front (warm):

| band | TF at draft (°C), 2 / 1 / 0.5 km | TF at plume base, 2 / 1 km | mean draft (m), 2 / 1 / 0.5 km |
|---|---|---|---|
| 0–2 km | 3.04 / 3.01 / 3.00 | 2.99 / 3.00 | −90 / −147 / −162 |
| 2–4 km | 3.00 / 3.01 / 3.01 | 3.01 / 3.03 | −173 / −185 / −188 |
| 4–10 km | 3.07 / 3.07 / 3.07 | 3.09 / 3.11 | −207 / −212 / −212 |
| 10–50 km | 3.16 / 3.15 / 3.15 | 3.21 / 3.22 | −265 / −266 / −268 |
| > 50 km | 3.08 / 3.07 / 3.10 | 3.14 / 3.15 | −465 / −467 / −468 |

- Within 1 % in every band, and the band areas agree.
- Coarse front cells average in shallower ice and sample warmer water (T_a 1.29 against 1.18 °C), but the freezing point offsets it.

**The plume against the forcing**, default front, warm, last slice of the 50-day continuations (gap 602 Gt/yr on this slice):

| band | plume TF / ambient TF, 2 / 1 km | melt (m/yr) | \|u\| (m/s) | band total (Gt/yr) | gap |
|---|---|---|---|---|---|
| 0–2 km | 0.78 / 0.72 | 27.7 / 22.9 | 0.126 / 0.107 | 545 / 472 | 72 |
| 2–4 km | 0.68 / 0.65 | 26.3 / 17.8 | 0.135 / 0.094 | 503 / 338 | 166 |
| 4–10 km | 0.57 / 0.57 | 17.1 / 14.4 | 0.101 / 0.086 | 872 / 732 | 140 |
| 10–50 km | 0.42 / 0.45 | 13.2 / 13.0 | 0.094 / 0.088 | 3256 / 3179 | 77 |
| > 50 km | 0.22 / 0.23 | 7.1 / 7.1 | 0.092 / 0.089 | 6857 / 6710 | 147 |

- Heat dilution scales with distance, not with cells: plume TF / ambient TF falls alike on both grids.
- The speed-up spans about two cells. At 2 km it reaches 4 km inland, at 1 km it is gone after 2 km.
- In the 2–4 km band speed, not TF, sets the gap: TF ratio 0.68 against 0.65, but |u| +44 % and melt +48 %.
- About 380 Gt/yr of the gap lies within 10 km of the front, mostly at 2–10 km. The rest is mostly area: the 2 km shelf is 1.2 % larger, at the same melt rate in the far interior.
- Comparing front rows alone (545 against 247 Gt/yr) overstates their share: the 0–2 km band holds one row at 2 km and two at 1 km.
- Not fully settled: over the last 30 days 2 km still rises by about 1.5 Gt/yr per day, 1 km by about 0.6.

## Candidate explanations

- **A numerical boundary layer at outflow faces.** The one-sided front face, with zero-gradient pressure and upwind fluxes, lets the outflow accelerate over the last one or two cells. Its width is then Δx, and the integral scales with Δx.
- **A physical acceleration shorter than 2 km**, unresolved at 2 km. Then the boost at 0.5 km would span the same distance in km as at 1 km, not the same number of cells.
- **Geometry.** Coarse front cells average in shallower ice (−90 against −162 m), and a coarser grid keeps more shelf. That explains the far-interior part of the gap, not the near-front one.

## Next test

- Warm 0.5 km continuation under the default front. Compare the speed and melt profile per cell against distance at 2, 1 and 0.5 km: two cells at every resolution means numerical.
- Momentum budget of the last three rows (advection, pressure gradient, friction, entrainment drag) at 2 and 1 km, to see which term carries the speed-up.
- Integrate melt on the common 2 km mask, to separate the geometry part.
- Remedies so far either fail or add a knob:
  - A Froude condition on outflow faces is a dead end (`deadends/froude-front-condition.md`).
  - A relaxation zone of physical width behind the front, or a Robin pressure condition with a decay length L (`improvements/ice-front-defaults.md`), would be resolution-independent but tuned.
