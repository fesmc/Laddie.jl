# FB-RK3 against the resolution dependence and the unstable gradients

- **Type:** dead end
- **Status:** `FBRK3Integrator` stays in main, opt-in (2026-11-23): it is the better forward scheme (`improvements/fbrk3-integrator.md`), but it does not address either problem below
- **Commits:** f79ed26 (feature), 4b92132 (tests), 3790f8d (default CFL target 2.0)
- **Code:** `FBRK3Integrator` (`src/numerics/integrator.jl`), `_step_model!` (`src/numerics/fbrk3.jl`); the runs use `INTEGRATOR=fbrk3` of [laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin) (`scripts/panais/model-setup.jl`, `inverse.jl`, `resolution-dependence-fbrk3.jl`)

## Idea

FB-RK3 has no computational mode and is 2.5–5.5× more accurate than the leapfrog at equal Δt (`improvements/fbrk3-integrator.md`).
If the time scheme caused part of the resolution dependence of the pan-Antarctic melt (`open-questions/front-resolution-dependence.md`), FB-RK3 would reduce it.
If the leapfrog's computational mode, or the filter, made the tangent and adjoint grow on long warm windows, FB-RK3 would let the inverse problem use longer windows.

## Why it is a dead end

**The resolution dependence is not a time-stepping error.**
At equal resolution and parameters, FB-RK3 and the leapfrog give the same integrated melt within 1 %, while the cell size moves it by up to 28 % (cold, 8 to 1 km).

Pan-Antarctic, BedMachine v4, v2 physics, OCEAN ice-climatology front-cell ocean, mean of the last 30 days (Gt/yr):

| series | 8 km | 4 km | 2 km | 1 km | 0.5 km |
|---|---|---|---|---|---|
| leapfrog CFL 0.4, past closures, fitted parameters, zero-gradient front inflow: cold (day 1000–1200) | 1018 | 1110 | 1229 | 1270 | 1215 |
| same, warm (Mathiot NEMO warm−cold change, day 1500–1600) | 12938 | 13470 | 13635 | 12603 | 12456 |
| FB-RK3 CFL 2.0, first-guess parameters, outflow-only front, from rest: cold (day 1000) | 991 | 1125 | | 1269 | |
| same, warm (day 1500) | 11842 | 12804 | | | |

At 2 km and 1 km with the fitted parameters and the outflow-only front, only the scheme differing:

| | FB-RK3, CFL 2.0, from rest | leapfrog, CFL 0.4, 50 d from the converged leapfrog state |
|---|---|---|
| 2 km cold | 1227 | 1229 (zero-gradient inflow) |
| 2 km warm | 13250 | 13122 |
| 1 km warm | | 12465 |

- From 8 to 4 km, FB-RK3's cold melt rises 13.5 % (leapfrog 9.0 %), its warm melt 8.1 % (leapfrog 4.1 %); from 8 to 1 km cold, 28 % (leapfrog 25 %).

**FB-RK3 does not stabilise the gradient on warm windows; at its default CFL it breaks earlier.**
Tangent of the cost J (melt misfit, 42 named shelves, mean of the last 2 days of the window) along log C_d_top, pan-Antarctic 2 km, fitted parameters:

| window (d) | 1 | 2 | 3 | 5 | 7 | 10 | 15 |
|---|---|---|---|---|---|---|---|
| cold, leapfrog CFL 0.4, past closures | 0.434 | 0.442 | 0.454 | 0.464 | 0.468 | 0.473 | 0.483 |
| cold, FB-RK3 CFL 2.0 | 0.433 | 0.442 | 0.454 | 0.464 | 0.469 | 0.471 | 0.400 |
| cold, FB-RK3 CFL 1.0, own spin-up, outflow-only front | 0.414 | 0.423 | 0.437 | 0.450 | 0.456 | 0.461 | 0.525 |
| warm, leapfrog CFL 0.4, past closures | 19.3 | 20.1 | 21.4 | 23.8 | 19.6 | 33.5 | −1.4e4 |
| warm, FB-RK3 CFL 2.0 | 19.3 | 721 | −2.2e5 | 3.4e22 | NaN | NaN | NaN |
| warm, FB-RK3 CFL 1.0, own spin-up, outflow-only front | 20.1 | 21.0 | 22.2 | 17.8 | −185 | −3.5e4 | |

- Cold, all three agree within 3 % up to 10 days.
- Warm, the leapfrog stays bounded to 10 days, FB-RK3 to 1 day at CFL 2.0 and to 5 days at CFL 1.0.
- The blow-up is local and the same under both schemes: tangents of 10⁸–10¹⁰ in a few cells with 2–260 m of layer (leapfrog at 15 d, FB-RK3 CFL 1.0 at 10 d), while the primal stays bounded (max |H − H_restart| 450–570 m).
- So the growth is a property of the warm dynamics (thin, fast, strongly melting cells), not of the leapfrog's computational mode or its filter.
  A larger Δt only lets it grow faster per day.

## What would reopen it

- A resolution dependence that changes with Δt at fixed cell size, or a front treatment that removes it under one scheme only.
- A warm-window tangent that stays bounded under FB-RK3 at small CFL (≤ 0.5) for longer than under the leapfrog: the CFL-1.0 sweep started from FB-RK3's own spin-up with a different front, so the scheme and the state are not yet separated at equal CFL.
- The FB-RK3 resolution series at 1 and 0.5 km (cold done at 1 km, warm not run): the table above has no FB-RK3 point at 2 km in the first-guess series and none below 1 km.
