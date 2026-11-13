# Two-equation melt

- **Type:** dead end
- **Status:** code in main (`TwoEquationMelting`, opt-in, warns on construction); out of the pan-Antarctic physics since 2026-10-27
- **Commits:** dbabc61 (feature; also adds the heat-to-salt ratio `ratio` of `UStarGamTMelting`, default 35, unchanged)
- **Code:** `TwoEquationMelting` in `src/melting.jl`
- **Docs:** `docs/src/physics/melting.md`

## Idea

Jenkins et al. (2010, Eqs. 14–15): put the interface at the freezing point of the layer salinity, so the melt is linear in the thermal driving.
It also takes the ice-conduction closures (`improvements/ice-conduction.md`).

## Why it is a dead end

Pan-Antarctic 2 km, v2 physics, −20 °C ice base, first-guess parameters, 200 d from rest; Gt/yr at day 200 ([laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin)):

| | three-equation | two-equation |
|---|---|---|
| 42 shelves | 1757 | **3439** |
| bias / RMSE vs observed mean | +665 / 40 | +2347 / 93 |
| basal refreezing | 4.4 | 18.8 |
| supercooled cells | 7010 | 19 289 |

- At its default Γ_TS = 0.0178 it melts 1.95× the three-equation form (1.43–2.28× per shelf; George VI 425, Pine Island 230).
  It makes the biases worse.
- The extra refreezing comes from the stronger exchange, not from the form.
- The melt pattern and the refreezing shortfall do not move.

## What would reopen it

Retuning Γ_TS together with C_d_top, to which it is tied. Only then would the form say anything about refreezing.
