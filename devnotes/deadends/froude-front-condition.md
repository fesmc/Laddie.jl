# Froude-number condition at the ice front

- **Type:** dead end
- **Status:** never implemented; dropped 2026-11-18 after a literature review. The front stays zero-gradient (`ZeroGradientFront` + `OutflowOnly`)
- **Commits:** none
- **Code:** the alternatives that stay are `AbstractFrontPressure` and `AbstractOpenOceanBC` in `src/bcs/open_ocean.jl`
- **Docs:** `docs/src/physics/open_ocean.md`

## Idea

Treat the outflow of the meltwater layer at the calving front as a hydraulic control, like a free overfall at the end of a channel.
At outflow front faces, set the face-normal velocity to u_n = Fr_c·√(g′H) (or only cap it there), with g′ = g·δρ/ρ₀ and Fr_c = 1 from theory.
The aim was a front condition independent of the cell size Δ.
In the pan-Antarctic runs, the integrated melt of the front row scales with Δ.

Front row at 2 km, pan-Antarctic, fitted v2 setup, warm forcing, 50 d continuations ([laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin)):

| Open-ocean option | Melt 2 km / 1 km (Gt/yr) | Front row: melt (m/yr), \|u\| (m/s), H (m) | Fr = \|u\|/√(g′H) |
|---|---|---|---|
| `ZeroGradientInflow` | 13636 / 12604 | 36.0, 0.218, 99.5 | ≈ 0.14 |
| `OutflowOnly` | 13123 / 12466 | 28.0, 0.127, 24.2 | ≈ 0.16 |

The interior has Fr ≈ 0.10 (u ≈ 0.10 m/s, H ≈ 40 m).
g′ ≈ 0.025 m/s² is from the salinity contrast only (layer S ≈ 31.2, ambient ≈ 34.5 psu).

## Why it is a dead end

- **No source makes the ice front a control on the outflow.**
  Hydraulic control near ice acts on the warm inflow over a sill, not on the meltwater outflow at the front.
  - Nilsson et al. (2023, TC 17, 2455), for Petermann, Ryder and 79N: control at the sill crest, written as a maximum transport.
  - Zhao, Stewart & McWilliams (2019, JPO 49, 163), for rotating cavities: control only in the low-friction, high-sill regime, on the deep layer.
- **With rotation, a control sets the transport across the whole front, not a Froude number per face.**
  - Through an opening wider than the Rossby radius Rd, the controlled transport is Q = g′h²/(2f), independent of the width (Whitehead, Leetmaa & Knox 1974; Pratt & Whitehead 2008, *Rotating Hydraulics*).
  - The outflow then leaves as a boundary current of width ~Rd.
  - Antarctic fronts are 10–100 km wide against Rd = √(g′H)/f ≈ 11 km.
  - A per-face Fr_c forces an even outflow along the front, which a rotating flow does not have.
- **Today's low Fr is what a rotating control would give anyway.**
  - Spreading g′H²/(2f) over a front of width W gives a mean Fr = Rd/(2W), i.e. 0.09–0.14 at W = 40–60 km.
  - That is the Fr ≈ 0.14–0.16 above.
  - So a cap at Fr_c = 1 would never act, and setting Fr = 1 would make the front row thin and fast with no physical basis.
- **Observed outflows are geostrophic and lopsided, not critical.**
  - Dotson (glider sections; Miles et al. 2016; Zheng et al., EGU24-11749): the meltwater leaves on the western flank in geostrophic balance, and the net volume flux across the front is close to zero.
  - Pine Island (Naveira Garabato et al. 2017, Nature 542, 219): the outflow goes unstable as it rises past the front and settles at several hundred metres depth. It does not spill over to the surface.
- **A tuned Fr_c < 1 would be a free parameter with no theory behind it.**
  It would also act on velocity only.
  The resolution dependence that is left is thermal.
  - With `OutflowOnly`, the front row is as fast as the second row (0.127 vs 0.136 m/s), so the pile-up is gone.
  - It still melts ~2× the interior through its thermal forcing (2.39 °C), at a similar rate at 2 and 1 km.
  - So its integral still scales with Δ: 549 vs 247 Gt/yr.

## What would reopen it

- If the front-row velocity in the pan-Antarctic runs turns out mostly *normal* to the front, not along it (the along-front boundary current that rotation predicts). Split it into normal and tangential parts at 2 and 1 km.
- If the mean front Fr is about the same on shelves of different front width W, rather than falling as Rd/(2W). Then a per-face Fr_c would be plausible.
- If the layer at the front is clearly lighter than the ambient water above the draft (an inverted overfall).
  Even then, classical overfall theory puts Fr > 1 at the brink and the critical section 3–4 critical depths upstream, not at the face.
- A rotating version would constrain the transport ∫ u_n·H dl over each connected front segment to g′H²/(2f), not each face.
  It needs a reduction per segment.
- Still unread: Wiskandt et al. (2025, JGR Oceans, doi:10.1029/2024JC021257), an integral Froude number in a 1.5-layer analogue for Greenland fjords; and Steiger et al. (2022, JPO), where viscosity breaks the potential-vorticity constraint at the ice front.
