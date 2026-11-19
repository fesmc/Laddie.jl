# Ice-shelf gaps: what is still unresolved

- **Type:** open question
- **Status:** `SinkGapsBC` default; `ConnectedGapsBC` and `LeakyGapsBC` opt-in (`improvements/leaky-gaps.md`)
- **Code:** `src/bcs/gaps.jl`; `MarkGapsPreprocess` in `src/utils/preprocessing.jl`
- **Docs:** `docs/src/bcs/gaps.md`, `docs/src/examples/jesse-gaps.jl`

## Question

The sink and the connected gap bracket the response (Jesse et al. 2026), and the leaky gap shows where an enclosed gap sits between them.
What is left before a gap result can be quoted on a real cavity?

## What we know

- **The gap lens.** A connected gap is an inverted basin that fills to its spill depth: 150–330 m of layer on the ISOMIP+ channel, against 11–13 m upstream. The lens recirculates around the dome (up to 7e5 m³/s through the wall rows, against 3–6.5e4 m³/s throughflow) and is renewed in about two weeks. Eroding it (0.35 psu over 250 m) takes ~4 months at 100 W m⁻². Not yet checked on real geometries (gap H against the shallowest draft on the way out).
- **No process reaches the sink at a realistic Q** (`improvements/leaky-gaps.md`); the sink stays the end-member for a gap the layer leaves through.
- **Gaps punched into thick ice.** A gap draft of 0 next to a −400 m ice base gives a large ice-slope pressure gradient. On the ISOMIP+ channel it drove H to 317 m in the gap and peak melt from 13 to 57 m/yr, inflating the downstream signal 23×. Finite and stable, but not quotable. The real melt-through geometries of Jesse et al. do not have the cliff, so this concerns hand-made masks.
- **Gaps that touch the open ocean** stay connected under `ConnectedGapsBC`, the paper's deliberate upper bound (their §4.3), argued harmless for its geometry. On a real cavity, a gap open to the shelf sea differs from an enclosed one.
- **Residuals against LADDIE v2** on both year-200 gap geometries: v2's layer is thicker and fresher (small-gap shelf: mean H 14 vs 7 m, mean S 33.84 vs 34.06 psu). Not specific to gaps, and the example page runs `CentredMomentumAdvection`, while the v2 MISMIP+ config uses `'upstream'`; part of its −9 % may be the advection scheme.
- **Not implemented:** mélange melt (volume and cooling), tiny per crossing; it would need its own field to stay out of the shelf melt totals.
- **Testing:** the idealised gap is in the suite; the real geometries are opt-in, since they need 7.6 GB of paper output.

## Candidate explanations

None: these are gaps in coverage, not unexplained results.

## Next test

- Real geometries: gap H against the spill depth.
- A build-time warning on the draft gradient at gap edges.
- Rerun the gaps example with `UpstreamMomentumAdvection` and recompare with v2.
- An option that breaks connectivity where a gap joins the open ocean.
- Commit a small regridded extract of one real gap geometry (a few MB) as a regression test.
- If the page should show the intermediate case, add a leaky run to `docs/src/examples/jesse-gaps.jl`.
