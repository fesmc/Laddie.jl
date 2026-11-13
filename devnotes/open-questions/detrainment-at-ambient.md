# Ordinary detrainment leaves at ambient T and S

- **Type:** open question
- **Status:** unchanged since v1
- **Code:** `nentr = entr + ent2 − detr` in `_net_entrainment_kernel!` (`src/physics.jl`); the tracer steps in `src/numerics.jl`

## Question

Ordinary detrainment (`detr`, limited by `max_detrainment`) is folded into the net entrainment, which enters the tracer equations as `nentr·T_a`.
So detrained water leaves at the *ambient* T and S, not at the layer's own.
Should it leave at the layer's T and S, as `detr_cap` already does and as v2 does (`− max(0, detr)·T`)?

## What we know

- In the advective form, a detraining cell gets `detr·(T − T_a)`, which pushes T *away* from ambient. Detraining layer water should leave T unchanged.
- In the flux form, the content loses `detr·T_a` instead of `detr·T`, which is the same error.
- `detr_cap` was built to avoid exactly this (`improvements/preventive-thickness-cap.md`), and it goes through the implicit sink of the flux form.
- Its size in reference runs has not been measured. In the 2 km corner probe `detr` was 0.

## Candidate explanations

It is inherited from v1, and probably harmless wherever `detr` is small compared with `entr`.

## Next test

- Map `detr` and `detr·|T − T_a|·H⁻¹` on a pan-Antarctic reference run to see where it matters.
- Move `detr` out of `nentr` into the implicit sink of the flux form, next to `detr_cap`, and compare melt per shelf.
