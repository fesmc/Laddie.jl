# Flush subnormal floats on device transfer

- **Type:** improvement
- **Status:** always on since 2026-10-28
- **Commits:** 823cc7c (feature)
- **Code:** `_flush_subnormals` in `_to_device` (`src/backend.jl`) and in the sharded placements of the Reactant extension

## Summary

A Float32 run on CUDA keeps subnormal values, e.g. velocities at rest that decay to ~1e-42 on wet–dry faces.
XLA on the GPU flushes subnormals to zero, while `_safe_div` and `_safe_sqrt` test only for an exact zero.
So the reverse pass hit ∞·0 = NaN, and every gradient was NaN.
Subnormals are now set to zero whenever a model moves to a backend.

## Evidence

- Pan-Antarctic 2 km restart: 5 subnormal U/V values on wet–dry faces at front corners made every reverse-mode gradient NaN.
  With the flush, the one-step reverse probe on the same restart is finite.
- A bisect over Laddie commits had blamed the physics commits since 9f6b8d7.
  It was confounded: every NaN run started from a restart with subnormals. On that restart, 9f6b8d7 and HEAD give the identical NaN.

## Caveats

- Any restart from a CUDA Float32 run can hold subnormals.
  If NaN adjoints appear again, check the input state for subnormals and non-finite values before bisecting commits.
- The Reactant forward-AD vs FD test tolerance was relaxed to 1e-4 in the same commit (the ±1e-3 °C central difference is good to ~1e-4 under the flux form).
