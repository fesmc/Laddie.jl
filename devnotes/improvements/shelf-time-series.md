# Shelf-mean time series in the output

- **Type:** improvement
- **Status:** opt-in since 2026-11-10 (`OutputConfig(; tsday)`, off by default)
- **Commits:** a25fd10 (feature)
- **Code:** `OutputConfig` `tsday` in `src/io.jl`

## Summary

With `tsday` set, `output.nc` gets the shelf-mean time series `ts_H`, `ts_T`, `ts_S`, `ts_tf`, `ts_u_norm`, `ts_m` (m/yr) and `ts_m_gt` (Gt/yr), on their own `ts_time` axis.
Each point is a mean over `imask` of the fields averaged over one `tsday` window.
One fused kernel accumulates the fields every step, without a reduction. The masked sums run once per point.
`ts_tf = T − T_f(S, z_draft)` is computed from the mean T and S at the mean draft, so it needs no accumulator.

## Caveats

- Native backends only: `to_backend` rejects a `ReactantBackend` when `tsday` is set.
- `ts_tf` is the thermal driving of the means, not the mean thermal driving.
