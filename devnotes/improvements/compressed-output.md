# Compressed map output

- **Type:** improvement
- **Status:** default since 2026-11-17 (`OutputConfig(; float_type = Float32)`, NaN fill, deflate always on)
- **Commits:** 5fb275d (feature and default)
- **Code:** `OutputConfig` (`float_type`), `_COMPRESSION`, `_create_output_file!`, `_write_output!` in `src/utils/io.jl`

## Summary

The time-averaged maps in `output.nc` are written as Float32 (Float64 on request), with the fill value NaN outside `tmask` and lossless deflate level 1 with shuffle, one time slice per chunk.
Before, they were uncompressed Float64 with 0 on dry cells.
On a pan-Antarctic 2 km grid this shrinks a map 30–57-fold. For a Float32 model nothing is lost.
Time series, static fields (`mask`, `z_draft`) and restarts are unchanged.

## Evidence

Final slice of two pan-Antarctic 2 km runs from [laddie-perbasin](https://github.com/JanJereczek/laddie-perbasin): warm forcing at day 200 and cold forcing at day 400.
Both runs are Float32 models with no tracer bounds and `v_cut = 2` m/s, on 2588 × 2191 cells, of which 6.2 % are active.
Each field was written alone, one 2D chunk; the table gives bytes per field for the warm run (cold within ±15 %).

| encoding | H | S | T | Ut | melt | max error vs Float64 |
|---|---|---|---|---|---|---|
| Float64, 0 on dry cells, uncompressed (previous default) | 45.4 MB | 45.4 | 45.4 | 45.4 | 45.4 | 0 |
| Float64 + NaN + deflate 1 + shuffle | 2.95 | 2.51 | 2.83 | 3.03 | 3.01 | 0 |
| **Float32 + NaN + deflate 1 + shuffle (default)** | **1.40** | **1.00** | **1.30** | **1.48** | **1.43** | Float32 rounding |
| Float32 + deflate 4 + shuffle | 1.30 | 0.88 | 1.19 | 1.37 | 1.33 | Float32 rounding |
| Int16 CF packing (not implemented) | 0.71 | 0.62 | 0.70 | 0.70 | 0.64 | S 4·10⁻⁵ psu, T 5·10⁻⁵ °C, U 3·10⁻⁵ m/s |
| BitRound 7 bits (not implemented) | 0.62 | 0.28 | 0.51 | 0.69 | 0.65 | 0.4 % relative, S 0.125 psu |

- Seven maps of one state: about 320 MB before, about 9.5 MB now.
- Write time is at most 0.3 s per map with any of these encodings.
- `test/output.jl` checks that the default maps equal the Float32 rounding of a Float64 output of the same run.

## Caveats

- Readers that sum or average whole maps without masking now get `missing` (NCDatasets) or NaN (xarray), not 0.
- Exact identities against Float64 quantities (supercooling, series vs maps) need `float_type = Float64`.
- Lossy encodings were measured but not implemented. Int16 packing would halve the files again. It needs a fixed range per field, and T/S have none when tracer bounds are off. BitRound is too coarse for S at sizes comparable to Int16.
- BitInformation.jl (99 % information) gave 0–2 keep-bits for U, V, H and melt on these runs, which are not credible tolerances. Pick precision from physical tolerances instead.
- zstd is unavailable without an HDF5 plugin path, and it compresses about as well as deflate here. It is not worth the setup.
- In memory, a full-grid Float32 map of this run is 22.7 MB against 1.4 MB for its active cells. A training pipeline gains more from keeping only active cells than from any file encoding.
