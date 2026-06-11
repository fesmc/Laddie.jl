# Implementation

The port proceeded in deliberately verifiable phases, each anchored to the one
before so correctness never had to be taken on faith.

## Key design decisions

- **`np.roll` → `circshift`.** Arrays are kept in numpy `[y, x]` orientation so
  `np.roll(·, k, axis)` maps exactly onto `circshift`. The grounded border makes
  periodic wrap harmless. This made the translation near-mechanical.
- **"Object bag" state.** `Model` forwards `m.field` to a `Dict` (via
  `getproperty`/`setproperty!`), mirroring Python's `object.attr` style. This
  kept the line-by-line translation honest and low-risk.
- **Location-typed prognostics.** ``D, U, V, T, S`` are `Var{LX,LY}` carrying
  three leapfrog levels; `rotate!` swaps references (≡ `np.roll(...,-1,axis=0)`),
  honouring the `CGridProto` design while staying allocation-free.

## GPU strategy (KernelAbstractions)

Two complementary mechanisms give device portability:

1. **Backend-generic arrays.** All arrays are allocated through the KA backend and
   moved with `to_backend!`; `Var` fields are `AbstractMatrix` so a CPU↔GPU swap
   works in place. Broadcasting and `circshift` already run on GPU via GPUArrays,
   so the *whole* model runs on any backend.
2. **Fused `@kernel`s** for the hot paths:
   - **elementwise physics**: EOS (`_density_kernel!`, `_freezing_point_kernel!`), the three-equation
     melt (`_three_eq_melt_kernel!`), Robert–Asselin filter (`_robert_asselin_kernel!`), velocity clip
     (`_clamp_kernel!`);
   - **stencil operators**: advection `_convT/U/V_kernel!`, diffusion `_lapT/U/V_kernel!`,
     with periodic neighbour indexing (`_east/_west/_north/_south` ≡ `circshift`);
   - **integration combines**: `_step_thickness/u_momentum/v_momentum/temperature/salinity_kernel!`
     — the `ip_t/jp_t` interpolations and pressure-gradient/Coriolis/drag terms move into kernels.

Each kernel computes the **same scalar expression per cell and same summation
order** as the corresponding broadcast, so there is no floating-point reordering
— the kernelised version is *bit-identical*, not merely close.

## Verification ladder

```
CGridProto unit tests ............................... pass
LaddieKA (CPU) vs Laddie reference ........ max|Δ| = 0.0
fused kernels vs broadcast, op-by-op ...... max|Δ| = 0.0
fused vs broadcast, end-to-end 1-day ...... max|Δ| = 0.0
fused 1-day run ........... mean 24.4 / max 140.7 m/yr
```

## GPU verification (NVIDIA RTX A4000)

```
fused kernels vs broadcast, on-device ..... max|Δ| = 0.0
GPU (fused) vs CPU (fused), 1-day ......... max|Δ| < 1e-11
```

The on-device bit-identity confirms there is no GPU-specific rounding in the
kernel logic.  The tiny GPU–CPU residual (< 10⁻¹¹) arises from different
floating-point operation ordering between CUDA hardware (which uses FMA) and
the CPU backend — not from physics bugs.

**Benchmark** (240 × 40 interior cells, 200 time steps, RTX A4000):

| Backend | ms / step | speedup vs CPU broadcast |
|:-------:|:---------:|:------------------------:|
| CPU broadcast | 13.5 | 1× |
| CPU fused | 4.7 | **2.9×** |
| GPU fused | 2.0 | **6.7×** |

The current test grid has only 9 600 cells, keeping the GPU under-saturated; the
speedup will grow substantially on realistic ice-sheet domains.

To reproduce these numbers on your own hardware, run `benchmark/gpu_vs_cpu.jl`
(sweeps three grid sizes, both dispatch paths, with and without a GPU):

```bash
julia --project=benchmark benchmark/gpu_vs_cpu.jl
```

## Validation against the Python reference

After a 1-day warm ISOMIP+ run from identical initial conditions:

| Field | mean \|Δ\| | relative error |
|:-----:|:----------:|:--------------:|
| D (m) | 0.019 | 0.13 % |
| T (°C) | 0.0015 | 0.07 % |
| S (psu) | 0.00056 | 0.001 % |
| U (mm s⁻¹) | 0.11 | 0.44 % |
| V (mm s⁻¹) | 0.089 | 0.04 % |

End-state melt: **Julia 24.43 m yr⁻¹ / Python log 24.42 m yr⁻¹** (4 sig. figs.).
The residuals are consistent with floating-point rounding between NumPy and
Julia — not physics discrepancies.  See the [Python validation](generated/python_comparison.md)
page for spatial maps.

## The ISOMIP+ test case

A self-contained, idealised ISOMIP+-style channel cavity (`build_isomip`):
a grounded grounding-line wall to the west, an ice draft sloping
``-720 \to -200`` m, an ice front, open ocean beyond, and grounded side/border
walls — driven by the **exact** ISOMIP+ warm/cold ambient forcing.
