# Output and restarts

A [`Simulation`](@ref) writes its files as its [`OutputConfig`](@ref) says, into
`joinpath(resultdir, name)`. A plain `OutputConfig()` writes nothing; set `saveday > 0` to
write

- time-averaged maps in `output.nc`, every `saveday` days;
- a line of diagnostics in `log.txt`, every `diagday` days;
- a JLD2 restart every `restday` days, and one at the end of every [`run!`](@ref).

Event times are absolute on the simulation clock, so a simulation advanced by several
`run!` calls writes on the same cadence as one advanced by a single call.

## Maps

`OutputConfig(; fields)` names the variables of `output.nc`. Time averages over each
`saveday` window, or with `meanday` over its last `meanday` days only (e.g. the state at the
end of a long run, `saveday = 100, meanday = 2`):

| name | field | units |
|------|-------|-------|
| `:Ut`, `:Vt` | x- and y-velocity averaged onto T-points | m s⁻¹ |
| `:Uu`, `:Vv` | x- and y-velocity on their own u- and v-points | m s⁻¹ |
| `:H`, `:T`, `:S` | layer thickness, temperature, salinity | m, °C, psu |
| `:melt` | basal melt rate | m yr⁻¹ |
| `:entr`, `:detr` | entrainment and detrainment rates | m yr⁻¹ |
| `:ent2` | the extra entrainment that keeps `H ≥ H_min` | m yr⁻¹ |
| `:detr_cap` | the detrainment that keeps `H` under `max_layer_thickness` | m yr⁻¹ |
| `:frazil` | frazil-ice formation, freshwater equivalent (zero unless `RelaxedFrazilIce`) | m yr⁻¹ |
| `:Tbase`, `:Tamb` | temperature at the ice–ocean interface, ambient temperature at the layer base | °C |
| `:gammaT` | turbulent heat-exchange velocity | m s⁻¹ |
| `:ustar` | friction velocity at the ice base | m s⁻¹ |
| `:drho` | reduced density contrast with the ambient water | – |
| `:convective_instability` | fraction of time a cell was convectively unstable | – |
| `:supercooling` | ``T - T_f(S, z_b)``, the layer temperature above its freezing point at the ice base (negative where supercooled) | °C |

Static fields, written once: `:mask` (the mask and the margin flags `at_isf`, `at_grl`,
`at_lnd`, `at_gap`) and `:z_draft` (the ice draft, m).

The maps are written as `float_type` (`Float32` by default, `Float64` on request), with the
fill value NaN on every cell outside the active cells (`tmask`), and compressed losslessly
(deflate level 1 with shuffle). NetCDF readers return these cells as missing (NCDatasets)
or NaN (xarray, ncview).

## Time series

Off by default. With `tsday` set, every `tsday` days `output.nc` gets the means over the
ice-covered cells (`imask`) of the fields averaged over that window, on their own time
axis `ts_time`:

| name | series | units |
|------|--------|-------|
| `ts_H`, `ts_T`, `ts_S` | mean layer thickness, temperature, salinity | m, °C, psu |
| `ts_tf` | mean thermal forcing ``T - T_f(S, z_b)`` (the mean of `:supercooling`) | °C |
| `ts_u_norm` | mean speed on T-points | m s⁻¹ |
| `ts_mean_melt` | mean net basal melt rate, `melt − frazil` | m yr⁻¹ |
| `ts_total_melt` | total net basal melt, freshwater mass | Gt yr⁻¹ |
| `ts_refreezing` | total refreezing, `max(−melt, 0) + frazil`, freshwater mass | Gt yr⁻¹ |
| `ts_dt` | mean time step over the window (its length over its number of steps) | s |

As in [`meltstats`](@ref), `ts_total_melt + ts_refreezing` is the melting alone. The
fields are accumulated on the device every step, like the maps; the spatial means are
reduced only once per `tsday`. Not available under a [`ReactantBackend`](@ref):
`to_backend` rejects a simulation with `tsday` set.

## Melt statistics

[`meltstats`](@ref) reduces the current state to `max_melt`, `mean_melt`, `max_speed`,
`total_melt` and `refreezing`, on the model's device (on a CPU copy under the Reactant
backend).

- The **net basal melt** is the interface melt rate `melt` (negative where the base
  refreezes) minus the frazil formation `frazil`: the basal mass balance of the ice.
  `refreezing` is the ice formed, ``\sum (\max(-\dot m, 0) + \dot f)``, so
  `total_melt + refreezing` is the melting alone.
- Rates are freshwater equivalent, the unit the model integrates; multiply by
  `rho_freshwater / rho_ice` for ice equivalent. The mass fluxes are
  ``\sum \text{rate}\,\Delta x\,\Delta y\,\rho_\text{fw}``, the same in either unit.
- The melt statistics average over the ice-covered cells (`imask`), not the active cells
  (`tmask`). Under [`ConnectedGapsBC`](@ref) gap cells are active but ice-free and melt
  exactly zero, so `tmask` would dilute the mean by the gap area. LADDIE v2 integrates over
  its `domain_a`, gaps included; without gaps the two coincide. `max_speed` stays on
  `tmask`, since the layer does flow through gaps.

## Restarts

A restart file holds the three leapfrog levels of `H`, `U`, `V`, `T` and `S`, the time,
the step that separates the levels and the active cells, as CPU arrays at the model's
precision; `restart_latest.jld2` is always the newest. `init_from_restart!`, or the
`restart` keyword of [`Simulation`](@ref), loads it onto any backend and precision.

**Continuation.** On the same grid a restart continues the run exactly: a run of 2 days
equals one of 1 day, restarted, plus 1 day, to the last bit. This needs the same `dt`.
[`AdaptiveTimeStepper`](@ref) resumes at the file's `dt`. [`FixedTimeStepper`](@ref), and a
stepper of your own, keeps the `dt` the simulation was built with; when it differs from the
file's, the leapfrog restarts with one forward step at the new `dt`, and the log says so.
A file written before the `dt` was stored restarts the leapfrog the same way. A file that
stores the active cells must match the model's.

**Refinement.** With `refine = f > 1`, the file is the restart of a run whose cells are
`f` times coarser on the same full domain:

- each coarse cell holds `f × f` fine ones; the fine domain, border ring included, is
  `f .* (coarse size .- 2) .+ 2` cells, or up to `f - 1` more at the far end of an axis
  (a grid coarsened by whole blocks drops the remainder of its source grid), and those
  extra cells keep the state the simulation was built with;
- each fine cell takes `H`, `T` and `S` of the coarse cell it lies in, where that cell was
  active, and each fine velocity point the cell-centred velocity of that coarse cell (the
  mean of its two faces), on the fine velocity masks;
- fine cells whose coarse cell was not active (a finer shelf reaches further) keep their
  state, so refine onto a simulation started from rest; with `refine_fill = :nearest` they
  take `H`, `T` and `S` of the nearest cell that has an active coarse cell, counted in steps
  through the active cells (velocities stay zero), which avoids a layer at rest next to a
  spun-up one;
- `dt` is the restart's divided by `f`, and the leapfrog restarts with a forward step;
- the two runs may crop their domains differently; the coarse grid is read from the file.

Pass `refine` and `refine_fill` to [`Simulation`](@ref) along with `restart`, so the
output schedule starts at the restart's time. Calling `init_from_restart!` on a
simulation built with output leaves the schedule at its start, and the first steps then
write every save, restart and series point in between (Laddie warns).
