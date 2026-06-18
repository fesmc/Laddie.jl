# 4 · Configuration reference

`build_from_config("config.toml")` reads a TOML file that controls every
aspect of a run. This page lists every recognised key, its type, default
value (if optional), and what it does.

All physical parameters are converted to the floating-point type `FT` chosen
at call time (default `Float64`).

---

## `[Run]`

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `name` | String | yes | Label used for the output directory name. |
| `days` | Float | yes | Total integration time (days). |

---

## `[Time]`

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `dt` | Float | yes | Time step (s). ISOMIP+ default: `210`. |
| `restday` | Float | yes | Interval between restart saves (days). |
| `saveday` | Float | yes | Interval between field output saves (days). |
| `diagday` | Float | yes | Interval between log-file diagnostics (days). |

---

## `[Geometry]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `filename` | String | — | Path to the NetCDF geometry file. |
| `maskoption` | String | — | `"ISOMIP"` (separate `groundedMask`/`floatingMask` variables) or `"BM"` (single `mask` variable, BedMachine convention: 0=ocean, 1/2=grounded, 3=ice shelf). |
| `geomyear` | Int | `0` | Year index to read from a time-varying geometry file. |
| `coarsen` | Int | `1` | Coarsening factor applied to the raw grid (1 = no coarsening). |
| `calvthresh` | Float | `0.0` | Remove ice shelf cells whose draft is shallower than `−rhoi/rho0_seawater * calvthresh` m (0 = disabled). |
| `cutdomain` | Bool | `true` | Crop the domain to the minimal bounding box around floating-ice cells before adding the border ring. |

The geometry NetCDF file must contain `x` and `y` coordinate vectors and one
of the following ice-draft variables: `draft`, `Hib`, `zb`, `lowerSurface`.
If none are found, the code attempts `surface − thickness`.

---

## `[Forcing]`

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `option` | String | yes | Forcing profile type: `"isomip"`, `"linear"`, `"linear2"`, `"tanh"`, or `"file"`. |

Depending on `option`, additional keys are required:

### `option = "isomip"`

| Key | Type | Description |
|-----|------|-------------|
| `isomipcond` | String | `"warm"` (T = +1 °C at depth) or `"cold"` (T = −1.9 °C at depth). |

### `option = "linear"` or `"linear2"`

| Key | Type | Description |
|-----|------|-------------|
| `z0` | Float | Reference depth (m, negative) at which the deep values are reached. |
| `S0` | Float | Surface salinity (psu). |
| `S1` | Float | Deep salinity (psu). |
| `T1` | Float | Deep temperature (°C). |

### `option = "tanh"`

All keys from `linear` plus:

| Key | Type | Description |
|-----|------|-------------|
| `z1` | Float | Depth scale of the tanh transition (m). |
| `drho0` | Float | Density perturbation amplitude for the `√|z|` salinity term (kg m⁻³). |

### `option = "file"`

| Key | Type | Description |
|-----|------|-------------|
| `filename` | String | NetCDF file with `z`, `T`, `S` variables (1-D depth profiles). |
| `filename_T` | String | Alternative: separate file for temperature (`z`, `temperature`). |
| `filename_S` | String | Alternative: separate file for salinity (`z`, `salinity`). |

Profiles are resampled to a 1-m depth grid covering −5000 m to −1 m.

---

## `[Options]`

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `slip` | Float | yes | Partial-slip factor at grounding-line walls (0 = free-slip, 1 = no-slip). |
| `convop` | Int | yes | Convection scheme: `0` = `ClampDensity`, `1` = `ResetToAmbient`, `2` = `RelaxToAmbient`. |
| `boundop` | Int | yes | Open-boundary condition: `1` = `ZeroGradientInflow`, any other value = `NoInflow`. |
| `usegamtfix` | Bool | yes | `true` = `FixedGamT`; `false` = `TurbulentGamT`. |
| `border_N` | Int | `1` | Mask value for the north border ring (1 = grounded). |
| `border_S` | Int | `1` | Mask value for the south border ring. |
| `border_E` | Int | `1` | Mask value for the east border ring. |
| `border_W` | Int | `1` | Mask value for the west border ring. |

---

## `[Convection]`  *(optional)*

Only relevant when `convop = 0` or `convop = 1` (`ClampDensity` /
`ResetToAmbient`), or `convop = 2` (`RelaxToAmbient`).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mindrho` | Float | `0.005` | Minimum density contrast δρ (kg m⁻³); used by `ClampDensity` and `ResetToAmbient`. |
| `convection_time` | Float | `10000.0` | Relaxation timescale (s); used by `RelaxToAmbient`. |

---

## `[Parameters]`

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `u_tide` | Float | yes | Tidal velocity amplitude used in u★ (m s⁻¹). |
| `T_i` | Float | yes | Ice temperature for the latent-heat correction (°C). |
| `f` | Float | yes | Coriolis parameter (s⁻¹; negative in Southern Hemisphere). |
| `rho_freshwater` | Float | yes | Fresh-water density (kg m⁻³; typically `1000`). |
| `rho0_seawater` | Float | yes | Reference ocean density (kg m⁻³; typically `1028`). |
| `rho_ice` | Float | yes | Ice density (kg m⁻³; typically `910`). |
| `C_d` | Float | yes | Bottom drag coefficient. |
| `C_d_top` | Float | yes | Top (ice–ocean) drag coefficient used in u★. |
| `A_h` | Float | yes | Laplacian viscosity (m² s⁻¹). |
| `K_h` | Float | yes | Laplacian diffusivity for T and S (m² s⁻¹). |
| `entrainment` | String | yes | `"Lambert"` (default), `"Gaspar"`, or `"Holland"` — selects the entrainment parameterisation. `"Lambert"` is the reference LADDIE form; `"Gaspar"` is the literal Eq. 14. |
| `max_detrainment` | Float | yes | Maximum detrainment rate (m s⁻¹). |
| `D_min` | Float | yes | Minimum layer thickness enforced by extra entrainment (m). |
| `v_cut` | Float | `1.414` | Velocity cutoff — U and V are clipped to ±`v_cut` m s⁻¹. |
| `mu` | Float | Lambert/Gaspar only | Entrainment efficiency parameter for `LambertEntrainment` / `GasparEntrainment`. |
| `cl` | Float | `0.01775` | Drag coefficient for `HollandEntrainment`. |
| `gamTfix` | Float | FixedGamT only | Fixed heat transfer coefficient γ_T (used when `usegamtfix = true`). |

---

## `[EOS]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `alpha` | Float | — | Thermal expansion coefficient (°C⁻¹). |
| `beta` | Float | — | Haline contraction coefficient (psu⁻¹). |
| `l1` | Float | `-5.73e-2` | Liquidus slope (°C psu⁻¹). |
| `l2` | Float | `8.32e-2` | Liquidus intercept (°C). |
| `l3` | Float | `7.61e-4` | Liquidus pressure coefficient (°C m⁻¹). |

---

## `[Constants]`  *(all optional)*

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `g` | Float | `9.81` | Gravitational acceleration (m s⁻²). |
| `L` | Float | `3.34e5` | Latent heat of fusion for ice (J kg⁻¹). |
| `c_p` | Float | `3.974e3` | Specific heat capacity of sea water (J kg⁻¹ K⁻¹). |
| `c_i` | Float | `2009.0` | Specific heat capacity of ice (J kg⁻¹ K⁻¹). |
| `Pr` | Float | `13.8` | Prandtl number (TurbulentGamT only). |
| `Sc` | Float | `2432.0` | Schmidt number (TurbulentGamT only). |
| `nu0` | Float | `1.95e-6` | Molecular kinematic viscosity (m² s⁻¹; TurbulentGamT only). |

---

## `[Numerics]`

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `nu` | Float | yes | Robert–Asselin filter strength (0–1; ISOMIP+ default `0.8`). |

---

## `[Initialisation]`

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `fromrestart` | Bool | yes | `true` = initialise from a restart file; `false` = cold start. |
| `restartfile` | String | if `fromrestart` | Path to the NetCDF restart file. |
| `D_init` | Float | `10.0` | Initial layer thickness for cold start (m). |
| `dT_init` | Float | `0.0` | Initial temperature offset from ambient (°C). |
| `dS_init` | Float | `-0.1` | Initial salinity offset from ambient (psu). |

---

## `[Directories]`  *(optional)*

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `results` | String | `"./output/"` | Root directory for output files. |
| `forcenewdir` | Bool | `true` | Create a new timestamped subdirectory inside `results` for each run. |

---

## `[Filenames]`  *(optional)*

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `logfile` | String | `"log.txt"` | Name of the diagnostic log file written inside the run directory. |

---

## `[Output]`  *(all optional)*

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `save_Ut` | Bool | `true` | Save cell-centre U (interpolated). |
| `save_Uu` | Bool | `false` | Save raw face-value U. |
| `save_Vt` | Bool | `true` | Save cell-centre V (interpolated). |
| `save_Vv` | Bool | `false` | Save raw face-value V. |
| `save_D` | Bool | `true` | Save layer thickness D. |
| `save_T` | Bool | `true` | Save layer temperature T. |
| `save_S` | Bool | `true` | Save layer salinity S. |
| `save_melt` | Bool | `true` | Save basal melt rate ṁ. |
| `save_entr` | Bool | `false` | Save entrainment rate ė. |
| `save_ent2` | Bool | `false` | Save extra entrainment (minimum-D correction). |
| `save_detr` | Bool | `false` | Save detrainment rate. |
| `save_Tbase` | Bool | `false` | Save ice-base temperature T_b. |
| `save_Tamb` | Bool | `false` | Save ambient temperature T_a at plume-base depth. |
| `save_gammaT` | Bool | `false` | Save turbulent heat transfer coefficient γ_T. |
| `save_mask` | Bool | `true` | Save the cavity mask. |
| `save_zb` | Bool | `true` | Save the ice-shelf draft z_b. |

---

## Minimal example

```toml
[Run]
name   = "isomip_warm"
days   = 30.0

[Time]
dt      = 210.0
restday = 10.0
saveday = 1.0
diagday = 0.5

[Geometry]
filename   = "isomip_geometry.nc"
maskoption = "ISOMIP"

[Forcing]
option     = "isomip"
isomipcond = "warm"

[Options]
slip       = 1.0
convop     = 1
boundop    = 1
usegamtfix = true

[Initialisation]
fromrestart = false
D_init       = 10.0

[Parameters]
u_tide   = 0.01
T_i      = -25.0
f       = -1.37e-4
rho_freshwater   = 1000.0
rho0_seawater    = 1028.0
rho_ice    = 910.0
C_d      = 2.5e-3
C_d_top   = 1.1e-3
A_h      = 6.0
K_h      = 1.0
entrainment  = "Gaspar"
mu      = 2.5
max_detrainment = 0.5
D_min    = 1.0
gamTfix = 1.8e-4

[EOS]
alpha = 3.733e-5
beta  = 7.843e-4

[Numerics]
nu = 0.8
```
