# Preprocessing

A run starts from raw geometry and ocean data, and the model only sees what preprocessing leaves
it. Real data carries artefacts: one-pixel ocean holes inside a shelf, shelf patches with no route
to the open ocean, pinning points of a single cell, ocean profiles with gaps. Most are harmless
to a forward run. Some are not, and for a differentiated run they can decide whether the gradient
exists at all.

!!! warning "Preprocess the geometry before differentiating a run"
    A geometry artefact that a forward run absorbs can make the derivative of the same run grow
    without bound. In a Crosson–Dotson run built from BedMachine, a **one-pixel ocean hole inside
    the shelf** drove a relaxation oscillation of the layer in the neighbouring cell (thickness
    swinging between about 100 m and 4 m, with spikes in entrainment) and made the tangent grow by
    one e-fold every ~35 steps. Forward mode, reverse mode and ForwardDiff all agreed, so the
    automatic differentiation was right; the finite difference of the window-mean melt, which
    averages the oscillation away, stayed smooth. Filling the hole with
    [`FillOceanHolesPreprocess`](@ref) removed the blow-up and left the melt unchanged (9.650 against
    9.652 m yr⁻¹ at day 20).

    Nothing is preprocessed by default: `Grid(...; preprocess = [...])` runs only the steps you
    list. Before computing a gradient, run the [recipe below](#Recommended-recipe), and
    [check the tangent](#Checking-a-geometry).

## The pipeline

Each stage has its own page; this one says what they are for and in which order to use them.

| stage | what it does | where |
|---|---|---|
| geometry ingestion | the 4-class mask, the ice-base depth and the bed from BedMachine-style arrays | [`build_laddie_mask`](@ref), [`ice_base_depth`](@ref), [`bed_elevation`](@ref) |
| mask preprocessing | removes holes and small patches, marks [ice-shelf gaps](physics/gaps.md) | [Mask preprocessing](numerics/preprocessing.md) |
| domain cropping | cuts the arrays to the cavity | [Domain cropping](numerics/domain_cropping.md) |
| ice-base slope | derives the slope from the draft, mask-aware | [Ice-base slope](numerics/ice_slope.md) |
| ocean forcing | brings a 3D ocean field onto the cells, and fills its missing values | [`regrid_ocean`](@ref), [`fill_ocean!`](@ref), [`front_ocean_forcing`](@ref) |

The first three act on the [`Grid`](@ref) and run in that order: ingestion, then the `preprocess`
list (on a copy of the mask), then cropping. Everything is size-preserving until the crop.

### Geometry ingestion

[`build_laddie_mask`](@ref) classifies each cell as open ocean (`0`), land (`1`), grounded ice (`2`)
or floating shelf (`3`) from the bed and the ice thickness, and adds the one-cell border ring of
land. Ice-free bedrock is land, not ocean, so a rock island inside a shelf is a wall and not an
open boundary in the middle of the cavity. Gaps (`4`) are not derived here; mark them with
[`MarkGapsPreprocess`](@ref). [`ice_base_depth`](@ref) and [`bed_elevation`](@ref) give the draft
and the bed in the model's sign convention.

### Cleaning the mask

The steps of the `preprocess` list, each of which can also be called on a mask directly, and what
each removes:

| step | removes | matters because |
|---|---|---|
| [`FillOceanHolesPreprocess`](@ref) | ocean pockets not connected to the open ocean; they become land | noisy topography leaves them fully surrounded by ice, where they create spurious ice-front dynamics and numerical instabilities |
| [`FillShelfHolesPreprocess`](@ref) | shelf patches not connected to the open ocean | a shelf with no outlet is a closed cavity the plume cannot leave |
| [`FillSmallShelfPatchesPreprocess`](@ref) | shelf patches below `min_size` cells | too small to carry a circulation |
| [`FillSmallGroundedPatchesPreprocess`](@ref) | grounded patches below `min_size` cells | single-cell pinning points and rumples |
| [`MarkGapsPreprocess`](@ref) | nothing: marks ice-free cells of a reference footprint as gaps | last, so the gaps are not reclassified |

Connectivity-based cleaning treats gaps as part of the shelf, so a shelf beyond a gap is not
grounded by mistake.

### Cropping

[`MinRectangleDomainCropping`](@ref), the default, crops every full-domain field to the smallest
rectangle around the cavity, with a margin of 4 cells. Cropping does not change the result, only
the cost, and is what makes a realistic domain affordable on a GPU.

### Ocean forcing

An ocean model without ice-shelf cavities gives no profile inside the cavity.
[`front_ocean_forcing`](@ref) assigns each cavity cell the water column at the ice front that feeds
it. When the ocean grid differs from the model's, [`regrid_ocean`](@ref) interpolates it onto the
cells, leaving `NaN` where there is no water, and [`fill_ocean!`](@ref) fills those so that every
column is complete: a column extends flat beyond its own water, and one with no water takes the
nearest column that has some. A `NaN` left in the forcing poisons the run, and its gradient with it.

## Recommended recipe

```julia
mask = build_laddie_mask(bed, thickness)
zb   = ice_base_depth(bed, thickness)

grid = Grid(mask, zb; x, y,
            preprocess = [FillOceanHolesPreprocess(),
                          FillShelfHolesPreprocess(),
                          FillSmallShelfPatchesPreprocess(),
                          FillSmallGroundedPatchesPreprocess()])   # then MarkGapsPreprocess, if you have gaps
```

The order is the cleaning steps first, so that each sees the mask the previous one left, and
`MarkGapsPreprocess` last, so that the gaps it marks are not reclassified. The thresholds (10 cells
by default) are a choice about resolution: ten cells at 500 m are not ten cells at 4 km.

## Checking a geometry

Preprocessing is a mask operation, so it cannot tell you whether a *remaining* feature is harmful.
Look for it where it shows: in the derivative.

- Run the [ForwardDiff](API_ad.md) tangent for a few hundred steps with respect to one smooth
  parameter, and look at where the largest layer-thickness response `max |dH|` sits. A response
  that is spatially concentrated on one or two cells and grows steadily, while the rest of the
  shelf responds smoothly, is a geometry artefact.
- Compare with a finite difference of the window mean. A finite difference that stays smooth
  while the tangent grows is the signature of the oscillation above: the AD is right and the
  mean hides it.
- Inspect the mask at that cell. A one-cell hole or a one-cell pinning point is the usual cause.

If the geometry is clean and the gradient is still large over a long window, the cause may be the
switches in the model rather than the data; see [Kinks](numerics/kinks.md).
