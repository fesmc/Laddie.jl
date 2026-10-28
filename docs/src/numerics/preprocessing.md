# Mask preprocessing

Masks derived from ice-sheet geometry often contain features the layer cannot use:
enclosed ocean pockets, shelf patches with no route to the open ocean, or isolated
single cells. `Grid(mask, z_draft, dx, dy; preprocess = [...])` runs a list of steps,
in order, on a copy of the mask before [domain cropping](domain_cropping.md):

| step | effect |
|---|---|
| [`FillOceanHolesPreprocess`](@ref) | enclosed ocean pockets become land |
| [`FillShelfHolesPreprocess`](@ref) | shelf patches not connected to the open ocean become grounded ice |
| [`FillSmallShelfPatchesPreprocess`](@ref) | shelf patches smaller than `min_size` cells become grounded ice |
| [`FillSmallGroundedPatchesPreprocess`](@ref) | isolated grounded patches (pinning points, rumples) smaller than `min_size` cells become shelf |
| [`MarkGapsPreprocess`](@ref) | open-ocean cells where a reference footprint has ice become gaps (`4`) |

Each step applies the function of the same name, which can also be called on a mask
directly: [`fill_ocean_holes!`](@ref), [`fill_shelf_holes!`](@ref),
[`fill_small_shelf_patches!`](@ref) and [`fill_small_grounded_patches!`](@ref).

`MarkGapsPreprocess` mirrors the `refgeo_Hi > 0` test of the reference code: a
gap is an ice-free cell where the reference geometry had ice, not a topological property
of the mask. Place it after any hole-filling steps, so the gaps it marks are not
reclassified afterwards. What happens in a gap is the [gaps boundary
condition](../physics/gaps.md).

```julia
grid = Grid(mask, z_draft, dx, dy;
            preprocess = [FillShelfHolesPreprocess(), MarkGapsPreprocess(reference_thickness)])
```

## API

```@docs
AbstractPreprocess
FillOceanHolesPreprocess
FillShelfHolesPreprocess
FillSmallShelfPatchesPreprocess
FillSmallGroundedPatchesPreprocess
MarkGapsPreprocess
```
