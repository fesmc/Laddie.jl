
"""
    Var{LX,LY}

Three leapfrog levels (`past`, `present`, `future`) for a prognostic field
located at staggering position `(LX, LY)` on the C-grid.

`AbstractMatrix` fields allow the same `Var` to hold CPU or GPU arrays;
`to_backend!` swaps them in-place without reallocating the struct.
"""
mutable struct Var{LX,LY,FT}
    past::AbstractMatrix{FT}
    present::AbstractMatrix{FT}
    future::AbstractMatrix{FT}
end
Var(::Type{LX}, ::Type{LY}, ::Type{FT}, ny, nx) where {LX,LY,FT} =
    Var{LX,LY,FT}(zeros(FT, ny, nx), zeros(FT, ny, nx), zeros(FT, ny, nx))
Var(::Type{LX}, ::Type{LY}, ny, nx) where {LX,LY} = Var(LX, LY, Float64, ny, nx)

function rotate!(v::Var)
    v.past, v.present, v.future = v.present, v.future, v.past
    return v
end
