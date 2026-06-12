
# Arakawa C-grid location vocabulary: a prognostic's staggering position is
# carried in its type as (LX, LY), e.g. U is Var{Face, Center}.  No method
# currently dispatches on these — they document the staggering and keep the
# door open for location-dispatched operators.
abstract type AbstractLoc end
struct Center <: AbstractLoc end
struct Face <: AbstractLoc end

"""
$(TYPEDSIGNATURES)

Three leapfrog levels (`past`, `present`, `future`) for a prognostic field
located at staggering position `(LX, LY)` on the C-grid.

`A` is the concrete array type (`Matrix{FT}` on CPU, `CuArray{FT,2}` on GPU).
`to_backend` returns a new `Var` with a different `A`; the struct itself is
never mutated to hold a different array type.
"""
mutable struct Var{LX, LY, FT, A<:AbstractMatrix{FT}}
    past   ::A
    present::A
    future ::A
end

Var(::Type{LX}, ::Type{LY}, ::Type{FT}, ny, nx) where {LX, LY, FT} =
    Var{LX, LY, FT, Matrix{FT}}(zeros(FT, ny, nx), zeros(FT, ny, nx), zeros(FT, ny, nx))

Var(::Type{LX}, ::Type{LY}, ny, nx) where {LX, LY} = Var(LX, LY, Float64, ny, nx)

function rotate!(v::Var)
    v.past, v.present, v.future = v.present, v.future, v.past
    return v
end
