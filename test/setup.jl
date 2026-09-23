using Laddie
using Test
using KernelAbstractions
using Aqua
using ForwardDiff
import CUDA

# Shared setup for every test file: precision, GPU detection, fake forcing types and
# test helpers.  Included once by runtests.jl.  To run a single area during
# development, activate the test environment (e.g. with TestEnv.jl) and include this
# file followed by that area's file, from the package root:
#
#     using TestEnv; TestEnv.activate("Laddie")
#     include("test/setup.jl"); include("test/forcing.jl")

FT = Float64

# Detect GPU: use CUDABackend if a functional CUDA device is present.
const gpu_backend = CUDA.functional() ? CUDA.CUDABackend() : nothing

# Fake forcing types for the property-forwarding collision guard tests
# (type definitions must live at top level, not inside a @testset).
struct CollidingForcing <: Laddie.AbstractOceanForcing
    Tz::Vector{Float64}
    Sz::Vector{Float64}
    z::Vector{Float64}
    dz::Float64
    z0::Float64
    melt::Float64   # collides with Cache.melt
end

struct ReservedNameForcing <: Laddie.AbstractOceanForcing
    Tz::Vector{Float64}
    Sz::Vector{Float64}
    z::Vector{Float64}
    dz::Float64
    z0::Float64
    nx::Int         # collides with the reserved Model property `nx`
end

struct CollidingIceForcing <: Laddie.AbstractIceForcing
    T_ice_base::Matrix{Float64}
    Tz::Vector{Float64}   # collides with the ocean forcing's profile
    CollidingIceForcing(T) = new(T, Float64[])
end

# Wall-face indicators of the whole wall mask (land ∪ grounded ice), built with
# the same stencil as the model's grounding-line (gl??) and land (lnd??) ones,
# which partition it.  The model itself only keeps the partition.
function wall_faces(g)
    grd = g.grd
    o = one(eltype(grd))
    xm1, xp1, ym1, yp1 = Laddie.xm1, Laddie.xp1, Laddie.ym1, Laddie.yp1
    (grdNu = o .- ym1((o .- grd) .* (o .- xm1(grd))),
     grdSu = o .- yp1((o .- grd) .* (o .- xm1(grd))),
     grdEv = o .- xm1((o .- grd) .* (o .- ym1(grd))),
     grdWv = o .- xp1((o .- grd) .* (o .- ym1(grd))))
end
