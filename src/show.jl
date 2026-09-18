
# ============================================================================
# Compact display — without these methods, printing a Model (or any of its
# sub-structs) at the REPL would dump every matrix it contains.
# The 2-arg `show` is the one-line form (also used inside other printing);
# `text/plain` adds detail for direct REPL display where useful.
# ============================================================================

_sz(a) = join(size(a), "×")
_r(x, d) = round(Float64(x); digits = d)
# A field summarised as its value, or its range when it varies.
function _range_str(a, d)
    lo, hi = extrema(a)
    return lo == hi ? string(_r(lo, d)) : "$(_r(lo, d)) … $(_r(hi, d))"
end

Base.show(io::IO, v::Var{LX,LY,FT}) where {LX,LY,FT} = print(
    io,
    "Var{$(nameof(LX)), $(nameof(LY))}($FT, $(_sz(v.present)), ",
    "levels: past/present/future)",
)

Base.show(io::IO, s::State{FT}) where {FT} =
    print(io, "State{$FT}: D, U, V, T, S — 3-level Vars of $(_sz(s.D.present))")

function Base.show(io::IO, c::Cache{FT}) where {FT}
    nmat = count(fn -> getfield(c, fn) isa AbstractMatrix, fieldnames(typeof(c)))
    print(io, "Cache{$FT}: $nmat scratch/diagnostic arrays of $(_sz(c.melt))")
end

function Base.show(io::IO, s::IOState{FT}) where {FT}
    rd = isempty(s.rundir) ? "I/O disabled" : "rundir = \"$(s.rundir)\""
    print(io, "IOState{$FT}: $(s.time_index) output slices, $rd")
end

# Generic one-liner for a profile forcing.  extrema/length are reductions, so this
# is GPU-safe (no scalar indexing).
function Base.show(io::IO, f::AbstractOceanForcing)
    bounds(a, d) = join(_r.(extrema(a), d), ", ")
    print(
        io,
        "$(nameof(typeof(f))){$(eltype(f.Tz))}: $(length(f.z))-point profile, ",
        "z ∈ [$(join(extrema(f.z), ", "))] m, T ∈ [$(bounds(f.Tz, 3))] °C, ",
        "S ∈ [$(bounds(f.Sz, 3))] psu",
    )
end

# The ice forcing is one field, so summarise it as a range rather than printing a
# whole matrix; a uniform field collapses to the single value.
Base.show(io::IO, f::AbstractIceForcing) =
    print(io, "$(nameof(typeof(f)))(T_ice_base = $(_range_str(f.T_ice_base, 2)) °C)")

Base.show(io::IO, f::CavityForcing) = print(io, f.ocean, "\n  ice: ", f.ice)

Base.show(io::IO, g::Grid{FT}) where {FT} = print(
    io,
    "Grid{$FT}: $(g.Nx)×$(g.Ny) cells ($(g.Nx - 2)×$(g.Ny - 2) interior), ",
    "dx = $(g.dx) m, dy = $(g.dy) m",
)

function Base.show(io::IO, ::MIME"text/plain", g::Grid)
    show(io, g)
    n(v) = count(==(v), g.mask)
    print(
        io,
        "\n  cells: $(n(3)) shelf, $(n(4)) gap, $(n(2)) grounded, $(n(0)) ocean, ",
        "$(n(1)) land/border",
    )
    r, c = g.crop
    (length(r), length(c)) == g.input_size || print(
        io,
        "\n  cropped from $(join(g.input_size, "×")) ",
        "(rows $(first(r)):$(last(r)), columns $(first(c)):$(last(c)))",
    )
end

Base.show(io::IO, g::Geometry{FT}) where {FT} = print(
    io,
    "Geometry{$FT}: $(count(>(0), g.tmask)) active cells ",
    "($(count(>(0), g.imask)) under ice)",
)

_schemes(p::Params) = join(
    (
        nameof(typeof(x)) for x in (
            p.entrainment,
            p.melting,
            p.convection_scheme,
            p.lateral_viscosity,
            p.front_pressure,
        )
    ),
    " + ",
)

Base.show(io::IO, p::Params{FT}) where {FT} = print(io, "Params{$FT}($(_schemes(p)))")

function Base.show(io::IO, ::MIME"text/plain", p::Params{FT}) where {FT}
    println(io, "Params{$FT}:")
    scal = [(fn, getfield(p, fn)) for fn in fieldnames(Params) if getfield(p, fn) isa Number]
    for chunk in Iterators.partition(scal, 4)
        println(io, "  ", join((rpad("$k = $v", 22) for (k, v) in chunk), " "))
    end
    println(io, "  entrainment    = ", p.entrainment)
    println(io, "  melt           = ", p.melting)
    println(io, "  convection     = ", p.convection_scheme)
    println(io, "  max layer D    = ", p.max_layer_thickness)
    println(io, "  lat. viscosity = ", p.lateral_viscosity)
    println(io, "  lap. weights   = ", p.laplacian_weights)
    println(io, "  front pressure = ", p.front_pressure)
    print(io, "  coriolis       = ", p.coriolis)
end

Base.show(io::IO, b::BoundaryConditions) = print(
    io,
    "BoundaryConditions(open ocean = $(nameof(typeof(b.open_ocean))), ",
    "grounding line = $(nameof(typeof(b.grounding_line))), ",
    "land = $(nameof(typeof(b.land))), gaps = $(nameof(typeof(b.gaps))), ",
    "wall advection = $(nameof(typeof(b.wall_advection))))",
)

Base.show(io::IO, ts::AdaptiveDt) = print(
    io,
    "AdaptiveDt(cfl_target = $(ts.cfl_target), q = $(ts.q), ncheck = $(ts.ncheck), ",
    "dt ∈ [$(ts.dtmin), $(ts.dtmax)])",
)

_backend_name(m::Model) = nameof(typeof(KA.get_backend(getfield(m, :grid).z_draft)))

function Base.show(io::IO, m::Model{FT}) where {FT}
    g = getfield(m, :grid)
    ocean = getfield(m, :forcing).ocean
    print(
        io,
        "Model{$FT} on $(_backend_name(m)): $(g.Nx - 2)×$(g.Ny - 2) interior, ",
        "$(nameof(typeof(ocean))) forcing",
    )
end

function Base.show(io::IO, ::MIME"text/plain", m::Model{FT}) where {FT}
    g = getfield(m, :grid)
    println(io, "Model{$FT} on $(_backend_name(m))")
    println(
        io,
        "  grid:    $(g.Nx - 2)×$(g.Ny - 2) interior cells, dx = $(g.dx) m, ",
        "dy = $(g.dy) m, $(count(==(3), g.mask)) shelf cells",
    )
    println(io, "  forcing: ", getfield(m, :forcing))
    println(io, "  params:  ", _schemes(getfield(m, :params)))
    print(io, "  boundary: ", getfield(m, :boundary))
end

_days(seconds, spd) = _r(seconds / Float64(spd), 3)

Base.show(io::IO, c::Clock{FT}) where {FT} = print(
    io,
    "Clock{$FT}(time = $(_r(c.time, 1)) s, iteration = $(c.iteration), dt = $(c.dt) s)",
)

Base.show(io::IO, o::OutputConfig) = print(
    io,
    o.saveday > 0 ?
    "OutputConfig: every $(o.saveday) d → $(joinpath(o.resultdir, o.name))" :
    "OutputConfig: disabled (saveday = 0)",
)

Base.show(io::IO, sim::Simulation{M,FT}) where {M,FT} = print(
    io,
    "Simulation{$FT} on $(_backend_name(sim.model)) at day ",
    "$(_days(sim.clock.time, sim.model.seconds_per_day)), dt = $(sim.clock.dt) s",
)

function Base.show(io::IO, ::MIME"text/plain", sim::Simulation{M,FT}) where {M,FT}
    m = sim.model
    c = sim.clock
    println(io, "Simulation{$FT} on $(_backend_name(m))")
    println(io, "  model:   ", m)
    println(
        io,
        "  clock:   day $(_days(c.time, m.seconds_per_day)), iteration $(c.iteration), ",
        "dt = $(c.dt) s",
    )
    println(io, "  stepper: ", sim.tstep, ", ", sim.cfl, ", Robert–Asselin ν = ", sim.nu)
    println(io, "  stop:    ", sim.stop)
    print(io, "  output:  ", sim.output)
end
