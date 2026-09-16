
# ============================================================================
# Compact display — without these methods, printing a Model (or any of its
# sub-structs) at the REPL would dump every matrix it contains.
# The 2-arg `show` is the one-line form (also used inside other printing);
# `text/plain` adds detail for direct REPL display where useful.
# ============================================================================

Base.show(io::IO, v::Var{LX,LY,FT}) where {LX,LY,FT} = print(
    io,
    "Var{",
    nameof(LX),
    ", ",
    nameof(LY),
    "}(",
    FT,
    ", ",
    size(v.present, 1),
    "×",
    size(v.present, 2),
    ", levels: past/present/future)",
)

function Base.show(io::IO, s::State{FT}) where {FT}
    nx, ny = size(s.D.present)
    print(io, "State{", FT, "}: D, U, V, T, S — 3-level Vars of ", nx, "×", ny)
end

function Base.show(io::IO, c::Cache{FT}) where {FT}
    nmat = count(fn -> getfield(c, fn) isa AbstractMatrix, fieldnames(typeof(c)))
    nx, ny = size(c.melt)
    print(io, "Cache{", FT, "}: ", nmat, " scratch/diagnostic arrays of ", nx, "×", ny)
end

function Base.show(io::IO, s::IOState{FT}) where {FT}
    rd = isempty(s.rundir) ? "I/O disabled" : string("rundir = \"", s.rundir, "\"")
    print(io, "IOState{", FT, "}: ", s.time_index, " output slices, ", rd)
end

# Generic one-liner for any forcing; relies only on the Tz/Sz/z profile fields
# that the model requires of every AbstractOceanForcing.  extrema/length are
# reductions, so this is GPU-safe (no scalar indexing).
function _show_forcing(io::IO, f::AbstractOceanForcing)
    zlo, zhi = extrema(f.z)
    Tlo, Thi = extrema(f.Tz)
    Slo, Shi = extrema(f.Sz)
    print(
        io,
        nameof(typeof(f)),
        "{",
        eltype(f.Tz),
        "}: ",
        length(f.z),
        "-point profile, z ∈ [",
        zlo,
        ", ",
        zhi,
        "] m, T ∈ [",
        round(Tlo; digits = 3),
        ", ",
        round(Thi; digits = 3),
        "] °C, S ∈ [",
        round(Slo; digits = 3),
        ", ",
        round(Shi; digits = 3),
        "] psu",
    )
end

Base.show(io::IO, f::AbstractOceanForcing) = _show_forcing(io, f)

# The ice forcing is one field, so summarise it as a range rather than printing a
# whole matrix; a uniform field collapses to the single value.
function Base.show(io::IO, f::AbstractIceForcing)
    lo, hi = extrema(f.T_ice_base)
    print(io, nameof(typeof(f)), "(T_ice_base = ")
    lo == hi ? print(io, round(Float64(lo); digits = 2)) :
    print(io, round(Float64(lo); digits = 2), " … ", round(Float64(hi); digits = 2))
    print(io, " °C)")
end

function Base.show(io::IO, f::CavityForcing)
    show(io, f.ocean)
    print(io, "\n  ice: ")
    show(io, f.ice)
end

function Base.show(io::IO, g::Grid{FT}) where {FT}
    print(
        io,
        "Grid{",
        FT,
        "}: ",
        g.Nx,
        "×",
        g.Ny,
        " cells (",
        g.Nx - 2,
        "×",
        g.Ny - 2,
        " interior), dx = ",
        g.dx,
        " m, dy = ",
        g.dy,
        " m",
    )
end

function Base.show(io::IO, ::MIME"text/plain", g::Grid)
    show(io, g)
    msk = g.mask
    print(
        io,
        "\n  cells: ",
        count(==(3), msk),
        " shelf, ",
        count(==(4), msk),
        " gap, ",
        count(==(2), msk),
        " grounded, ",
        count(==(0), msk),
        " ocean, ",
        count(==(1), msk),
        " land/border",
    )
    r, c = g.crop
    (length(r), length(c)) == g.input_size || print(
        io,
        "\n  cropped from ",
        g.input_size[1],
        "×",
        g.input_size[2],
        " (rows ",
        first(r),
        ":",
        last(r),
        ", columns ",
        first(c),
        ":",
        last(c),
        ")",
    )
end

function Base.show(io::IO, g::Geometry{FT}) where {FT}
    print(
        io,
        "Geometry{",
        FT,
        "}: ",
        count(>(0), g.tmask),
        " active cells (",
        count(>(0), g.imask),
        " under ice)",
    )
end

function Base.show(io::IO, p::Params{FT}) where {FT}
    print(
        io,
        "Params{",
        FT,
        "}(",
        nameof(typeof(p.entrainment)),
        " + ",
        nameof(typeof(p.melting)),
        " + ",
        nameof(typeof(p.convection_scheme)),
        " + ",
        nameof(typeof(p.lateral_viscosity)),
        " + ",
        nameof(typeof(p.front_pressure)),
        ")",
    )
end

Base.show(io::IO, b::BoundaryConditions) = print(
    io,
    "BoundaryConditions(open ocean = ",
    nameof(typeof(b.open_ocean)),
    ", grounding line = ",
    nameof(typeof(b.grounding_line)),
    ", land = ",
    nameof(typeof(b.land)),
    ", gaps = ",
    nameof(typeof(b.gaps)),
    ")",
)

Base.show(io::IO, ::ConservativeCFL) = print(io, "ConservativeCFL()")
Base.show(io::IO, ::ExactCFL) = print(io, "ExactCFL()")

Base.show(io::IO, ::FixedDt) = print(io, "FixedDt()")
Base.show(io::IO, ts::AdaptiveDt) = print(
    io,
    "AdaptiveDt(cfl_target = ",
    ts.cfl_target,
    ", q = ",
    ts.q,
    ", ncheck = ",
    ts.ncheck,
    ", dt ∈ [",
    ts.dtmin,
    ", ",
    ts.dtmax,
    "])",
)

function Base.show(io::IO, ::MIME"text/plain", p::Params{FT}) where {FT}
    println(io, "Params{", FT, "}:")
    scal = [
        (fn, getfield(p, fn)) for
        fn in fieldnames(typeof(p)) if getfield(p, fn) isa Number
    ]
    for chunk in Iterators.partition(scal, 4)
        println(
            io,
            "  ",
            join((rpad(string(k, " = ", v), 22) for (k, v) in chunk), " "),
        )
    end
    println(io, "  entrainment    = ", p.entrainment)
    println(io, "  melt           = ", p.melting)
    println(io, "  convection     = ", p.convection_scheme)
    println(io, "  max layer D    = ", p.max_layer_thickness)
    println(io, "  lat. viscosity = ", p.lateral_viscosity)
    println(io, "  front pressure = ", p.front_pressure)
    print(io, "  coriolis       = ", p.coriolis)
end

_backend_name(m::Model) = nameof(typeof(KA.get_backend(getfield(m, :grid).z_draft)))

function Base.show(io::IO, m::Model{FT}) where {FT}
    g = getfield(m, :grid)
    print(
        io,
        "Model{",
        FT,
        "} on ",
        _backend_name(m),
        ": ",
        g.Nx - 2,
        "×",
        g.Ny - 2,
        " interior, ",
        nameof(typeof(getfield(m, :forcing))),
        " forcing",
    )
end

function Base.show(io::IO, ::MIME"text/plain", m::Model{FT}) where {FT}
    g = getfield(m, :grid)
    p = getfield(m, :params)
    println(io, "Model{", FT, "} on ", _backend_name(m))
    println(
        io,
        "  grid:    ",
        g.Nx - 2,
        "×",
        g.Ny - 2,
        " interior cells, dx = ",
        g.dx,
        " m, dy = ",
        g.dy,
        " m, ",
        count(==(3), g.mask),
        " shelf cells",
    )
    println(io, "  forcing: ", getfield(m, :forcing))
    print(
        io,
        "  params:  ",
        nameof(typeof(p.entrainment)),
        " + ",
        nameof(typeof(p.melting)),
        " + ",
        nameof(typeof(p.convection_scheme)),
        " + ",
        nameof(typeof(p.lateral_viscosity)),
        " + ",
        nameof(typeof(p.front_pressure)),
    )
    print(io, "\n  boundary: ", getfield(m, :boundary))
end

_days(seconds, spd) = round(seconds / Float64(spd); digits = 3)

Base.show(io::IO, c::Clock{FT}) where {FT} = print(
    io,
    "Clock{",
    FT,
    "}(time = ",
    round(c.time; digits = 1),
    " s, iteration = ",
    c.iteration,
    ", dt = ",
    c.dt,
    " s)",
)

function Base.show(io::IO, o::OutputConfig)
    if o.saveday > 0
        print(
            io,
            "OutputConfig: every ",
            o.saveday,
            " d → ",
            joinpath(o.resultdir, o.name),
        )
    else
        print(io, "OutputConfig: disabled (saveday = 0)")
    end
end

function Base.show(io::IO, sim::Simulation{M,FT}) where {M,FT}
    m = sim.model
    print(
        io,
        "Simulation{",
        FT,
        "} on ",
        _backend_name(m),
        " at day ",
        _days(sim.clock.time, m.seconds_per_day),
        ", dt = ",
        sim.clock.dt,
        " s",
    )
end

function Base.show(io::IO, ::MIME"text/plain", sim::Simulation{M,FT}) where {M,FT}
    m = sim.model
    c = sim.clock
    println(io, "Simulation{", FT, "} on ", _backend_name(m))
    println(io, "  model:   ", m)
    println(
        io,
        "  clock:   day ",
        _days(c.time, m.seconds_per_day),
        ", iteration ",
        c.iteration,
        ", dt = ",
        c.dt,
        " s",
    )
    println(io, "  stepper: ", sim.tstep, ", ", sim.cfl, ", Robert–Asselin ν = ", sim.nu)
    println(io, "  stop:    ", sim.stop)
    print(io, "  output:  ", sim.output)
end
