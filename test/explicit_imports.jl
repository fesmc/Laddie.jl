# Every name Laddie uses from a dependency is imported by name (`using X: a, b`), and
# qualified accesses go through the module that owns the name.  The check covers the
# loaded extensions too.  Allowed non-public names: `@adapt_structure`, `Base.RefValue`
# and `Base.typename` are the standard spellings; the ForwardDiff extension extends
# Laddie internals (`_primal`, `_safe_sqrt`) and uses `Dual` and `ForwardDiff.value`,
# which ForwardDiff documents but does not declare public.
# `test_explicit_imports` runs its own `@test`s.

test_explicit_imports(Laddie;
    all_explicit_imports_are_public = (; ignore = (:Dual,)),
    all_qualified_accesses_are_public = (;
        ignore = (Symbol("@adapt_structure"), :RefValue, :typename,
                  :_primal, :_safe_sqrt, :value)))
