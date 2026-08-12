# Contributing To SmallField

Rules for `src/`, `cli/`, `bench/`, `test/`, and `python/`.
Where tooling can enforce a rule it does, in `.githooks/pre-commit`; where it cannot, this document is the source of truth.
When you change a rule, change it here first.

## Enforcement

Configuring the CMake build arms the hooks:

```sh
cmake -B build .        # runs `git config core.hooksPath .githooks`
git config core.hooksPath   # should print .githooks
```

__Every check inspects only the added lines of a staged file.__
Legacy code migrates on its own schedule, cost scales with the size of the diff rather than the size of the tree, and no flag-day refactor is needed to adopt a rule.
A file that already contains a violation stays committable; a diff that introduces one does not.

External tools are guarded, so a fresh clone without `clang-format`, `cmake-format`, or `ruff` still commits.

There is deliberately no in-band escape comment.
If a rule needs to bend for a legitimate one-off, edit the rule here and document the bend.
`git commit --no-verify` exists and you should almost never reach for it.

__That applies to the formatters too: no `// clang-format off` and no `# cmake-format: off`.__
The checked-in `.clang-format` and `.cmake-format.py` profiles are the law, so a file formats clean as a whole rather than fencing off the parts it disagrees with.

__In Markdown, bold is `__like this__` and italic is `_like this_`__, never the asterisk forms, so the rendered and the raw text agree across every renderer these documents pass through.
Inside a Doxygen comment neither form is used at all — that is what `@b` is for.

__Run `bash .githooks/selftest` after changing a check.__
It builds a throwaway repository per fixture and asserts that a rejection fires for exactly one reason, which makes every rejection case a false-positive test for all the other checks.
A check that arrives without a fixture is a check nobody has seen fail: of the six checks found broken in 2026-08, every one that failed completely rather than narrowly had no fixture.

__The public release of 2026-08 was the one deliberate exception to the added-lines discipline.__
A whole-tree sweep corrected documentation that contradicted the code, deleted dead code, and normalised includes.
Reformatting whole files turns every legacy line into an added line, so the hook then reports violations the file has carried for months; that is the cost of a flag day and the reason the rule exists.
Outside a release sweep, format only the lines you changed — `git clang-format --staged` does exactly that.

## Commit Messages

The subject is `<Verb>: <Capitalized summary>`.
`Add`, `Improve`, and `Fix` carry almost everything.
`Make` is for build-system adaptations, `Docs` for documentation, and `Chore` for mechanical behavior-preserving churn — all three are rare.
`Break` is rarer still, and marks a commit that lands a vocabulary its consumers do not yet spell: the tree does not build at that commit, on purpose, because the alternative is one reviewable change spread across every file that mentions the old name.
A `Break` says so in its last line and names the commits that finish the propagation.
`Merge` commits are exempt because Git writes their subject.

The body describes the semantic change.
Filename moves, naming tweaks, and comment polish are noise; keep the first paragraph for what actually landed.

Documentation folds into the commit that makes it true rather than landing on its own.

__History stays human-authored.__
No `Co-authored-by` bot trailer, no "generated with" line, no assistant name, vendor URL, or robot emoji.

## Naming

Types carry `_t`, compile-time constants and enumerators carry `_k`, internal free functions and private members carry a trailing underscore, and template parameters carry a trailing underscore with type slots ending `_type_`.

__A class or struct template takes no `_t`, because it is not a type.__
`_t` marks something you can declare a variable of — a non-template class, an alias, a concrete specialization.
A template is the thing that makes those, so it wears the bare noun and its specializations wear the suffix:

```cpp
template <typename value_type_, memory_allocator allocator_type_>
class buffer { … };

using pinned_bytes_t = buffer<std::byte, pinned_allocator_t>;
using device_bytes_t = buffer<std::byte, device_allocator_t>;
using device_indices_t = buffer<std::uint32_t, device_allocator_t>;
```

Concept names take neither suffix for the same reason.

__Everything this project declares is snake_case.__
No camelCase, no leading capital, no `SCREAMING_SNAKE` outside a macro, no `kPrefixed` constant.

__A name reached through a library object keeps that library's spelling.__
`sc->d_params[k].kbaIJK` is RDKit's field and renaming it does not compile, while the `kba_ijk` it is assigned into is ours.
The adapter in `parameters/mmff94_tables_rdkit.hpp` is the boundary: library spelling on the right, ours on the left.

__A name crossing into Python keeps Python's spelling, for the same reason.__
`PascalCase` classes and `UPPER` enum members are what a Python caller expects, and the binding is a boundary exactly like the RDKit adapter — ours on one side, theirs on the other.
So `ConformerStatus.SUCCESS` and `class ShardStats:` stay, while the C++ they bind to remains `conformer_status_t::success_k` and `shard_stats_t`.
The carve-out reaches the binding and the Python package only; a C++ declaration is snake_case wherever it sits.

Spell concepts out.
`index` not `idx`, `config` not `cfg`, `source` and `destination` not `src` and `dst`, `scratch` not `tmp`, `count` not `cnt`, `table` not `tbl`, `buffer` not `buf`, `pointer` not `ptr`, `neighbor` not `nbr`, `descriptor` not `desc`, `conformer` not `conf`, `compute` not `calc`.

__A physics symbol is spelled out, and the symbol itself goes in the docstring.__
`mmff94_bond_t::force_constant` documented as `kb` in mdyn per angstrom, never a bare `kb` in our own code.
The published MMFF94 parameter-table columns are spelled out too: `coordination_number` rather than `crd`, `valence` rather than `val`.

__A loop variable names the axis it walks__ — `atom`, `bond`, `molecule`, `conformer`, `neighbor`, `term`, `row`, `device`, `bin`.
Never a bare letter, even where the letter matches the specification's own index notation.

__No binding under four characters.__
`check_short_identifiers` enforces it, because the alternative scales badly and this tree proved how badly: `a` once meant an atom, a neighbour slot, an apex, a loop counter and an angle record inside one file, `c` was both the current SMARTS character and a conjunct index inside one parser, and `n` was a CSR cursor in one function and MMFF.V equation 18's electronegativity exponent in the next.

The exempt words are short because they are whole words, not because they are abbreviations: `out`, `end`, `key`, `row`, `dim`, `hit`, `ok`, `top`, `sum`, `max`, `min`, `raw`, `bin`, `tag`, `on`, `bit`, `log`, `got`, `now`, `arg`, `sha`, `oop`, `arc`, `cis`.
`lhs` and `rhs` name an operator's two operands; `dx`, `dy` and `dz` are axes.
Namespace aliases are exempt outright.

__MMFF term positions keep the specification's letter as a suffix.__
MMFF94 defines a bond over atoms i-j, an angle over i-j-k and a torsion over i-j-k-l, so `atom_i` maps onto the published equations where `first_atom` does not — and `mmff94_bond_t::atom_i` already spells it that way.
The exemption is narrow: it holds only where the letter genuinely is a term position.
A matrix walk names `row` and `column`, and a cursor that merely happens to be spelled `i` names its axis like any other.
Getting this backwards is not cosmetic — `aj` in four files held the MMFF __k__ atom, so the obvious rename would have been wrong and would have compiled.

__Directory trees stay flat.__
The maximum depth is `src/smallfield/<subsystem>/<file>`.
A shared helper that is generic over its operand type, depends on nothing outside the standard headers, and is used by two or more subsystems goes in `src/smallfield/utilities.hpp`; anything failing all three goes to the subsystem that owns it.
No catch-all directory families.

__Namespaces mirror the directory structure, and `detail` namespaces do not exist.__
An internal helper lives in the file's own namespace with a trailing underscore.
A namespace alias spells its target out; `sfp` bound to two different namespaces in one test suite is what the short forms cost.

`#pragma region <name>` and `#pragma endregion <name>` mark sections, with plain words in the name — a dash makes `clang-format` reflow around it and the markers drift out of pairing, and a bracket does the same.

## Includes

__Every quoted include is anchored at a root on the include path, never at the includer's own directory.__
The roots are `src/` and the repository root, so the allowed prefixes are `smallfield/`, `test/`, `bench/` and `cli/`.
A sibling header in the same directory is written out in full like any other.

```cpp
#include "smallfield/preprocess/csd_torsion_native.hpp" // sibling of the includer, still spelled out
#include "test/cuda/test_mmff_gradient_kernels.hpp"
#include "test/test_rdkit_reference_helpers.hpp"
```

The bare form resolves only because `""` searches the includer's own directory first, so the path a reader sees is not the path the compiler used.
Worse, two includes sitting next to each other then anchor at different roots with nothing to distinguish them, and moving a file changes the meaning of an unmoved line.
`check_include_anchors` in `.githooks/pre-commit` rejects the bare form on added lines.

Group `<c*>` C-library trampolines first, then pure C++ headers, then third-party, then project includes, with a blank line between groups and alphabetical order inside the standard groups.
Third-party groups keep their own required order — RDKit's headers are not self-contained and sorting them breaks the build.

## State and Memory

__Per-worker state is an explicit rank into caller-owned scratch, never `thread_local`.__
Hidden global state is not a style preference here: the file-scope scratch in `etkdg_prepare_batch.hpp` is precisely why `prepare_molecule_batch_into` cannot be called concurrently, which blocks the deeper pipeline ring.

__Fill a caller-owned output rather than returning a container.__
Write `fill_x_into(input, std::vector<T> &out)` with `out.clear()`, not `std::vector<T> compute_x(input)`, and declare the scratch above the loop that consumes it so a growth phase happens once.

The test that matters is not whether a function returns a container, but where it is called from: __a container returned by value from a function called inside a loop is a per-iteration allocation with a name that hides it.__
A one-shot builder returning a `std::vector` at start-up is fine and should stay.

```cpp
// no — two allocations per molecule, and the inner one is discarded on the same line
static std::vector<float> interleaved_to_soa_(std::vector<float> const &interleaved);
std::vector<float> positions = interleaved_to_soa_(extract_coords_interleaved(*molecule));

// yes — the buffers outlive the loop and grow once
std::vector<float> interleaved, positions;
for (auto const &smiles : corpus) {
    extract_coords_into(*molecule, interleaved);
    transpose_into(interleaved, positions);
}
```

The shape to copy already exists in `cli/`: a struct owning every output array with one `resize_for` that only ever grows, so nothing on the hot path allocates or frees.

__An extent belongs inside the view, not beside it.__
A flat buffer plus a separately carried count is three parameters describing one thing, and it lets the count disagree with the storage undetected.
`src/smallfield/types/views.hpp` holds the aliases; `reset_square_matrix` is the builder that grows caller storage and hands back a view over it, so sizing never drifts from the extent.
`fill_bond_hops_into(std::span<mmff94_bond_t const>, hop_matrix_t)` is the API shape to copy: a span in, a view out, and the storage owned by whoever called.

Count the occurrences before deciding this is cosmetic:

```cpp
// no — `num_atoms` appears five times in fifteen lines, and nothing compares them
perturb_(positions, prepared.num_atoms, perturbation);
std::vector<float> interleaved(static_cast<std::size_t>(prepared.num_atoms) * 3, 0.0f);
for (std::uint32_t atom = 0; atom < prepared.num_atoms; ++atom)
    for (std::uint32_t axis = 0; axis < 3; ++axis)
        interleaved[atom * 3 + axis] = positions[axis * prepared.num_atoms + atom];
set_coords_interleaved(*molecule, interleaved.data());

// yes — twice, both times constructing a view, and the loops are gone
auto const aos = reset_coordinates_aos(scratch.interleaved, prepared.num_atoms);
auto const soa = reset_coordinates_soa(scratch.positions, prepared.num_atoms);
perturb_(soa, perturbation);                 // the extent travels with the data
transpose_into(soa, aos);
set_coords_interleaved(*molecule, aos.data_handle());
```

__A memory layout is a type, not an index expression.__
`positions[axis * num_atoms + atom]` and `interleaved[atom * 3 + axis]` are two different layouts written as arithmetic a reader has to decode, and passing one where the other is expected compiles.
Name them, and the compiler separates them.

The matrix half of this is done — `packed_matrix` and `hop_matrix_t` are in use.
The coordinate half is `coordinates_soa` and `coordinates_aos`, and __the layout policy carries the storage order while the accessor carries the role__:

```cpp
template <coordinate_role_t role_, typename value_type_, std::size_t dimensions_, typename layout_type_>
using coordinates = custd::mdspan<value_type_, custd::extents<std::uint32_t, custd::dynamic_extent, dimensions_>,
                                  layout_type_, role_accessor<role_, value_type_>>;

template <coordinate_role_t role_, typename value_type_, std::size_t dimensions_>
using coordinates_soa = coordinates<role_, value_type_, dimensions_, custd::layout_left>;
template <coordinate_role_t role_, typename value_type_>
using coordinates_aos = coordinates<role_, value_type_, conformer_dimensions_k, custd::layout_right>;
```

Both are addressed `(atom, axis)` and the mapping supplies the stride, so a reader never decodes an index expression and `transpose_into`'s body is layout-agnostic.

__Distinguishing the two by which extent is static does not work, and that is measured rather than argued.__
A static extent is always convertible to a dynamic one, so `extents<u32, 3, dyn>` and `extents<u32, dyn, 3>` both convert implicitly to `packed_matrix`'s `extents<u32, dyn, dyn>` — a conformer passed as a bounds matrix, compiling silently.
`layout_left` against `layout_right` blocks it in both directions, because mdspan's cross-layout mapping constructor is constrained to rank one or lower and these are rank two.

__The role is an accessor, not a wrapper.__
Positions, velocities and a gradient are identical in shape and layout and ride adjacent through the FIRE driver, so nothing but the role separates them.
mdspan's converting constructor requires the accessors be constructible from one another, and the accessor is already one of mdspan's four template parameters, so the role costs no bytes and needs no new class.
One scoped `coordinate_role_t` names the closed set in one place rather than scattering empty tag structs.

The transposition between the two layouts is one named function, overloaded on direction so a call site cannot invert it.
Writing it inline is how the same three lines came to exist in four files in four spellings, two of them sharing a name with different signatures.

__`custd` is `cuda::std`, and that is not a preference.__
`std::mdspan` needs C++23, and `nvcc` 12.8 and 12.9 reject `-std=c++23` outright, so it cannot enter any translation unit `nvcc` touches.
libcu++ ships the same interface at C++20 for host and device alike and arrives with `CUDA::cudart`, already a `PUBLIC` dependency.
The alias names the namespace once so the swap is a single `#if` when `nvcc` 13 and libstdc++ 16 land together.
It is spelled for the namespace rather than for one type in it, because it names the span as well as the views.

__One span type, and it is `custd::span`.__
`std::span`'s members are not `__host__ __device__`, which is the whole reason `parameters/mmff94_tables.hpp` once carried a twelve-line `mmff94_span` reimplementing it.
`custd::span` is, so a host helper and a device helper share one signature and the hand-rolled copy is deleted.

Three constraints come with it.
Extents are `std::uint32_t`, which is a claim about scope: these view one molecule, and a packed view is 16 bytes at `std::uint32_t` against 24 at `std::size_t`, per kernel argument.
__Never view the conformer pool through them__ — a 10 M-molecule run at three conformers passes `UINT32_MAX` elements around conformer 11.2 M, and a 32-bit offset wraps into another molecule's coordinates.
And `cuda::std::mdspan` performs __no bounds checking__, even with `CCCL_ENABLE_ASSERTIONS`, so a `(9, 9)` access on a four-by-four view reads whatever was there; raw arithmetic that survives keeps its `sf_assert_`.

__An alias arrives with its first consumer, never ahead of it.__
The two view types this header replaced sat unused long enough for one of them to acquire a docblock that misstated its own size.

__Owning storage is one template, and the allocator is one parameter.__
`buffer<value_type_, allocator_type_>` in `types/storage.hpp` owns every allocation the project makes.
Two policy parameters let `byte_buffer<pinned_alloc_t, device_dealloc_t>` compile, which is a use-after-free the type system was in a position to refuse.

`capacity()` is always in elements, so a byte arena's elements are bytes and the unit never has to be inferred.

__Device memory is allocated and released in stream order.__
`device_allocator_t` holds a `cudaStream_t` and goes through `cudaMallocAsync` and `cudaFreeAsync`.
`cudaMalloc` and `cudaFree` both synchronize the whole device, so under the per-device fan-out every allocation on any device serializes against every other device's in-flight work — the contention that pre-allocation postpones rather than removes.
The stream also carries the device, so a free reaches the context that made the allocation without anyone storing a device id and switching to it by hand; that is what lets every owner of a device buffer default its destructor and its move operations.
The stream must outlive the buffer, which is the lifetime rule every other stream-ordered resource in the pipeline already follows.

`pinned_allocator_t` carries no stream, and that is not an omission.
`cudaHostAlloc` has no stream-ordered form, so a stream there would suggest an ordering the call does not have.

Sub-allocation survives only where the single allocation is what makes an upload one `cudaMemcpyAsync` rather than N, which is true of the wave pool and of nothing else here.
An arena carved by hand exists mostly to amortize `cudaMalloc`, and stream-ordered allocation removes that cost — so a table that used to be placed into a shared slab is now simply its own `buffer`.

__A grow allocates before it releases, and reports what it did.__

```cpp
enum class reserve_outcome_t : std::uint8_t { unchanged_k, grown_k, failed_k };
```

`unchanged_k` against `grown_k` is not pedantry: the constructor pre-reserves so that no allocation fires inside the per-device fan-out, and that invariant is only assertable if the two are distinguishable.
`failed_k` carries no reason because a stream-ordered allocation's reason belongs to the stream, and is read at the sync point the caller already has rather than copied into a return value at every call site that cannot act on it.

__A transfer names both extents and the stream it rides.__

```cpp
template <typename element_>
[[nodiscard]] bool upload_into(custd::span<element_> destination, custd::span<element_ const> source,
                               cudaStream_t stream, cuda_error_sink_t const &sink = {}) noexcept;
```

Both extents come from the spans, so a caller cannot restate one and get it wrong, and a destination smaller than its source is refused rather than truncated.
The stream is required, never defaulted: the legacy default stream serializes against every other stream on the device.
A pageable source silently degrades the copy to a blocking staged one inside the driver, which is why anything on the wave path stages through `pinned_bytes_t`.

__Ownership is readable from a declaration, never reconstructed from call sites.__
An owning buffer is `buffer<value_type_, allocator_type_>` and a borrowed one is a `custd::span` or a view.
There is no third thing, and a raw pointer that outlives the expression using it is a defect.

A pointer into a container, held across a phase boundary, is a lifetime bug waiting for a reallocation — `gpu_pipeline_handle_t` carried one until it was replaced by the span it actually read.
Residence is a compile-time property for the same reason: host code holding a device pointer should not compile rather than fail at run time.

__Two facts that must agree are one fact, or they are a defect.__
An agreement maintained by hand at every write site is one the compiler will not help with, and the type system exists to carry most of them.

A count beside a pointer is one span.
A struct of `<name>_byte_offset` and `num_<name>` pairs is a layout of spans, and a pool carved into such a struct is one walk handing out subspans — `pool_cursor` in `types/views.hpp` sizes an allocation and carves it with the same sequence, so `preprocess/wave_slice_pool.hpp` stores element counts and never an offset.
A sizing pass and a carving pass written separately are the same defect one level up: two sequences of the same 27 steps, with nothing to notice when a step is added to one of them.
Parallel arrays walked by a shared index are one struct, or a view that carries the relation.
A field equal by construction to another is deleted rather than documented.
And two identically-shaped things with different meanings get different types, which is what `coordinate_role_t` does and what generalizes beyond it.

Where a relation genuinely cannot be typed, say so in a comment naming the invariant.
Do not invent a wrapper whose only content is a cast.

__A call that can outlast a millisecond releases the GIL.__
Nothing inside the released scope may touch a Python object.
A wave runs for minutes, and holding the lock across it blocks every other thread in the interpreter for no reason.

__No pointers inside containers.__
`std::vector<T *>` and `std::vector<std::unique_ptr<T>>` are a pointer chase plus an allocation per element for something whose count is usually known and small.
Store values, or store indices into a flat arena.

Arrow is the one exception, and it is the same boundary the naming rules draw around RDKit.
`arrow::schema()` and `arrow::Table::Make()` take `std::vector<std::shared_ptr<arrow::Field>>` and `std::vector<std::shared_ptr<arrow::Array>>` in their own signatures, so a caller cannot hand them anything else — the shape is that library's rather than ours.

__Lazy initialization is a function-local `static const`. There is no second mechanism.__

A `static const` whose initializer is an immediately-invoked lambda is not a different mechanism — it is the same one, spelled for an initializer that needs statements.
And `std::call_once` buys nothing a magic static does not already guarantee: both run once, both are thread-safe, both retry if the initializer throws.

```cpp
// no
static std::once_flag flag;
std::call_once(flag, [] { … });

// yes — same guarantee, no flag, and it reads as what it is
[[maybe_unused]] static bool const warmed = [] { …; return true; }();
```

`std::once_flag` and `std::call_once` are banned.
Warming third-party state is not an exception: bind the side effect to a `static const` and the contract is identical.
Two near-duplicate RDKit warm-ups each carried their own flag, so the warm-up ran twice in one binary — which is what having two mechanisms buys.

__A test counts through a caller-owned tally, not module state.__

Eighteen of twenty-two test files hold no mutable file-scope object.
The four that do are the ones that became hard to read: one counts mismatches three ways inside a single `main` — module-scope integers, a caller-owned `lookup_mismatch_counts_t`, and a reporter with its own internal counters — and gates on two of the three independently.

```cpp
// no — module state, reachable only through `using` re-exports so `main` can see it
static int host_failure_count_ = 0;

// yes — a tally threaded by reference, which also aggregates across corpora for free
struct oracle_tally_t { std::uint64_t molecules = 0; std::uint64_t centers = 0; };
```

Where a `matching_` / `mismatched_` counter pair is genuinely not redundant — because one side of the comparison is conditionally skipped, so the denominators differ — keep the pair and say so in a comment.
Otherwise `compared` minus `mismatches` is the same number and cannot drift from its partner.

__A trailing underscore already marks internal linkage; `g_` adds nothing.__
Carrying both is two conventions for one fact, and `g_` is a poor thing to grep for in a codebase where `g_factor` is the MMFF94 Slater-Kirkwood term.

## Control Flow

__A function's return type is written, never deduced.__
`auto` on a return sends a reader to the body and then to whatever the body calls, and it is not what a grep for the type finds.

The real cost is that it converts a compile error into silent propagation.
Change what a helper returns, and an explicit signature fails at the one accessor that forwards it; a deduced one carries the new type out through every call site with nothing to say so.

```cpp
// no — the contract is only discoverable by reading whatever the body forwards to
auto etk_torsions() const noexcept { return carve<etk_torsion_descriptor_t>(…); }

// yes
custd::span<etk_torsion_descriptor_t> etk_torsions() const noexcept { return carve<…>(…); }
```

A trailing return type is the same promise written on the right and is fine.
A deduced return under a template head is fine too, because there it may genuinely depend on a deduced parameter — `align_up_` returns the width of whatever it was handed, which is the whole point of it.
`auto` on a local whose type is obvious from its initializer is unaffected.

__A wrapper whose body is a call with the same arguments is a name, not a function.__
It adds one identifier to look up and one place for the two to drift, and it makes no wrong state harder to spell.
Call the thing directly.

```cpp
// no — the name is the whole content
static constexpr auto bits_identical_ = [](float host_value, float device_value) {
    return bitwise_equal_(host_value, device_value);
};
```

When such a wrapper is genuinely load-bearing, it is usually adapting arity, and the fix belongs at the callee.
A predicate is a `constexpr` callable object rather than a function carrying a default argument: a function name decays to a pointer whose type does not include the default, so it cannot be passed where a two-argument callable is wanted, and every call site grows a lambda to bind what the declaration already said.
`exactly_equal` in `test/test_helpers.hpp` is the shape to copy — it is used at a hundred call sites and needs an adapter at none of them.

__Discard a return value by binding it `[[maybe_unused]]`, never with a `(void)` cast.__
The cast is a C-era idiom that reads as hiding something, and it does not compose with `[[nodiscard]]` the way a bound variable does.
`[[maybe_unused]] cudaError_t const status = allocator.deallocate(pointer);` names what was ignored, so the next reader can see whether ignoring it is still right.

__Three or more `[&]` lambdas in one function means that function's locals are a struct.__
The lambdas are its methods, and each should receive exactly the state it needs.

`generate_conformers` holds seven across 108 of its 270 lines and `write_shard` holds eight in 321.
Every one of them reaches an arbitrary subset of some twenty enclosing locals, so nothing declares which piece touches what, no piece can be reasoned about alone, and none can be tested.
`[&]` is the smell rather than the lambda: where one genuinely wants two names, capture those two.

__A `bool` that steers behavior is an enum.__
Already stated under Naming for parameters; it applies equally to a `bool` return, which throws away the reason.

```cpp
// no — five distinguishable refusals fold into one `false`, and the caller cannot tell
//      "not a number" from "does not fit uint32_t" from "above the ceiling"
static bool parse_number_(std::string_view text, number_type_ &number);

// yes
enum class number_parse_t { parsed_k, malformed_k, out_of_range_k, below_minimum_k, above_maximum_k };
```

Both spellings are live in this tree, written for the same job.
The extreme case is a shard writer with twenty-four distinct failure modes, all collapsed to `false`, whose callers can only print `write failed`.

__A three-state answer is an enum, never a bool pair and never a sentinel.__

```cpp
// no — cis / trans / no-perceivable-stereo, spread across a return and a reference.
//      `is_cis == false` is meaningless unless `found_stereo` is separately consulted, so a caller
//      that forgets silently classifies every unperceived conformer as trans.
static bool perceived_any_cis(RDKit::RWMol &molecule, bool &found_stereo);

// no — the enum is written out, in a comment
std::vector<std::int8_t> row_outcomes;  // 1: success, 0: conformer failure, -1: molecule failure

// yes — the omission the bool pair allows becomes a compile error
enum class bond_stereo_t { cis_k, trans_k, unperceived_k };
switch (perceive_bond_stereo(molecule)) {
case bond_stereo_t::unperceived_k: ++unperceived; continue;
…
}
```

__One name, one meaning.__
A flag used as both "something failed" and "no header printed yet" is two variables sharing a declaration — and in the one function that does this, the header-once idiom is implemented properly nine lines below, with its own flag.

__One fact, one encoding.__
Two fields that are equal by construction are an invariant maintained by hand at every write site, and the compiler will not help.
A `bool is_valid` beside a `std::unique_ptr` that is null on exactly the same paths is five hand-maintained agreements waiting for a sixth failure path to forget one.
Where the failure modes are already distinguished and then discarded, an enum keeps them and costs nothing.

__An accumulator boolean far from its test is a counting function.__
Eight writes spread over two hundred lines to a single read is not a flag, it is a report.
Return counts from named predicates and sum them; the flag disappears and the counters stay.

```cpp
// no — and the write below is dead: every caller `continue`s past the only read
bool molecule_matches = true;
auto const require_count_match = [&](…) {
    if (cpu_count != gpu_count) { molecule_matches = false; return false; }
    return true;
};
if (!require_count_match("bond", …)) { ++mismatched_molecule_count; continue; }
…                                    // 201 lines
if (molecule_matches) ++matching_molecule_count;

// yes
std::size_t const mismatches = compare_bonds(…) + compare_angles(…) + compare_torsions(…);
```

Two bookkeeping schemes for one fact is what makes a dead write invisible.
Note the loop must still run to completion — those counters are the output — so the fix is extraction, never a `break`.

__A trailing `if` that wraps the rest of the body is a guard.__

```cpp
// no
for (…) {
    …two statements…
    if (has_coords && status == 0 && num_atoms > 0) { …ten lines, two nested loops… }
}

// yes — each guard names one reason to skip, at the point the reason is known
if (num_atoms_u8 == 0) continue;                                  // preprocess failed
if (!context.ok || context.num_atoms != num_atoms_u8) continue;   // parse must reproduce atom order
if (status_codes[flat] != success_k) continue;
```

Comment each guard with why it skips.
That is the part worth copying, not the `continue`.

__Write a cross product once.__
The `atom` by `axis` transposition between the coordinate pool and an interleaved buffer is written six times across the tooling, twice inside one loop body — gather, then scatter back twenty lines later — and once hand-unrolled into three copies.
Two named helpers delete all six and put the direction in the name.

__A loop that re-scans its input once per bucket is one pass.__
Looping bins on the outside and every sample on the inside is quadratic where bucketing is linear, and it usually hides the previous antipattern inside it.

__Some loops must run to completion, and they look exactly like the ones that should not.__
Before converting a flag to a `break`, establish what the loop's later iterations still write.
The riskiest instance here fills a per-conformer status array; breaking on the first failure leaves the later slots unwritten.
Where a loop is deliberately non-short-circuiting, say so in a comment, because the next reader will try to simplify it.

Likewise, a bound of `< 8`, `< 4` or `< 2` in term-construction code is an arity or a buffer size, not a reporting cap.
Display budgets belong to the reporter, which preserves and prints the suppressed count.

## Aggregates

__Past three fields, initialize by name.__

Positional aggregate initialization silently survives a field insertion or rename: the literal keeps compiling and means something else.
This has cost this project twice.
Renaming `mmff94_angle_t::padding` to `is_linear` compiled clean at all four construction sites and meant "not linear" at every one.
Inserting `bounds_row_stride` into `distance_geometry_energy_functor_t` shifted every later field in a test's positional initializer and broke `test_dg_energy`, caught only by bisecting against the baseline.

Add a `static_assert` on `sizeof` for any struct whose layout the GPU depends on.

## Comments

Documentation is `/** ... */`.
Every file opens with `@file` carrying the project-relative path, then `@brief`.
The path rather than the bare basename, because eleven pairs of files differ only by extension and three of those pairs carry a byte-identical `@brief`, so the basename alone cannot tell a reader which half of the pair they are in.

A block that fits on one line is just the sentence, with no `@brief` to announce it.
A block long enough to need detail opens with `@brief` on its own line, then a bare star line, then the detail.
`//` is for implementation notes inside a function body.
Continuation lines inside a block are a star and two spaces.

__A member is documented above it, never beside it.__
A trailing `///<` ties the declaration column to the longest comment in the struct, so editing one comment re-pads every neighbour.
An identifier that opens a comment keeps its own case and takes backticks, so `atom_flag_t` never becomes `Atom_flag_t`.

__Anything describing one symbol is `/** ... */`.__
__A label over a run of members is `/* ... */`__, because Doxygen would bind it to the first member alone and silently mis-document the rest.
`/* ... */` also covers prose that documents nothing, such as the rationale above a macro guard.
Both carry the same star-and-two-spaces continuation as everything else.
A run of consecutive `//` lines at namespace or class scope collapses into one block of whichever kind applies.
A standalone line that names the declarations beneath it is a section, so it becomes `#pragma region` rather than a banner comment.
A trailing `//` on an `#include`, an `#endif`, or a closing brace is an idiom and stays.

Every `@param` declares its direction as `[in]`, `[out]`, or `[inout]` — the comma form is not a spelling of the last one.

__Inline code and symbol names take backticks, and none of `@c`, `@ref` or `@a` is used.__
`@b` remains for emphasis of a word that is not code, because `**bold**` inside a comment prints literally.
Emphasis is `@b`, never capitals: `bits that must @b not be set` rather than `bits that MUST NOT be set`.
Backticks promise something a compiler would accept, so they never wrap prose, punctuation, a bare numeral, or math.
The vocabulary in `GLOSSARY.md` is the deliberate exception, and SMARTS patterns are backticked because SMARTS is a language.

Sentences are separated by a single space.

Display math and layout diagrams are fenced with `@verbatim`.
Nothing in this tree runs Doxygen, so LaTeX would be markup no reader ever sees, and `@code` cross-links identifiers and mangles the notation.
An existing `@code` block holding a real usage example stays as it is.

__Inside a fence the notation is ASCII, and Greek letters are the one exception.__
Write `d_ij` and `^2` and `*`, not `dᵢⱼ` and `²` and `·`.
A fence is a maths context, so `^` cannot be mistaken for exclusive-or; and a typographic subscript is missing from half the monospace fonts in use, so its fallback glyph destroys the column alignment the fence exists to preserve.
`d_ij` is also the only one of the two a reader can grep for.

One spelling per quantity in prose: `×` for a dimension, `θ` and `φ` for angles, `→` for a mapping, `²` for a square, `—` for a prose dash.

__Comments describe what the code is and does now.__
No "previously", "used to", "no longer", "the old design", no phase numbers.
History lives in commit messages and `ROADMAP.md`.

__A comment that asserts a measurable fact is a test.__

If a comment states a property, either the code enforces it or the comment goes.
Every instance found in this codebase redirected real work:

- a comment claiming the MMFF angle scratch lives in shared memory, above a declaration that made it a kernel-local array — ten kilobytes of local memory per thread, hidden by the comment
- a docstring describing a three-stream pipeline, above code that creates one stream
- a header claiming 84 % native pattern coverage against a measured 86.8 %
- a roadmap table quoting a theoretical occupancy unreachable on any GPU this project targets
- a benchmark document naming shared memory as the binding occupancy constraint, when at the real molecule-size distribution it is registers

Decorative rulers carry no information the sentence beneath them does not already carry.

__Split a long string literal yourself and pin each fragment with a trailing `//`.__
Left alone, clang-format chops a literal that overruns the column limit and chops the pieces again on the next run, until a word reads `"a" "r" "r" "a" "y" "s"` — which is what the Python binding's docstring had decayed into.
The pin holds the split where you put it.

## Diagnostics

__The printed precision of a comparison diagnostic must be at least as fine as the predicate that decided it was a mismatch.__

A diagnostic that can render two failing values identically converts a caught bug into a confusing one.
Both instances found here were in tests written to catch single-bit divergence:

- a __bitwise__ equality predicate reporting its operands at `%.4f`, so any one-ULP host versus device difference — exactly what the test exists to catch — printed as two identical numbers
- a `1e-5` tolerance also reporting at `%.4f`, where a failing pair is __guaranteed__ to render identically, because the threshold sits an order of magnitude below the print resolution

`{}` in `{fmt}` is shortest-round-trip, the minimum digits that reconstruct the value, so two distinct floats can never print alike.
Unlike `%.9g` it is not float-versus-double specific.

__The house grammar is one line per differing field, a verb at column zero, ASCII keys, and `key=value` throughout.__

```
MISMATCH vdw atom_type=7 field=alpha host=0.65 device=0.65000004 delta=+5.96e-08
PASS vdw compared=570 mismatches=0 suppressed=0
```

Four properties earn their keep.
One line per differing field, so the failing field names itself rather than the reader diffing four numbers.
ASCII keys — `alpha`, not `α` — because keys are what you grep.
`key=value` throughout, so `awk -F'[ =]'` and `grep 'field=alpha'` both work.
A verb at column zero, so `^MISMATCH` filters alongside `^PASS` and `^FAIL`.

Precision is keyed to what the number __is__, not chosen per site:

| role | spec | why |
|---|---|---|
| a compared value | `{}` | shortest round-trip, so two distinct floats never print alike |
| a delta | `{:+.3g}` | sign always visible, three digits shows the magnitude |
| a tolerance | `{:.3g}` | matches the delta it is compared against |
| a duration in milliseconds | `{:.1f}` | 0.1 ms is the meaningful resolution |
| a percentage | `{:.3f}` | gate and report must agree, and one pair here did not |
| a count | `{}`, or `fmt::group_digits` when a human reads it | never group a number a script might parse |

__The reporter in `test/test_helpers.hpp` owns the grammar, so a call site passes values and never a format string.__
It formats and never compares.
`field` takes the caller's predicate, because the live ones disagree at exactly `error == tolerance` and a built-in tolerance would move a boundary by stealth.

__Every section ends with its comparison count.__
`PASS vdw compared=0` and a real pass must not read alike.
A validator that silently compares nothing is the defect this convention was written after: one aliased its reference to our own entry and compared 5000 values against themselves, and another bound its result to an unused variable and checked nothing at all.

__Three sinks, chosen by who reads the output, and an iostream is none of them.__
`std::cout` and its siblings carry no level, no timestamp and no sink, so a library header reaching for one writes past whatever the caller configured.
`src/` and `cli/` are operator-facing and long-running, so they use `logger_t` and want the wall clock.
`bench/` is read as a report, so it uses bare `fmt::println`.
`test/` uses the reporter and __not__ the logger, because a 34-character varying timestamp destroys column alignment and makes two runs undiffable, and ctest already timestamps.

## Tooling Entry Points

These govern `cli/`, `bench/` and `test/` — the twenty-seven shipped binaries.
They differ enough in nature that there is no shared scaffolding to inherit, so they are held to rules rather than to a base class.

__One announcement, naming the executable, before any work starts.__

```cpp
// no — a banner, and it does not even match the binary's name
fmt::println("bench_chembl: SmallField GPU vs RDKit CPU");
fmt::println("==========================================");

// yes
fmt::println("{}: device warp BFS vs host CSR BFS over the packed hop matrix", executable_name_k);
```

__Decorative rules are banned in output as well as in comments.__
A run of `===`, `---` or `###` inside a string literal carries nothing the sentence beside it does not, and it makes two runs of the same binary diff noisily.
The name a binary prints must be the name of the binary.

__`main` is a driver.__
The body should read as the outline of what the program does, with the steps as named functions.
A `main` carrying its own comparison loops, or declaring a sixty-entry table inline, has stopped being an outline.

__Exit codes mean one thing each.__
`0` success, `1` a check failed, `2` the command line was refused.
A binary that prints a failure and returns `0` is worse than one that says nothing, because a log filtered on `^FAIL` and a shell testing `$?` then disagree.

__Prefer returning a failure to exiting from inside a helper.__
`std::exit` from deep in a call chain skips flushes and unwinding.
The assertion macro in `test_helpers.hpp` is the deliberate exception — it is the suite's primary failure path, and unwinding would only obscure the site.

## Datasets

__Every dataset is reached through `./data/<name>`.__
`data/.gitkeep` is tracked and `data/*` is ignored, so the entries under it are symlinks each machine makes for itself.
Where they point is a property of the machine, not of the project, and belongs in a shell profile or a setup note rather than here.
Never copy a dataset into the repository.

__No absolute path appears anywhere in the tree.__
Not in a source file, not in a default, not in a docstring example, and not in this document.
An absolute path is a claim about one filesystem that the next machine falsifies silently: three binaries here defaulted to corpora under `/tmp` that exist on nobody's disk, and nothing reported it because a missing corpus read as an empty one.

__A corpus arrives by flag__, with `SMALLFIELD_TEST_CORPUS` as the fallback, and no built-in default.

__A named path that does not open is a failure, never an empty run.__
Treating a missing corpus as zero molecules produces a validator that passes because it compared nothing.
That shape has now been found five times in this project; it is the reason every comparison section prints its `compared=` count.

## Performance Claims

__A performance number states its corpus, its sample size, and its weighting.__

A microbenchmark of an isolated access pattern is not an expected result.
This project measured a 6.9× improvement on the distance-geometry bounds read and quoted it as the headline.
It was real — at 128 atoms, which is __0.1 % of PubChem__.
Weighted by the measured atom-count distribution the same change is 2.47×, and end to end it was under 3 %.

So:

- Name the corpus and the sample size. `data/usearch-molecules/pubchem` and `data/usearch-molecules/gdb13` are the standing ones; they have different molecule-size distributions and a blended figure hides which half moved.
- Weight any distribution-sensitive number by the measured distribution before quoting it.
- Compare A against B within one run. Absolute throughput drifts by 10–15 % on a loaded machine while the ratio inside a single run stays stable.
- Report the tail, not only the median. A shifted median and a heavy one-sided tail look identical otherwise, which is how an unpaired energy comparison hid a real defect here for months.
- Say what you could not measure. An 8-GPU claim from a single-GPU box is a projection, and should read as one.

## Environment

`pixi.toml` pins the only environment this is built and tested in, and `pyproject.toml`'s floors follow it rather than claiming a wider range.

__`cmake --preset dev` is the inner loop__ — one architecture detected from the local GPU, `RelWithDebInfo`.
`release` is the shipped fatbin and `profile` adds line info for `ncu`.

__`sf_assert_` is live only under `cmake --preset debug`.__
The other three presets define `NDEBUG`, where it compiles to nothing, so a check written with it is a check the `debug` build runs and the inner loop does not.
Anything that must hold in a shipped build belongs in the code rather than in that macro.

__A user site-packages directory silently shadows the pixi environment.__
`~/.local/lib/python3.X/site-packages` takes precedence, so `import numpy` inside the pixi Python can resolve to a different version than the one pinned.
Check a version with `PYTHONNOUSERSITE=1` before believing it.

## CUDA

One host thread orchestrates devices, with per-device non-blocking streams and events.
Kernel launches are already asynchronous, cross-device ordering is `cudaStreamWaitEvent`, and completion is an event-sync loop, so a host thread adds only scheduling noise.

`__restrict__` on every pointer a kernel takes.

__A `__global__` parameter is a raw `__restrict__` pointer, and the view vocabulary stops at the launch.__
`custd::default_accessor`'s handle type is a plain `element_type *`, and the aliasing guarantee `nvcc` acts on is the one it derives from a kernel formal parameter, so an mdspan cannot carry `__restrict__` across that boundary.
The boundary is the launch and not the file: a `__device__ __forceinline__` helper inlined into a kernel body is not covered, because it has no formal parameter of its own by the time `ptxas` sees it.

The budget is why, and there is none to spare: on the Blackwell targets this project aims at, the largest pipeline kernel already carries a stack frame, so it is spilling before anything is added.

`CHANGELOG.md` records both directions of that sensitivity, and the asymmetry is the point.
Two added pointer parameters and an early-return guard cost half again the registers, a third of the occupancy and most of a factor of throughput — with the compiler's schedule across the body degrading __despite the new code paths being tiny__.
The kernel split that went the other way bought a quarter more throughput.
Register pressure here converts into wall time in both directions, and it does so for reasons that do not follow from reading the diff.

So anything that could reach a kernel is measured with `cuobjdump --dump-resource-usage` against the recorded per-architecture baseline before it lands, and any movement in `REG`, `STACK` or `LOCAL` on the Blackwell targets is a revert rather than a discussion.
The baseline itself is not reproduced here — it is a measurement of today's binary, not a rule, and a number pasted into a rulebook is a comment asserting a measurable fact that nothing re-checks.

Name hardware constants once rather than spelling `32` for the warp size at each site.

Every architecture the project runs on needs an entry in `SMALLFIELD_CUDA_ARCH`.
PTX JIT is forward-only, so PTX from the newest entry covers future hardware and covers nothing already on the list; a device missing from it finds no cubin, no JIT path, and does not run.

__Do not enable interprocedural optimization on a CUDA target.__
Combined with `CUDA_SEPARABLE_COMPILATION`, it switches nvcc from emitting SASS to emitting NVVM LTO IR, and every architecture's `ptxas` codegen then happens at the device link — once per executable, on the whole library, 101.9 s a time against 0.25 s for a plain `nvlink`.
It was about 84 % of the build.
Device LTO optimizes across translation units and every target here has at most one, so it had nothing to work on.
That is also why the compiler cache looked weak at 23 %: `ccache` caches compiles and can never cache a device link, so the dominant cost was invisible to it.

Relocatable device code earns its keep only when a `__device__` function or `__device__`/`__constant__` variable defined in one translation unit is referenced from another.
Nothing here does that, so `CUDA_SEPARABLE_COMPILATION` is off and the device link disappears entirely.

__Precompiled headers cannot work for `.cu` files.__
nvcc preprocesses to a synthetic flat file before the host compiler runs, so no `#include` directive survives for a `.gch` to match.
Measured on the C++ half alone it was 111 s against 109 s, and nvcc `--threads` measured slower than none, 380 s against 366 s, because `-j` already saturates the machine.
A linker swap is equally pointless: the host link is 0.64 s, and the bytes come from `ptxas`, which no host linker replaces.

__Never add `#include <fmt/…>` to a header under `cuda/`, to any `*_device.cuh`, or to `mmff94_tables_device.cuh`, and never use `{fmt}` inside a `__device__`, `__global__` or `__host__ __device__` body.__

The boundary is not the file extension, and getting that wrong costs work in both directions.
`{fmt}` is already compiled by nvcc here — `smallfield.cu` is forced to `LANGUAGE CUDA` and reaches `fmt/chrono.h` transitively, with consteval-checked `logger.*` calls that build.
And there is no device-side `printf` anywhere in this repository; every `printf` in a `.cu` or `.cuh` is host-side, in a launch wrapper or an allocation-failure path, so all of it is migratable.
So the exclusion is a list of function bodies, enforced by a rule about includes.
A host-side failure path inside one of those headers takes a `logger_t const *` instead, as `cuda_error.hpp` already does.

Build settings are interface targets, never `CMAKE_<LANG>_FLAGS` mutations.
A global flag variable reaches every target in every subdirectory, and the `if(CMAKE_BUILD_TYPE STREQUAL "Release")` guard around one is invisible to multi-config generators, where that variable is empty — which silently dropped every optimization flag under Ninja Multi-Config and dropped `--expt-relaxed-constexpr` from every test `.cu` in Debug.
Generator expressions are evaluated per configuration and do not have either failure mode.

## Deliberately Not Rules

- __STL containers in headers.__ Other projects in this family ban them in public headers and route to a house container. SmallField is an application pipeline consuming RDKit, whose API returns `std::vector` throughout, so the boundary is already impure and there is nothing to migrate to.
- __`PointerAlignment`.__ This repo carries `Right`, the C-idiomatic form, in a C++ project where the family heuristic says `Left`. Changing it reformats the whole tree for no functional gain.
- __`clang-tidy` as a gate.__ Shipped advisory in `.clang-tidy`, deny-by-default so `__device__` code and intrinsics do not drown it. Too slow for a hook.
- __Style checks in CI.__ CI is for correctness. Hooks are for style.
