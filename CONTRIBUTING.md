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

__The public release of 2026-08 was the one deliberate exception to the added-lines discipline.__
A whole-tree sweep corrected documentation that contradicted the code, deleted dead code, and normalised includes.
Reformatting whole files turns every legacy line into an added line, so the hook then reports violations the file has carried for months; that is the cost of a flag day and the reason the rule exists.
Outside a release sweep, format only the lines you changed — `git clang-format --staged` does exactly that.

## Commit Messages

The subject is `<Verb>: <Capitalized summary>`.
`Add`, `Improve`, and `Fix` carry almost everything.
`Make` is for build-system adaptations, `Docs` for documentation, and `Chore` for mechanical behavior-preserving churn — all three are rare.
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
template <typename allocator_, typename deallocator_>
class byte_buffer { … };

using pinned_byte_buffer_t = byte_buffer<pinned_alloc_t, pinned_dealloc_t>;
using device_byte_buffer_t = byte_buffer<device_alloc_t, device_dealloc_t>;
```

Concept names take neither suffix for the same reason.

__Everything this project declares is snake_case.__
No camelCase, no leading capital, no `SCREAMING_SNAKE` outside a macro, no `kPrefixed` constant.

__A name reached through a library object keeps that library's spelling.__
`sc->d_params[k].kbaIJK` is RDKit's field and renaming it does not compile, while the `kba_ijk` it is assigned into is ours.
The adapter in `parameters/mmff94_tables_rdkit.hpp` is the boundary: library spelling on the right, ours on the left.

Spell concepts out.
`index` not `idx`, `config` not `cfg`, `source` and `destination` not `src` and `dst`, `scratch` not `tmp`, `count` not `cnt`.

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

`#pragma region <name>` and `#pragma endregion <name>` mark sections, with plain words in the name — a dash makes `clang-format` reflow around it and the markers drift out of pairing.

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

__No pointers inside containers.__
`std::vector<T *>` and `std::vector<std::unique_ptr<T>>` are a pointer chase plus an allocation per element for something whose count is usually known and small.
Store values, or store indices into a flat arena.

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

__Inline code and symbol names take backticks, and neither `@c` nor `@ref` is used.__
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

__Three sinks, chosen by who reads the output.__
`src/` and `cli/` are operator-facing and long-running, so they use `logger_t` and want the wall clock.
`bench/` is read as a report, so it uses bare `fmt::println`.
`test/` uses the reporter and __not__ the logger, because a 34-character varying timestamp destroys column alignment and makes two runs undiffable, and ctest already timestamps.

## Performance Claims

__A performance number states its corpus, its sample size, and its weighting.__

A microbenchmark of an isolated access pattern is not an expected result.
This project measured a 6.9× improvement on the distance-geometry bounds read and quoted it as the headline.
It was real — at 128 atoms, which is __0.1 % of PubChem__.
Weighted by the measured atom-count distribution the same change is 2.47×, and end to end it was under 3 %.

So:

- Name the corpus and the sample size. `../USearchMolecules/data/pubchem` and `data/gdb13` are the standing ones; they have different molecule-size distributions and a blended figure hides which half moved.
- Weight any distribution-sensitive number by the measured distribution before quoting it.
- Compare A against B within one run. Absolute throughput drifts by 10–15 % on a loaded machine while the ratio inside a single run stays stable.
- Report the tail, not only the median. A shifted median and a heavy one-sided tail look identical otherwise, which is how an unpaired energy comparison hid a real defect here for months.
- Say what you could not measure. An 8-GPU claim from a single-GPU box is a projection, and should read as one.

## Environment

`pixi.toml` pins the only environment this is built and tested in, and `pyproject.toml`'s floors follow it rather than claiming a wider range.

__A user site-packages directory silently shadows the pixi environment.__
`~/.local/lib/python3.X/site-packages` takes precedence, so `import numpy` inside the pixi Python can resolve to a different version than the one pinned.
Check a version with `PYTHONNOUSERSITE=1` before believing it.

## CUDA

One host thread orchestrates devices, with per-device non-blocking streams and events.
Kernel launches are already asynchronous, cross-device ordering is `cudaStreamWaitEvent`, and completion is an event-sync loop, so a host thread adds only scheduling noise.

`__restrict__` on every pointer a kernel takes.

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
