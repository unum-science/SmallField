# SmallField Benchmarks

How to measure this project, and what the current numbers are.

For where the time goes and what to do about it see [`ROADMAP.md`](ROADMAP.md); for the canonical vocabulary — `parquet-shard`, `coprocessed-batch`, `embedding-wave`, `atom-limited-bin`, `pipeline-role`, `wave-step` — see [`GLOSSARY.md`](GLOSSARY.md).

## Harnesses

Four binaries measure four different things, and quoting one as another is the most common mistake made with this document.

| Binary                      | Measures                                            | Includes the Pipeline Roles                |
| :-------------------------- | :-------------------------------------------------- | :----------------------------------------- |
| `smallfield_generate`       | End-to-end corpus throughput, Parquet in and out    | Yes — reader, gpu-runner, fallback, writer |
| `smallfield_bench_profile`  | One `generate_conformers` pass, GPU only            | No                                         |
| `smallfield_bench_backends` | SmallField against RDKit CPU, plus the quality gate | No                                         |
| `smallfield_bench_mmff`     | The MMFF typing, extraction and CSR kernel trio     | No                                         |

__Only `smallfield_generate` measures the shipping system.__
The library harnesses skip the Parquet reader, the writer, and the RDKit fallback that re-embeds what the GPU refused.
That fallback is 42 % of a shard's serial wall clock on Enamine REAL and roughly 85 % on PubChem, so a library number quoted as pipeline throughput overstates the pipeline by up to 7×.

## Running a Benchmark

```bash
# Build. `dev` is the inner loop; `release` is the shipped fatbin.
cmake --preset dev -DSMALLFIELD_BUILD_CLI=ON
cmake --build build_dev --parallel

# End-to-end pipeline throughput — the number that describes the shipping system.
# Stage at least six shards into a folder, or the depth-5 ring never fills.
LD_PRELOAD=$PWD/.pixi/envs/default/lib/libjemalloc.so.2 \
  build_dev/smallfield_generate \
  --input <folder of parquet-shards> --output <folder on local disk> \
  --dataset real --num-conformers 3 --verbose

# GPU stage alone, no reader, writer or fallback
build_dev/smallfield_bench_profile --smi <corpus>.smi --conformers 3

# RDKit comparison and the paired quality gate
build_dev/smallfield_bench_backends --smi <corpus>.smi --conformers 3 --quality-samples 512
```

Three conformers per molecule is the production mode; K changes which resource binds, so a number at another K answers a different question.

`--verbose` prints the per-`embedding-wave` table, which is where host preprocess, launch, and per-stage GPU time become visible.
`--verbose=2` adds the per-device breakdown.

Write output to local NVMe.
The shared filesystem matches local sequential throughput but is 20 to 200× slower on small-file metadata operations, which lands entirely on the writer-role.

## Corpora

Datasets are reached through `data/` symlinks, which each machine makes for itself; never copy a dataset into the repository.

`bench/sample_corpora.py` draws a stratified SMILES sample when a harness wants a flat file rather than shards.
It walks every shard and takes a random row group from each, because rows within a shard share a deposition batch or a reagent series — the head of one shard is not a sample of the corpus.

```bash
python bench/sample_corpora.py --corpus real --count 100000 --out real-100k.smi
```

Corpus shape decides which `atom-limited-bins` exist at all, so a result on one does not transfer to another.

| Corpus  | Heavy Atoms p50 / max | With Hydrogens p50 / p99 / max | Over the 192-Atom Cap |
| :------ | :-------------------- | :----------------------------- | --------------------: |
| PubChem | 24 / 202              | 46 / 168 / 412                 |                0.68 % |
| GDB-13  | 13 / 13               | 27 / 34 / 39                   |                0.00 % |
| REAL    | 27 / 34               | 52 / 64 / 70                   |                0.00 % |

The atom cap applies __after__ explicit hydrogens, which is where the two counts diverge most: PubChem's p99 is 79 heavy atoms and 168 total.
That 0.68 % predicts the measured `too_many_atoms_k` rate of 0.69 % from an independent path.
REAL never reaches the 96, 112, 128 or 192 bins.

Shards are not ordered by molecule size — across REAL's 8 shards the mean with-hydrogen count spans 1.6 %, non-monotone, and PubChem gives a Spearman correlation of −0.01 against shard index.
Consuming shards in filename order does not bias a run.

## Reading the Output

__The progress line.__
`fallback 25,999→6,823 rec` names molecules attempted and molecules __recovered__, not molecules remaining, and both are molecules rather than conformer slots.

__Per-shard against cumulative.__
The per-shard `mol/s` charges one shard for its own serial GPU-then-fallback path.
The cumulative rate is what the pipeline delivers once shard N's fallback overlaps shard N+1's GPU work, and it climbs for the first several shards as the ring fills.
__Quote the cumulative figure__ from the closing `complete` line.

__The quality gate__ compares RDKit's MMFF94 evaluated on SmallField's own output geometry, so both columns describe identical coordinates and the paired per-conformer difference is the quantity that measures agreement.
Percentiles of the two distributions taken separately do not — a uniform offset, a heavy one-sided tail and partial cancellation all look the same that way.
Read the tail, not the median.

Both columns describe identical coordinates only while both describe the same molecule.
`generate_conformers` embeds a molecule's largest fragment, so the cross-validation pairs its readback against the `num_atoms_out` the same call filled and skips any sample whose re-parsed reference molecule disagrees — 8.8 % of PubChem's first shard, printed as `skipped atom_count_mismatch=` beside the `Gathered:` line.
Skipping is the honest answer: a salt form's reference molecule is a different molecule, not a repairable one.

The gate reads the heavy bin's clash rate before it reads any energy, because a readback addressing the wrong atoms produces a pair under 0.5 Å long before it produces a number anyone can attribute to a force-field term.
`failed_threshold=` on the verdict line names which of the four thresholds decided; `none` on a `PASS`.

__Exit codes__ are `0` success, `1` a check failed, `2` the command line was refused.

## Reference Numbers

One H100 80 GB HBM3, Intel Xeon Platinum 8468 with 16 threads, driver 570.195.03 and CUDA 12.8.
Sixteen host threads is this box's defining constraint: `compute_thread_budget_` sizes the gpu-runner-role and fallback-role at `num_procs - 4` each, so they are 12 wide and contend, where a many-threaded server would not.

__Pipeline, `smallfield_generate`, K=3, one GPU.__

| Corpus                        |    End-to-End | Molecule Success | Measured               |
| :---------------------------- | ------------: | ---------------: | :--------------------- |
| REAL, 8M molecules, 8 shards  | 10,968 conf/s |          99.99 % | `0096e82`, 2026-08-13  |
| REAL, 8M molecules, 8 shards  |  9,291 conf/s |          99.99 % | 2026-08-12             |
| PubChem, first shard __only__ |  1,326 conf/s |                — | 2026-08-12             |

The PubChem row is __one shard__, which is one `coprocessed-batch`, so all four roles run serially and nothing overlaps.
It is the right number for "how long does a cold shard take" and the wrong one for a pipeline share; do not compare it against the REAL rows.
A number quoted from it as though it described the pipeline is what produced the claim that the fallback is 86 % of wall clock, which the row decomposition below refutes on REAL.

### Where The Host Time Goes

`8M REAL, K=3, 8 shards, 1× H100`, at `0096e82` on 2026-08-13, from the `role decomposition` block the run prints itself.
Spans sum to 3,738.8 s over 2,188 s of wall, so the roles overlap 1.71× — this is steady state, not ramp.

| Span              |     Wall | Share of Summed | Per Molecule | Molecules |
| :---------------- | -------: | --------------: | -----------: | --------: |
| gpu-runner-role   | 2000.2 s |      __53.5 %__ |     250.0 µs | 8,000,000 |
| rdkit-fallback    | 1077.7 s |      __28.8 %__ |  20,088.2 µs |    53,647 |
| shape-descriptors |  404.1 s |          10.8 % |      50.5 µs | 8,000,000 |
| writer-role       |  255.4 s |           6.8 % |      31.9 µs | 8,000,000 |
| reader-role       |    1.4 s |          0.04 % |       0.2 µs | 8,000,000 |

__The fallback is 28.8 % on REAL, not 86 %.__
That 86 % is PubChem's, and the two do not transfer: REAL dispatches 0.67 % of molecules against PubChem's 2.60 %, and recovers 98.9 % of them against 26 %.
Per __retried__ molecule REAL costs 20 ms where PubChem costs 287 ms, so the corpus decides both how many molecules enter the tail and how expensive each one is.

__`shape-descriptors` is the second-largest host cost on REAL__, and it is not fallback work.
`compute_shape_descriptors_` runs over every molecule, outside the retry guard, and its body is a second full `SmilesToMol` + `addHs` plus four `SubstructMatch` calls over a SMILES the `preprocess-step` already parsed.
On a 200 K PubChem run the same span reads 1.6 %, because there the fallback is large enough to dwarf it — another figure that does not transfer between corpora.

__Library, `smallfield_bench_backends`, PubChem 20,000 molecules, K=1.__

| System                | conf/s |
| :-------------------- | -----: |
| RDKit CPU, 16 threads |   77.5 |
| SmallField, 1 GPU     |  4,027 |

That multiplier is against a 16-thread baseline and says as much about this host's CPU as about either system.

__Kernel resources__, from `cuobjdump --dump-resource-usage` on the `sm_90` build.

| Kernel                            | Registers | Stack | Note       |
| :-------------------------------- | --------: | ----: | :--------- |
| `etkdg_pipeline_warp_kernel`      |        69 |     0 | no spills  |
| `mmff_polish_warp_kernel`         |        64 |     0 | no spills  |
| `validate_polished_stereo_kernel` |        43 |     0 | no spills  |
| `csd_torsion_match_kernel`        |       255 |   192 | __spills__ |

A 32-thread block occupies at most 32 of a multiprocessor's 64 warp slots, so 50 % is the ceiling this shape reaches at any register count.
The polish kernel reaches it: at 64 registers nothing but the slot limit binds, and its shared memory is 3584 B at bin 64 against 228 KB per multiprocessor.
Its sibling is still register-bound, at 29 blocks per multiprocessor.

__Quality__ — nothing is recorded here, because the figures that were recorded measured the harness rather than the conformers.

The withdrawn table read 93.6 % finite, a paired P50 of −0.013 kcal/mol, a paired minimum of −5.14e+10 kcal/mol, and a failing heavy-bin gate.
The cross-validation re-derived its reference molecule from the raw SMILES with every fragment kept, then gathered coordinates at that molecule's atom count, while `generate_conformers` had embedded the largest fragment alone.
On a salt form the extra atoms come out of slot padding, which places them at the origin on top of the parent's, and RDKit's van der Waals term answers with a value nine orders of magnitude past any conformer energy.
`paired_delta` is SmallField minus RDKit, so the artefact lands in the negative tail and nowhere else — which is exactly where the withdrawn table pointed.
Multi-fragment rows are 8.8 % of PubChem's first shard, all 999,663 of them, and 22.4 % of the 100-128 atom bin the gate judges over a 40,000-row sample of the same shard, since a counterion also carries its molecule one bin up.

No force-field defect was involved.
The sentence this section carried — that the MMFF94 cubic bend factor turns negative past a 143° deviation and that FIRE has no energy-decrease check to reject the step — named a mechanism reproduction did not find, and is withdrawn with the numbers.

Record __the gate's own verdict line, verbatim__, when the harness is next run:

```
PASS heavy_bin_gate atoms=100-128 gathered=… compared=… clash_percent=… P50=… P95=… delta_P50=… failed_threshold=none
```

Four of its fields are the thresholds the verdict is rendered against and `failed_threshold` names the one that refused, so a recorded `FAIL` says what failed.
`gathered` against `compared` says how much of the drawn sample RDKit could type, which is a corpus property rather than a regression.
A statistic the gate does not read — `delta_min`, the marginal percentiles, the per-atom-count table — goes beside that line, never in place of it.
Quoting `delta_min` as the regression while the gate never looked at it is what let a harness defect read as a force-field one.

### Stereochemical Fidelity

The quality question energy cannot answer: does the geometry carry the handedness the input SMILES asserted?
Measured by clearing the graph's stereo, re-perceiving it from the coordinates with `AssignStereochemistryFrom3D`, and comparing `_CIPCode` on both sides — never raw chiral tags, which are relative to neighbour ordering.

First 400 molecules with a conformer from one shard of each corpus, 2026-08-13/14.

| Corpus and Shard                                |       Centres Inverted | Molecules Fully Right | Partial | Mirror |
| :---------------------------------------------- | ---------------------: | --------------------: | ------: | -----: |
| REAL `0821000000-0822000000`, 2026-07 corpus    | 69 / 345 = __20.00 %__ |                    74 |      40 |      5 |
| REAL `0821000000-0822000000`, at `0096e82`      |  21 / 345 = __6.09 %__ |                   105 |      11 |      3 |
| REAL __control__, RDKit ETKDGv3                 |   0 / 345 = __0.00 %__ |                     — |       — |      — |
| PubChem `0000000000-0001000000`, 2026-07 corpus | 47 / 202 = __23.27 %__ |                    28 |      23 |      3 |
| PubChem `0000000000-0001000000`, at `0096e82`   |  10 / 182 = __5.49 %__ |                    41 |       8 |      0 |
| PubChem __control__, RDKit ETKDGv3              |   0 / 193 = __0.00 %__ |                     — |       — |      — |

Two corpora of very different shape move the same way and by the same magnitude — 20.0 → 6.1 % and 23.3 → 5.5 % — which is the strongest evidence available that this is a real change rather than a sampling artefact.
Mirror-image failures reach zero on PubChem.

__E/Z double-bond stereo is a separate quantity__, measured separately below.
A chiral figure says nothing about the E/Z rate, and quoting one for the other is the same class of mistake as quoting a PubChem throughput number for the pipeline.

### E/Z Double-Bond Stereo

Both sides are normalised through `rdCIPLabeler::AssignCIPLabels`, which writes an `E` or `Z` onto the bond itself.
Comparing raw descriptors does not work: `AssignStereochemistryFrom3D` spells bond stereo as CIS/TRANS against whichever reference atoms it picked, where a SMILES parse spells it E/Z, so a direct comparison reports nearly everything wrong.

`PubChem 0000000000-0001000000`, same 400 molecules as above, of which 39 carry E/Z stereo at all.

| Corpus and Shard |    Stereo Bonds Wrong | All Wrong | Partial |
| :--------------- | --------------------: | --------: | ------: |
| 2026-07 corpus   | 25 / 48 = __52.08 %__ |         2 |       3 |
| at `0096e82`     |   3 / 40 = __7.50 %__ |         0 |       0 |
| __control__      |   0 / 48 = __0.00 %__ |         — |       — |

__The sample is small and the interval is wide.__
Forty-eight bonds is what 400 drug-like molecules yield, because most carry no E/Z stereo at all.
The direction is unambiguous and every multi-bond failure mode reaches zero, but do not treat 7.50 % as a precise rate — it rests on three bonds.
The chiral figures above rest on 345 and 202 centres and are firm; this one wants a larger draw before it carries weight.

__Three E/Z Rates Are Quoted In This Project, And They Are Three Different Measurements.__
They do not average and they do not convert into one another, so each is carried with the configuration and the sample it was taken on.

| Rate        | Configuration Scored                                                    | Sample                                                                                       | Recorded In                        |
| :---------- | :---------------------------------------------------------------------- | :------------------------------------------------------------------------------------------- | :--------------------------------- |
| __52.08 %__ | The 2026-07 3D corpus, produced without a post-polish stereo check      | 25 of 48 declared bonds over 400 PubChem molecules                                           | the table above                    |
| __6.441 %__ | Current code with the chiral-volume target and no post-polish E/Z check | declared bonds of the 5,000 molecules scored out of one 20,000-molecule PubChem shard at K=3 | _Closing The Tetrahedral Residual_ |
| __0.171 %__ | Current code as it ships, post-polish E/Z check included                | 1 of 584 declared bonds, same 5,000 molecules                                                | _Closing The Tetrahedral Residual_ |

The first two differ by producing code and by sample, not by method — both score declared bonds against `AssignCIPLabels` on the geometry.
The 52.08 % row is a property of stored data and moves only when the corpus is regenerated; the other two move with the pipeline.

`DEFECTS.md` carries a fourth figure, 279 of 4,921 bonds and 5.67 % on PubChem with 17 of 360 and 4.72 % on REAL, over a 20,000-molecule run at K=3 scoring every declared bond of every slot returned `success_k`.
That measurement was taken without the post-polish E/Z check and therefore describes the 6.441 % configuration, which its agreement in magnitude confirms.
What it needs is the configuration named beside the number and a pointer to the 0.171 % row, since as written it reads as a description of what ships.

__Run the control first and run it every time.__
It scores RDKit's own ETKDGv3 geometry of the same molecules through the identical code path, and it must come back at zero.
A non-zero control means the instrument is broken rather than the pipeline, which is the failure mode that produced — and then hid — the withdrawn energy table above.

Two things this measures that a rate alone does not.
__Partial__ counts molecules with two or more centres where some inverted and some did not, which is a per-constraint embedding failure.
__Mirror__ counts molecules where every centre inverted together and the conformer is the enantiomer of what was asked for, repairable by a single reflection.
They have different fixes, so the shape is worth more than the percentage: the recompute cut partial failures 40 → 11 while mirrors barely moved, so what improved is per-centre constraint satisfaction rather than a global handedness choice.

6.09 % is not zero, and the residual is mostly partial.

### Closing The Tetrahedral Residual

Measured in-tree by `bench/validate_stereo.py`, which runs the control every time and refuses a run whose control is non-zero.
One 20,000-molecule PubChem shard through `smallfield_generate` at K=3, scored over 5,000 molecules selected by `input_row` at an even stride so every configuration covers the same molecules.
`1× H100`, 8 physical cores, 2026-08-14.

| Configuration                             |                    Centres Inverted | E/Z Inverted | Fallback Molecules |      Wall |
| :---------------------------------------- | ----------------------------------: | -----------: | -----------------: | --------: |
| before                                    | __5.542 %__ _(600-molecule sample)_ |            — |                171 |    60.1 s |
| chiral volume target only                 |                         __2.351 %__ |      6.441 % |                 66 |    57.2 s |
| target + post-polish chirality check      |                         __0.080 %__ |      6.441 % |                365 |   118.8 s |
| __+ post-polish E/Z check__ — __shipped__ |                         __0.044 %__ |  __0.171 %__ |                599 |   108.2 s |
| target + torsion-stage repair             |                         __4.918 %__ |            — |              __0__ | __8.3 s__ |
| target + repair + post-polish check       |                         __0.000 %__ |            — |                449 |   111.3 s |

The E/Z column is the quantity `count_invalid_chiral_centers_` is structurally blind to: it counts tetrahedral centres, and no chiral descriptor is ever emitted for a double bond, so `stereo_failure_k` was unreachable for one until the shipped row.
One bond in 584, from one in sixteen.
The check reads the sign of `cos φ` over the two recorded reference substituents and ignores any bond twisted more than 60° from planar, because such a bond carries no cis-or-trans answer to compare against — 90 % of declared bonds land inside that band.

__Read the rows together, not separately, and note that the best correctness number is not what ships.__
The repair pass re-enters the distance geometry functor in three dimensions when the final gate fails, rather than abandoning the attempt.
On its own it __doubles__ inversion, because it ships the marginal conformers the CPU fallback was fixing correctly, and its 7.2× wall-clock gain is bought with them.
Behind the post-polish check it reaches zero.
Measuring either piece alone is misleading in opposite directions, which cost four runs to establish.

__It is still refused.__
Its correctness contribution over the shipped pair is two centres in 2,500, inside the noise.
Its wall-clock gain is 6 %, measured on a box where the fallback is 93 % of wall — which is not the production shape, and on the 188-thread machine that gain largely disappears.
Against those it spends eight more bytes of stack on both Blackwell targets, where `CONTRIBUTING.md` treats any movement as a revert and this box cannot measure the consequence.
Weak benefit on the wrong machine shape against a real cost on a priority target.

__365 dispatches per 20,000 is not new breakage.__
It is 1.8 % of molecules that were already shipping with a centre MMFF polish had inverted after the last gate, now visible for the first time.
Whether that cost matters depends on the box, the same way the repair's gain does.

### A Centre Carries Two Signed Volumes, And They Are Not The Same Number

The same 19,952-molecule PubChem shard through `smallfield_generate` at K=3, scored over 10,000 molecules by `bench/validate_stereo.py`.
`1× H100`, 8 physical cores, 2026-08-16.
The control reads 0.000 % on every row.

| Configuration                            | Declared Centres | Centres Inverted | Fallback Molecules | Energy p90 | Energy Max |
| :--------------------------------------- | ---------------: | ---------------: | -----------------: | ---------: | ---------: |
| post-polish check only — __was shipped__ |            4,607 |      __0.065 %__ |                362 |  __130.6__ |  __6,462__ |
| + MMFF chiral restraint                  |            4,924 |      __0.772 %__ |            __153__ |      132.9 |      6,463 |
| __+ centre-based read__ — __shipped__    |            4,766 |      __0.021 %__ |                257 |  __131.2__ |      6,463 |

__The restraint alone makes the shipped stereochemistry worse, and not because it fails.__
Its 38 offenders hold a signed volume of 4.86 to 6.33 cubic angstrom, median 4.93, against the restraint's floor of 5 — parked exactly where it stops pushing — and 17 of 17 that a control embedding could resolve carry the __correct__ sign.
So the geometry satisfies the constraint and RDKit still reads it inverted, because the two are reading different expressions.

`DistGeom::ChiralSet` takes the triple product over the four neighbours from the __fourth__, which is what the embedding and the restraint hold.
RDKit's `assignChiralTypesFrom3D` takes it over the first three from the __centre atom__ and never reads the fourth at all.
The two agree for any centre whose atom lies on the same side of its first three neighbours' plane as the fourth, which is every ordinary tetrahedron.
The restraint moves the four neighbours and applies no force to the centre, so it reaches the geometries where they do not: 44 of 4,917 declared centres come out split, and RDKit reads 37 of them inverted.

Half-precision narrowing was the first suspect and is not the cause.
At those magnitudes the fp16 coordinate quantum is about 4e-3 Å, worth under 0.05 cubic angstrom of volume, and the offenders sit five cubic angstrom from a sign change.

__The fix is in the reader, not the floor.__
`validate_polished_stereo_kernel` now takes both volumes, so a slot ships only when the stage that produced the geometry and the stage that will perceive it agree on what it is.
It costs 34 more fallback dispatches per 10,000 molecules and leaves the restraint's throughput gain intact — 257 dispatches against the 362 the post-polish check alone pays.
`mmff_polish_warp_kernel` is untouched — 63 registers and no stack on `sm_90`, 71 and 8 bytes on `sm_100`, both where the restraint left them.
`validate_polished_stereo_kernel` lands at 47 registers and no stack on `sm_90`, and moves 40/0 → 48/8 on `sm_100`, against the 256 per thread its `__launch_bounds__(32, 8)` allows.

### Restraining Only One Of The Two Volumes Costs REAL 1.586 % Of Its Molecules

The section above fixed PubChem's stereochemistry by teaching the reader about the second volume.
It did not fix the reason there is a second reading to disagree with, and on Enamine REAL — the corpus about to be encoded — that is 1.553 % of a full 1,000,000-molecule shard dispatched to the CPU embedder as `polish_inverted_center`, against 1.535 % before the restraint landed.
The refusal is correct and it is nearly the whole fallback bill: 15,527 of 19,868 dispatches, and 410 s of a 555 s shard.

__A centre's two volumes share one plane normal, which is what makes holding both cheap.__
The signed volume of a tetrahedron over a fixed base triangle is affine in its apex, so with `N = (r_1 - r_0) × (r_2 - r_0)` the pivot-based volume is `(r_0 - r_3) · N` and the centre-based one is `(r_0 - c) · N`.
One cross product serves both, each term's derivative with respect to its own apex is `-N`, and that second `-N` is the force on the centre atom the old expression could not produce.

__The two floors cannot be one number.__
Over 7,649 declared centres of 8,000 shipped GPU rows of a REAL slice, `|V|` reads:

| Expression   | p0.5 |   p1 |   p5 |      p50 |  p90 |
| :----------- | ---: | ---: | ---: | -------: | ---: |
| pivot-based  | 6.07 | 6.11 | 6.34 | __8.38__ | 8.71 |
| centre-based | 1.53 | 2.08 | 2.25 | __2.57__ | 2.91 |

`minimum_volume` is 5 for a four-coordinate centre, so transplanted whole it would sit above the 37th percentile of the centre-based distribution and bend a third of the corpus.
At `mmff_center_volume_fraction_k = 0.3` the floor is 1.5, which is that distribution's own 0.5th percentile: the flat bottom is inactive for 99.5 % of centres that already ship, and 1.5 cubic angstrom is thirty times what the half-precision coordinate quantum is worth.
A three-coordinate centre is stored as its own fourth neighbour, so the two expressions are one there and the smaller floor can never be the binding one — confirmed at `collect_chiral_centers`, and all 7,649 centres of this slice are four-coordinate, so the case is covered by fixture rather than by corpus.

__The result, on a 100,000-molecule stride sample of REAL shard 0 at K=3, `1× H100`, Release `sm_90`, 2026-08-16.__
The baseline column reproduces the full-shard rate to within 0.03 points.

| Measurement                                  | pivot volume only | __both volumes__ |
| :------------------------------------------- | ----------------: | ---------------: |
| `polish_inverted_center`, share of molecules |       __1.586 %__ |      __0.000 %__ |
| `stereo_failure`                             |               416 |              453 |
| `polish_inverted_bond`                       |                 7 |                9 |
| fallback molecules, all causes               |             2,009 |          __462__ |
| `rdkit-fallback` wall                        |            41.7 s |        __6.7 s__ |
| `gpu-runner-role` wall                       |            16.1 s |           15.0 s |
| __end to end__                               |   __1,653 mol/s__ |  __3,919 mol/s__ |
| conformer energy p90                         |             100.9 |            103.3 |
| conformer energy max                         |             1,518 |            2,420 |

The `stereo_failure` and `polish_inverted_bond` rises are bookkeeping, not regression: a cause is tallied once per molecule, so a molecule refused for both reasons was counted under the centre cause and is now counted under the other one.

__Geometry is almost untouched, and the tail that moves is the tail that was being refused.__
Paired by input row over 299,921 slots both configurations ship, 81.3 % are bit-identical — the flat bottom never binds — and of the 18.7 % that move the median is +0.04 and the p90 +1.99 kcal/mol.
The p99 of +538 is the slots the baseline hands to RDKit's geometry rather than ours, which is the cost of shipping them instead of refusing them.

__Fidelity is held, not traded.__
On the REAL slice itself `bench/validate_stereo.py` reads 0.000 % of declared centres inverted both before and after — 8,810 declared centres against 8,586 — on a control of 0.000 %.
The baseline reached that number by refusing 1.586 % of molecules to the CPU; this reaches it by not needing to.
On the same 20,000-molecule PubChem slice the rate goes 0.021 % → __0.000 %__ over 4,610 declared centres, control 0.000 %, and `polish_inverted_center` goes 114 → 0.
E/Z reads 0.163 % there against a control of 0.202 %, both two bonds of a ~1,200 draw and both inside the noise this file already cautions about at that sample size; REAL declares 72 bonds over 10,000 molecules and answers nothing about E/Z at all.

`mmff_polish_warp_kernel` does not move on any architecture — 63 registers and no stack on `sm_90`, 71 and 8 bytes on `sm_100`, 79 and 8 bytes on `sm_120`, each byte-identical to the same build with the second term removed.
The pivot path lost the three cross products it used to sum and gained a shared normal, which is what pays for the second term's two.
`compute-sanitizer --tool racecheck` reports 0 hazards over `smallfield_test_pipeline`.

### The Distance-Geometry Budget Is Not Free, And Still Worth Raising

Doubling the two four-dimensional stages — distance geometry 100 → 200 iterations, fourth-dimension crush 50 → 100 — on the same 20,000-molecule PubChem shard.

| Measurement                                  |      before |       after |       delta |
| :------------------------------------------- | ----------: | ----------: | ----------: |
| `stereo_failure`, `smallfield_bench_backends` |     1.830 % |     1.240 % |   __−32 %__ |
| GPU arm, same harness                        |    3,649 ms |    4,225 ms | __+15.8 %__ |
| fallback dispatches, full CLI at K=3         |         599 |         401 |   __−33 %__ |
| `gpu-runner` wall, full CLI                  |       5.6 s |       7.2 s |       +29 % |
| __end-to-end wall, full CLI__                | __108.2 s__ | __106.3 s__ |  __−1.8 %__ |

__The stage is not the rounding error it is often called.__
The ETKDG kernel's 3.8 % of wall is measured on REAL at K=3 with the CPU fallback folded in; against a GPU-only arm the same two stages are roughly 16 %.
Quoting the first number to justify spending in the second is the same category error as quoting a PubChem throughput figure for the pipeline.

__It is still the right trade here, for a different reason than cheapness.__
The geometry these two stages hand forward is what the reflection-blind torsion stage works from and what the stereo gates then judge, so better input means fewer conformers arrive in a state worth refusing — and a refusal costs the CPU embedder 240 ms against microseconds on the GPU.
On this shape the fallback is 92 % of wall and the trade is clearly positive.
On a GPU-bound REAL run, with a 0.347 % refusal rate and far less to recover, it would most likely be a net loss.

__E/Z is not what this budget buys, and no E/Z figure was taken across the doubling.__
Nothing in this table is evidence either way about it.
The two rates that bracket the quantity are 6.441 % without the post-polish check and 0.171 % with it, both from _Closing The Tetrahedral Residual_ and both from a different experiment on a different sample.
A double bond reaches `stereo_failure_k` only through that check: `count_invalid_chiral_centers_` counts tetrahedral centres, and no chiral descriptor is ever emitted for a bond.

One hypothesis was tested and refused: `molecule_descriptor.hpp` maps RDKit's CIP spelling and its reference-atom spelling onto one flag, which would build the wrong isomer wherever the recorded stereo atoms are not the CIP-preferred pair — measured over 4,493 declared bonds, __0 disagree__, because RDKit sets those atoms to the CIP pair on the SMILES path.
The conflation is latent, not live.

## Profiling

`ncu` and `nsys` crash any binary linked against jemalloc, with SIGSEGV inside `cudaGetDeviceCount` from unbounded recursion between jemalloc's malloc interposition and the profiler's `dlopen` hooks.
Preloading or not preloading it proves nothing, because `SMALLFIELD_USE_JEMALLOC=ON` puts it in `DT_NEEDED` regardless.

```bash
cmake -B build_nojemalloc -DCMAKE_BUILD_TYPE=Release -DSMALLFIELD_CUDA_ARCH="90-real" \
      -DSMALLFIELD_CUDA_LINEINFO=ON -DSMALLFIELD_USE_JEMALLOC=OFF
```

`ncu` needs `sudo` because `RmProfilingAdminOnly` is 1, and needs Nsight Compute __2025.1.1__ — version 2026.1.1 reports a driver incompatibility against 570.195.03 and fails multi-pass replay.
The binary still exits 11 during teardown under the profiler, after printing its full result and after every launch is captured, so a profiler-driven check must ignore the exit code.

The `profile` preset adds CUDA line info and NVTX ranges, so an `nsys -t cuda,nvtx` capture attributes host time to the four pipeline-roles rather than showing an anonymous CUDA timeline.

Device codegen is unaffected by the allocator, which is what makes a jemalloc-free measurement valid: `cuobjdump --dump-resource-usage` is byte-identical between builds.

## What Invalidates a Number

__Too few shards.__
The reader flushes a `coprocessed-batch` as soon as it reaches the target, which a 1,000,000-row shard does alone, so one shard is one batch and the handoffs are depth-5.
A three-shard run is all ramp and drain and measures roughly half what eight shards measure.

__The wrong harness.__
See the table at the top.

__A cold first shard.__
Wave 0 preprocess costs about 20× wave 12's as caches warm.

__A corpus that does not fill the machine.__
Achieved occupancy is grid-bound below roughly 50,000 conformers per launch — 1.6 % at 200 molecules against 40.9 % at 100,000 — so small-corpus occupancy says nothing about production.

__A profiled run's wall clock.__
Replay and clock locking inflate `gpu__time_duration.sum` well past the unprofiled time.
Never read it as throughput.

__A microbenchmark of one access pattern.__
This project once measured 6.9× on the bounds read and quoted it as a headline.
It was real at 128 atoms, which is 0.1 % of PubChem, and worth under 3 % end to end.
Weight anything distribution-sensitive by the measured atom-count distribution before quoting it.
