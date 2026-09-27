# lightgpu toolkit maintenance notes

This document contains contributor and maintainer guidance that is intentionally
separate from the user-facing README.

## Adding a kernel to a live toolkit

When adding or changing a shared kernel, preserve the existing caller contracts.
A kernel must not acquire a new requirement that callers must satisfy implicitly.
For example, reduction scratch should use fixed storage sized for the largest
legal `blockDim` (1024), rather than requiring callers to provide dynamic shared
memory. The same principle applies to argument count and order, in-place versus
out-of-place behavior, and vector indexing dimensions.

A call site that compiles and runs is not necessarily correct. Audit the kernel
arity and read the operation contract for argument order and in-place behavior.
For a consumer regression, use that consumer's dump facility and compare the
first divergent stage against the original implementation. RMBG dump files carry
a 12-byte `(c,h,w)` header; account for it when reading them.

## Where a kernel belongs

The toolkit is the shared, generic layer. Put an operation in the toolkit when a
different model family could plausibly call it without changing its contract,
even if there is only one caller today. Keep an operation in the consumer when
its fusion, layout, tap order, or other contract is specific to one model family.

The practical test is:

* could another model family plausibly call this with no contract change? Put it
  in the toolkit;
* does it exist because of one architecture's fusion, layout, or tap order? Keep
  it in the project.

"Plausibly" is deliberately forward-looking and includes work that is planned
rather than present. A plain elementwise product had exactly one caller in the
family (`mx_mul`, in MAXIM) and still belongs here, because the gated
architectures need one on their own terms: NAFNet's SimpleGate multiplies the two
halves of a channel split, and every gMLP-style block multiplies a gate by a
value. The test is whether the OPERATION is generic, not whether there are two
call sites today - and the cost of waiting is a second copy that has to be
reconciled later, which is the thing this document exists to prevent.

### Writing one here first

Most of the kernels in section 12 of `cuda/kernels.cu` were written inside one
engine, proved there against that engine's own tests and fixture, and moved here
once they were 2-10x an op the toolkit already had. That is the intended order,
because a measurement is what decides a kernel and an engine is where the
geometries to measure against are. The move has three parts, all of them
mechanical:

1. copy the kernel into `cuda/kernels.cu`, dropping its `sc_`/`mx_`/`nf_` prefix
   for `lg_`, and write the accumulation order and the bias order on it. If it
   replaces an existing op, KEEP THAT ORDER EXACTLY - the toolkit's own
   `lg_conv1x1` folds the bias into the accumulator before the first multiply and
   `lg_linear` adds it after the last, and `lg_conv1x1_rb`/`lg_linear_rb`
   reproduce each, which is what lets a caller swap one for the other and compare
   by EQUALITY rather than by tolerance;
2. add it to `OPS` and `NAMES` in `src/ops/mod.rs`, a CPU twin in
   `src/ops/cpu.rs` and a case in `ops::cpu::selftest`. The twin is written from
   the kernel's stated contract rather than from the kernel's code, so the two
   can disagree; where the kernel is a replacement, demand equality, not a
   tolerance. `cargo run --release --bin gpuinfo` resolves every name in
   `NAMES` and runs the self-test, and it is the check that the move is complete;
3. leave the engine's original in place behind its environment switch until the
   A/B has been re-run from the toolkit's copy. The old kernel is the
   measurement's other arm and, in the case of a form the CPU backend was first
   matched to, the only counterexample that can still be re-run.

Two practical notes from doing this. Do NOT replace an existing op in place when
the new one needs a different BLOCK SHAPE: a register-blocked GEMM needs
`block = (16,16,1)` with the grid derived from its tile, while every caller of
`lg_linear` has `(16,16,1)` hard-coded with a grid of its own, so the promoted
kernel needs its own name.

The second note is about WHEN a promotion reaches anybody, and every consumer in
this family is in the same case: each declares `lightgpu` as a `git` dependency
AND pins a revision in its own `Cargo.lock`, and nothing here uses `[patch]` or
`[replace]`. A kernel added here therefore reaches NO consumer until that
consumer's lockfile is refreshed - a promotion is invisible to every engine until
somebody chooses to opt in, which is what makes promoting into a live toolkit
safe. (An earlier version of this note split the engines into `path`-dependent
and `git`-dependent, some arriving "as soon as it compiles"; there are no
`path`-dependent consumers left.) When a consumer DOES refresh, a build failing
at its `known_kernel` assertion is the normal first symptom of a name that moved,
not a mistake in the consumer. Commit and push here first: a consumer cannot see
an unpushed commit, whichever way its lockfile is refreshed.

Consumer-specific kernels can be compiled alongside toolkit kernels with
`lightgpu_build::fatbin_modules`, using one fatbin/module per source file. This
preserves entry pruning and separate namespaces. See the build helper API and
consumer `build.rs` files for the exact invocation.

### The same move on the CPU side

`src/ops/cpu.rs` is where the same argument applies to the other half of an
engine. A twin is the path a machine with no driver takes, so it is not a test
harness and is held to the same standard as the kernel - and the convolutions are
where that bites: in ifan-rs at 256x256 the 3x3s are 72.7% of a CPU pass, and
THREE engines had each written their own AVX2 + rayon copy of that one op. A
"plain Rust, obviously correct" twin was not the conservative choice there; it was
a fifth copy, and a slow one.

So a twin may be parallel and vectorised, and `conv1x1`/`conv3x3s1p1` are, under
two rules that are not negotiable:

* the SIMD path is resolved at RUN time (`is_x86_feature_detected!`) and compiled
  out on a target that is not x86-64. Nothing in this family sets
  `-C target-cpu=native` or raises the baseline, and a binary built here has to
  run on a machine without AVX2;
* a twin may not choose its own SUMMATION ORDER - see `cuda/CONVENTIONS.md`
  section 4. Parallelism and vector width do not touch a sum; a reordered sum
  does. `ops::cpu::selftest` asserts that the scalar inner loop through the
  parallel framing is BIT-IDENTICAL to a plain reference, and that the vector path
  is within the single FMA it issues.

`rayon` is a dependency of the toolkit for this reason and only this reason.
Every engine in the family that runs a real model already depends on it and uses
it for the same work (nine of ten; the tenth is a CPU-only toolset), so it adds no
crate to an engine build that did not already have one.

## Debugging shared-kernel regressions

The toolkit's self-test and `gpuinfo` completeness check do not replace testing a
consumer's complete model. A contract mismatch can pass small-tensor checks and
still fail in a full model. Validate both the shared operation and the consumer
wrapper, then test the complete engine on its reference fixture: a golden pair -
input and the output the upstream reference implementation produced from it -
kept with the consumer and checked by its own test suite. Backend agreement is a
debugging aid, not the correctness record: it cannot see a mistake both backends
share, and a number in a README with no command behind it is not evidence at all.

## Reading a result back

Download the buffer a caller can actually see, not the arena it happens to live
in. The rule is the same one that governs launches: the cost should be sized to
what the caller will read.

The failure mode this prevents is silent in development and expensive in use. In
this family the activations of a whole forward pass live in one device arena, and
downloading "the arena, because the output is somewhere inside it" works - every
test passes, every image comes out right - while moving orders of magnitude more
bytes than any caller reads. On MAXIM's 600x400 fixture the arena is 942.7 MiB and
the output image is 2.8 MiB, and the extra transfer was 0.4 s of a 1.9 s run and
~970 MiB of resident host memory, because the host side has to materialise whatever
it asks the device for.

Both halves matter, and the host half is the one that is easy to miss:

* transfer only the range `[offs[id], offs[id] + len(bufs[id]))` of the buffer the
  caller wants, the way `realesrgan-rs` and `rmbg-rs` `download` a single `Buf`,
  `lama-rs` downloads `out.buf`, and `locate-anything-rs` replaces a 610 KB round
  trip with an 8-byte readback where 8 bytes is all the caller needs;
* allocate the host arena to the size of what will be read into it. A `Vec` sized
  for the device arena is not free even when nothing looks at it: pages are faulted
  in on write, so the allocation - not the read - is what shows up in RSS and in
  system time.

The arena is a DEVICE-side layout: its size is a property of how the graph packs
activations, not of what the engine produces. A per-op comparison harness, or a
debug dump that names intermediate activations, legitimately wants all of it, and
should say so at the site where it reads - an explicit whole-arena read on the dump
path is clearer and cheaper than paying for it on every run. What should never
happen is a normal inference quietly paying for the dump path's convenience.
