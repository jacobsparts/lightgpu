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

Consumer-specific kernels can be compiled alongside toolkit kernels with
`lightgpu_build::fatbin_modules`, using one fatbin/module per source file. This
preserves entry pruning and separate namespaces. See the build helper API and
consumer `build.rs` files for the exact invocation.

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
