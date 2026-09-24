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

Consumer-specific kernels can be compiled alongside toolkit kernels with
`lightgpu_build::fatbin_modules`, using one fatbin/module per source file. This
preserves entry pruning and separate namespaces. See the build helper API and
consumer `build.rs` files for the exact invocation.

## Debugging shared-kernel regressions

The toolkit's self-test and `gpuinfo` completeness check do not replace testing a
consumer's complete model. A contract mismatch can pass small-tensor checks and
still fail in a full model. Validate both the shared operation and the consumer
wrapper, then test the complete engine on its reference fixture.
