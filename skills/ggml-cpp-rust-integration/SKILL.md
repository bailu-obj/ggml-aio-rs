---
name: ggml-cpp-rust-integration
description: Integrate a new GGML-related C/C++ library into a monorepo and expose it as a Rust crate. Use when a user provides a Git URL for a GGML-adjacent project and asks to vendor source into ggml-aio-sys/cc, align or upgrade GGML, wire CMake + build.rs, generate or handcraft FFI, implement a safe Rust wrapper crate, add workspace membership, and provide runnable examples.
---

# GGML C++ to Rust Integration

Integrate a third-party GGML-related C/C++ project into a workspace that already has `ggml-aio-sys` and Rust wrapper crates. Execute the workflow in order, then iterate until `cargo check` and example builds succeed.

## Workflow

1. **Collect required inputs and constraints**
2. **Vendor source into `ggml-aio-sys/cc`**
3. **Inspect sample usage + exported headers**
4. **Resolve GGML version compatibility**
5. **Plan C ABI / Rust FFI boundary**
6. **Wire CMake + `build.rs` + feature gates**
7. **Implement safe Rust wrapper crate**
8. **Add workspace membership + example**
9. **Verify build/link/runtime path**

Use `references/integration-checklist.md` as execution checklist and `references/ffi-wrapper-guide.md` for ABI/wrapper decisions.

## 1) Collect Inputs

Obtain:
- Git URL and preferred revision (tag/branch/commit)
- Target crate name (for new wrapper crate)
- Required model/runtime assets (if examples need external files)
- Whether this integration must avoid patching upstream code

If inputs are missing, ask concise clarifying questions before editing code.

## 2) Vendor the Library

- Clone/fetch the provided repository into `ggml-aio-sys/cc/<lib-name>`.
- Prefer pinning to a commit (record it in comments or docs).
- Keep upstream layout mostly intact to simplify future updates.
- Avoid ad-hoc file copies; preserve `.git`-free vendored sources.

## 3) Read Samples and Locate API Headers

- Inspect upstream examples, tests, and docs first.
- Identify:
  - Initialization/shutdown flow
  - Main context/object lifetimes
  - Inference entrypoints
  - Error model and return codes
- Locate exported public headers used by consumers.
- If no stable C API exists, define a thin C-compatible bridge layer.

## 4) Check GGML Version and Align

- Compare vendored library GGML version with current workspace GGML.
- If upstream requires newer GGML:
  - Update workspace GGML source in a controlled change.
  - Ensure all existing crates still compile.
  - Update CMake to allow selecting external/shared GGML implementation.
- If versions differ but are compatible:
  - Prefer single GGML in final link graph.
  - Avoid duplicate GGML symbols.

## 5) Plan Export Surface (C++ -> Rust)

Design a minimal stable ABI:
- Expose plain C functions (`extern "C"`), opaque pointers, POD config structs.
- Translate C++ exceptions to error codes; never let exceptions cross FFI.
- Define ownership functions explicitly (`create/free`, `*_new/*_destroy`).
- Keep strings and buffers ABI-safe (`const char*`, `(ptr, len)`).

If upstream API is not FFI-friendly:
- Add a small adapter library in C/C++.
- Patch CMake targets so adapter compiles and links cleanly.
- Prefer minimal, documented patches over broad refactors.

Detailed design rules: `references/ffi-wrapper-guide.md`.

## 6) Build System Integration (`ggml-aio-sys`)

Implement in `ggml-aio-sys`:
- Extend CMake build graph for new vendored source and optional adapter layer.
- Reuse existing public feature strategy where possible.
- Update `build.rs` to:
  - Configure feature flags
  - Trigger correct CMake options
  - Emit link directives for all required static/shared libs
  - Rebuild when relevant headers/sources change

Ensure build is deterministic and does not require manual local tweaks.

## 7) Rust Wrapper Crate

Under project root:
- Create `<new-lib>-cpp-rs` (or repo naming convention equivalent).
- Add low-level FFI bindings to sys crate exports.
- Add safe API layer:
  - RAII wrappers for opaque handles
  - `Result<T, E>` error mapping from C status codes
  - Lifetimes/ownership that prevent use-after-free
  - Thread-safety markers only when justified (`Send`/`Sync`)

Avoid exposing unsafe raw pointers in public ergonomic APIs.

## 8) Workspace + Example

- Add new crate to workspace `Cargo.toml`.
- Add at least one runnable example demonstrating end-to-end use.
- Keep example minimal but realistic: init, load/configure, run, print result, cleanup.
- Document required assets/env vars in crate README.

## 9) Verification

Run and fix failures:
- `cargo check` (workspace or affected crates)
- `cargo build -p ggml-aio-sys`
- `cargo build -p <new-wrapper-crate>`
- `cargo run -p <new-wrapper-crate> --example <example-name>` (if assets available)

If runtime execution is blocked by missing model assets, still verify compile/link path and document how to run fully.

## Deliverables

Produce:
- Vendored source under `ggml-aio-sys/cc/<lib-name>`
- Updated `ggml-aio-sys` CMake and `build.rs`
- New safe wrapper crate + README
- Workspace membership updates
- Example program
- Short integration notes (what changed, GGML version decision, known limitations)
