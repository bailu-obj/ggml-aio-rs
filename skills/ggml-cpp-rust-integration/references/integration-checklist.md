# Integration Checklist

Use this as an execution checklist during implementation.

## 0. Inputs

- [ ] Git URL provided
- [ ] Revision pinned (commit/tag)
- [ ] Target Rust crate name decided
- [ ] Runtime asset requirements clarified

## 1. Source Import

- [ ] Vendor upstream source into `ggml-aio-sys/cc/<lib-name>`
- [ ] Preserve upstream structure
- [ ] Record upstream revision in docs/comments

## 2. API Discovery

- [ ] Read examples and docs
- [ ] Identify lifecycle functions (init/use/free)
- [ ] Locate public headers
- [ ] Confirm error handling conventions

## 3. GGML Alignment

- [ ] Determine upstream GGML baseline
- [ ] Compare with workspace GGML
- [ ] If upgrade needed, update once and validate existing crates
- [ ] Ensure single GGML symbol provider in final link graph

## 4. Build System

- [ ] Add/adjust CMake targets
- [ ] Add optional adapter bridge target when needed
- [ ] Update `build.rs` for features, options, rerun hints
- [ ] Verify link directives cover all transitive deps

## 5. Rust Binding Layers

- [ ] Low-level FFI declarations complete
- [ ] Unsafe boundary isolated to small module(s)
- [ ] Safe wrapper provides RAII and `Result`-based errors
- [ ] Public API avoids raw pointer usage

## 6. Workspace + Docs + Example

- [ ] Add crate to workspace
- [ ] Add README usage notes and prerequisites
- [ ] Add at least one runnable example

## 7. Validation

- [ ] `cargo check`
- [ ] `cargo build -p ggml-aio-sys`
- [ ] `cargo build -p <wrapper-crate>`
- [ ] `cargo run --example ...` or documented runtime constraints
