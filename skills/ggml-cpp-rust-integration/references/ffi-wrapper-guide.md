# FFI Wrapper Guide

Use this guide when converting C++ library APIs into safe Rust interfaces.

## C ABI Contract

- Export only `extern "C"` functions.
- Use opaque structs for handles:
  - `typedef struct mylib_ctx mylib_ctx;`
- Expose explicit ownership APIs:
  - `mylib_ctx* mylib_create(const mylib_config* cfg);`
  - `void mylib_destroy(mylib_ctx* ctx);`
- Return explicit status codes for fallible functions.
- Provide `mylib_last_error()` style retrieval if needed.

## C++ Bridge Rules

- Do not allow C++ exceptions to cross FFI boundary.
- Catch all exceptions in bridge layer and map to error code.
- Convert C++ strings/containers into C-compatible representations.
- Keep bridge layer thin and deterministic.

## Rust Layering

Implement three layers:

1. **Sys/raw layer**
   - Generated or handwritten `extern "C"` bindings
   - No safety guarantees

2. **Unsafe internal wrapper**
   - Small module enforcing call ordering and pointer invariants
   - Still `unsafe` internally

3. **Public safe API**
   - Owns resources via RAII (`Drop`)
   - Uses typed configs/builders
   - Returns `Result<T, Error>`
   - Avoids exposing raw pointers

## Memory and Lifetimes

- Define clear ownership for every pointer parameter.
- For borrowed buffers, pass `(ptr, len)` and validate non-null/len.
- For returned buffers, provide explicit free function from same allocator domain.
- Avoid global mutable state where possible.

## Thread Safety

- Mark wrapper types `Send`/`Sync` only after confirming upstream guarantees.
- If uncertain, keep types non-`Send` and document constraints.

## API Shape Recommendations

- Prefer coarse-grained operations over many tiny FFI calls.
- Use builder/config structs for options.
- Keep zero-copy APIs internal until safety story is proven.
- Preserve deterministic cleanup on all error paths.

## Build and Linking Tips

- Ensure bridge target and upstream libs are linked in correct order.
- Prefer one GGML provider to avoid duplicate symbols.
- Keep feature flags aligned between CMake and Cargo.
- Emit full rerun metadata in `build.rs` for headers/sources/options.
