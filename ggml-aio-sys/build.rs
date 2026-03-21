#![allow(clippy::uninlined_format_args)]

extern crate bindgen;

use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if env::var_os("GGML_AIO_SYS_VERBOSE").is_some() {
        for (key, _) in env::vars() {
            if key.starts_with("CARGO_FEATURE_") {
                let feature = key
                    .strip_prefix("CARGO_FEATURE_")
                    .unwrap()
                    .to_lowercase()
                    .replace('_', "-");
                println!("cargo:warning=enabled feature: {feature}");
            }
        }
    }

    let target = env::var("TARGET").expect("TARGET must be set by Cargo");
    let arch: &str = target.split('-').next().expect("Invalid TARGET format");
    let is_android = target.contains("android");
    let is_windows_target = target.contains("windows");

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set by Cargo"));
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set");
    let cc_root = PathBuf::from(&manifest_dir).join("cc");

    let mut config = Config::new(&cc_root);

    if let Some(cpp_stdlib) = get_cpp_link_stdlib(&target) {
        println!("cargo:rustc-link-lib=dylib={cpp_stdlib}");
    }

    if is_android {
        println!("cargo:rerun-if-env-changed=ANDROID_NDK_HOME");
        println!("cargo:rerun-if-env-changed=NDK_HOME");

        let ndk_home = env::var("ANDROID_NDK_HOME")
            .or_else(|_| env::var("NDK_HOME"))
            .expect("ANDROID_NDK_HOME or NDK_HOME must be set for Android builds");
        let ndk = PathBuf::from(ndk_home);

        let android_abi = match arch {
            "aarch64" => "arm64-v8a",
            _ => panic!("Unsupported Android architecture: {arch}"),
        };
        config.cflag("-march=armv8.7a");
        config.cxxflag("-march=armv8.7a");

        let toolchain_cmake = ndk
            .join("build")
            .join("cmake")
            .join("android.toolchain.cmake");

        config.define("CMAKE_TOOLCHAIN_FILE", toolchain_cmake);

        config.define("ANDROID_ABI", android_abi);
        config.define("ANDROID_PLATFORM", "android-28");
        config.define("GGML_LLAMAFILE", "OFF");
    }

    // Link macOS Accelerate framework for matrix calculations
    if target.contains("apple") {
        if arch == "x86_64" {
            config.define("GGML_ACCELERATE", "OFF");
            config.define("GGML_BLAS", "OFF");
        } else {
            config.define("GGML_BLAS", "OFF");
            config.define("GGML_ACCELERATE", "ON");
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
        // qwen3-tts `coreml_code_predictor.mm` (static lib) — rustc must link CoreML on the final binary.
        println!("cargo:rustc-link-lib=framework=CoreML");
        #[cfg(feature = "metal")]
        {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
    }

    #[cfg(feature = "cuda")]
    {
        if is_android {
            panic!("CUDA is not supported on Android targets");
        }
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_ROOT");
        println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");
        println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");

        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-link-lib=cuda");

        let cuda_lib_dirs = find_cuda_helper::find_cuda_lib_dirs();
        if cuda_lib_dirs.is_empty() {
            panic!(
                "Could not find CUDA libraries; set CUDA_PATH, CUDA_ROOT, CUDA_TOOLKIT_ROOT_DIR, or CUDA_LIBRARY_PATH"
            );
        }
        for dir in cuda_lib_dirs {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
        if !is_windows_target {
            println!("cargo:rustc-link-lib=culibos");
        }
    }

    #[cfg(feature = "hipblas")]
    {
        if is_android {
            panic!("HIPBLAS is not supported on Android targets");
        }
        if is_windows_target {
            panic!("Due to a problem with the last revision of the ROCm 5.7 library, it is not possible to compile the library for the windows environment.\nSee https://github.com/ggerganov/ggml.cpp/issues/2202 for more details.");
        }
        println!("cargo:rustc-link-lib=hipblas");
        println!("cargo:rustc-link-lib=rocblas");
        println!("cargo:rustc-link-lib=amdhip64");
        println!("cargo:rerun-if-env-changed=HIP_PATH");
        let hip_path = match env::var("HIP_PATH") {
            Ok(path) => PathBuf::from(path),
            Err(_) => PathBuf::from("/opt/rocm"),
        };
        let hip_lib_path = hip_path.join("lib");
        println!("cargo:rustc-link-search=native={}", hip_lib_path.display());
    }

    #[cfg(feature = "openmp")]
    {
        if is_android {
            // OpenMP may require additional setup for Android
            config.define("GGML_OPENMP", "ON");
            println!("cargo:rustc-link-lib=gomp");
        } else if target.contains("gnu") {
            println!("cargo:rustc-link-lib=gomp");
        }
    }

    let mut bindings = bindgen::Builder::default().header("wrapper.h");

    #[cfg(feature = "metal")]
    {
        bindings = bindings.header(
            cc_root
                .join("ggml/include/ggml-metal.h")
                .display()
                .to_string(),
        );
    }

    #[cfg(feature = "mtmd")]
    {
        bindings = bindings
            .header("wrapper_mtmd.h")
            .allowlist_function("mtmd_.*")
            .allowlist_type("mtmd_.*")
            .allowlist_function("clip_.*")
            .allowlist_type("clip_.*");
    }

    bindings = bindings
        .clang_arg(format!("-I{}", cc_root.join("llama.cpp/include/").display()))
        .clang_arg(format!("-I{}", cc_root.join("llama.cpp/tools/mtmd/").display()))
        .clang_arg(format!("-I{}", cc_root.join("models/").display()))
        .clang_arg(format!("-I{}", cc_root.display()))
        .clang_arg(format!("-I{}", cc_root.join("ggml/include/").display()));

    if is_android {
        let ndk = PathBuf::from(
            env::var("ANDROID_NDK_HOME")
                .or_else(|_| env::var("NDK_HOME"))
                .expect("ANDROID_NDK_HOME or NDK_HOME must be set"),
        );
        let sysroot = ndk_llvm_sysroot(&ndk);
        bindings = bindings.clang_arg(format!("--sysroot={}", sysroot.display()));
    }

    let bindings = bindings
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_partialeq(true)
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_function("whisper.*")
        .allowlist_type("whisper.*")
        .allowlist_function("sense_voice.*")
        .allowlist_type("sense_voice.*")
        .allowlist_function("qwen3_tts_.*")
        .allowlist_type("Qwen3Tts.*")
        .prepend_enum_name(false)
        .generate()
        .expect("Failed to generate bindings");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=wrapper_mtmd.h");
    println!(
        "cargo:rerun-if-changed={}",
        cc_root.join("ggml/src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cc_root.join("llama.cpp/src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cc_root.join("llama.cpp/tools/mtmd").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cc_root.join("whisper.cpp/src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cc_root.join("sense-voice.cpp/src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cc_root.join("qwen3-tts.cpp/src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cc_root
            .join("qwen3-tts.cpp/src/qwen3tts_c_api.h")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cc_root
            .join("qwen3-tts.cpp/src/qwen3tts_c_api.cpp")
            .display()
    );

    let bindings_path = out.join("bindings.rs");
    bindings
        .write_to_file(bindings_path)
        .expect("Failed to write bindings");

    if env::var("DOCS_RS").is_ok() {
        return;
    }

    config
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .very_verbose(true)
        .pic(true);

    if target.contains("msvc") {
        config.cxxflag("/utf-8");
    }

    if cfg!(feature = "cuda") {
        config.define("GGML_CUDA", "ON");
    }

    if cfg!(feature = "hipblas") {
        config.define("GGML_HIPBLAS", "ON");
        config.define("CMAKE_C_COMPILER", "hipcc");
        config.define("CMAKE_CXX_COMPILER", "hipcc");
        println!("cargo:rerun-if-env-changed=AMDGPU_TARGETS");
        if let Ok(gpu_targets) = env::var("AMDGPU_TARGETS") {
            config.define("AMDGPU_TARGETS", gpu_targets);
        }
    }

    if cfg!(feature = "vulkan") {
        if is_android {
            println!("cargo:rustc-link-lib=vulkan");
            // Vulkan on Android uses the NDK's Vulkan headers
            let ndk_home = env::var("ANDROID_NDK_HOME")
                .or_else(|_| env::var("NDK_HOME"))
                .expect("ANDROID_NDK_HOME or NDK_HOME must be set");
            let vulkan_path = PathBuf::from(ndk_home).join("sources/third_party/vulkan/src/libs");
            if vulkan_path.exists() {
                println!("cargo:rustc-link-search=native={}", vulkan_path.display());
            }
        } else {
            config.define("GGML_VULKAN", "ON");
            if is_windows_target {
                println!("cargo:rerun-if-env-changed=VULKAN_SDK");
                println!("cargo:rustc-link-lib=vulkan-1");
                let vulkan_path = match env::var("VULKAN_SDK") {
                    Ok(path) => PathBuf::from(path),
                    Err(_) => panic!(
                        "Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set"
                    ),
                };
                let vulkan_lib_path = vulkan_path.join("Lib");
                println!("cargo:rustc-link-search=native={}", vulkan_lib_path.display());
            } else if target.contains("-apple-darwin") {
                println!("cargo:rerun-if-env-changed=VULKAN_SDK");
                println!("cargo:rustc-link-lib=vulkan");
                let vulkan_path = match env::var("VULKAN_SDK") {
                    Ok(path) => PathBuf::from(path),
                    Err(_) => panic!(
                        "Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set"
                    ),
                };
                let vulkan_lib_path = vulkan_path.join("lib");
                println!("cargo:rustc-link-search=native={}", vulkan_lib_path.display());
            } else {
                println!("cargo:rustc-link-lib=vulkan");
            }
        }
    }

    if cfg!(feature = "metal") {
        if is_android {
            panic!("Metal is not supported on Android targets");
        }
        config.define("GGML_METAL", "ON");
        config.define("GGML_METAL_NDEBUG", "ON");
        config.define("GGML_METAL_EMBED_LIBRARY", "ON");
    } else {
        config.define("GGML_METAL", "OFF");
    }

    if cfg!(debug_assertions) {
        config.define("CMAKE_BUILD_TYPE", "RelWithDebInfo");
    }

    for (key, value) in env::vars() {
        let is_useful_flag =
            key.starts_with("WHISPER_") || key.starts_with("LLAMA_") || key.starts_with("GGML_");
        let is_cmake_flag = key.starts_with("CMAKE_");
        if (is_useful_flag || is_cmake_flag) && !value.is_empty() {
            config.define(&key, &value);
        }
    }

    if cfg!(not(feature = "openmp")) {
        config.define("GGML_OPENMP", "OFF");
    }

    let destination = config.build();

    add_link_search_path(&out.join("build")).unwrap();

    println!("cargo:rustc-link-search=native={}", destination.display());
    println!("cargo:rustc-link-lib=static=whisper");
    println!("cargo:rustc-link-lib=static=sense-voice-core");
    println!("cargo:rustc-link-lib=static=qwen3tts");
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    if cfg!(feature = "vulkan") {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
    }
    if cfg!(feature = "metal") {
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }
    if cfg!(feature = "cuda") {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }

    // Build mtmd library directly with cc::Build
    if cfg!(feature = "mtmd") {
        let mtmd_src = cc_root.join("llama.cpp/tools/mtmd");
        let llama_src = cc_root.join("llama.cpp");
        let mut mtmd_build = cc::Build::new();
        mtmd_build
            .cpp(true)
            .include(&mtmd_src)
            .include(&llama_src)
            .include(llama_src.join("include"))
            .include(llama_src.join("ggml/include"))
            .include(llama_src.join("common"))
            .include(llama_src.join("vendor"))
            .include(cc_root.join("ggml/include"))
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-Wno-cast-qual")
            .pic(true);

        if target.contains("msvc") {
            mtmd_build.flag("/std:c++17");
        }

        // Collect all .cpp files in tools/mtmd and its subdirectories
        for entry in std::fs::read_dir(&mtmd_src).expect("Failed to read mtmd directory") {
            let entry = entry.expect("Failed to read entry");
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "cpp") {
                let filename = path.file_name().unwrap().to_str().unwrap();
                // Skip CLI / deprecation-warning binaries — we only want the library sources
                if filename == "mtmd-cli.cpp" || filename == "deprecation-warning.cpp" {
                    continue;
                }
                mtmd_build.file(&path);
            }
        }

        // Also include model files
        let models_dir = mtmd_src.join("models");
        if models_dir.exists() {
            for entry in std::fs::read_dir(&models_dir).expect("Failed to read models directory") {
                let entry = entry.expect("Failed to read entry");
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "cpp") {
                    mtmd_build.file(&path);
                }
            }
        }

        mtmd_build.compile("mtmd");
        println!("cargo:rustc-link-lib=static=mtmd");
    }
}

fn get_cpp_link_stdlib(target: &str) -> Option<&'static str> {
    if target.contains("msvc") {
        None
    } else if target.contains("apple") || target.contains("freebsd") || target.contains("openbsd") {
        Some("c++")
    } else if target.contains("android") {
        Some("c++_shared")
    } else {
        Some("stdc++")
    }
}

fn ndk_llvm_sysroot(ndk: &Path) -> PathBuf {
    let prebuilt = ndk.join("toolchains/llvm/prebuilt");
    let candidates: &[&str] = if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            &["darwin-aarch64", "darwin-x86_64"]
        } else {
            &["darwin-x86_64", "darwin-aarch64"]
        }
    } else if cfg!(target_os = "linux") {
        if cfg!(target_arch = "aarch64") {
            &["linux-aarch64", "linux-x86_64"]
        } else {
            &["linux-x86_64", "linux-aarch64"]
        }
    } else if cfg!(target_os = "windows") {
        &["windows-x86_64"]
    } else {
        &["linux-x86_64"]
    };

    for name in candidates {
        let sysroot = prebuilt.join(name).join("sysroot");
        if sysroot.is_dir() {
            return sysroot;
        }
    }

    panic!(
        "could not find NDK LLVM sysroot under {} (tried {:?})",
        prebuilt.display(),
        candidates
    );
}

fn add_link_search_path(dir: &Path) -> std::io::Result<()> {
    if dir.is_dir() {
        println!("cargo:rustc-link-search=native={}", dir.display());
        for entry in std::fs::read_dir(dir)? {
            add_link_search_path(&entry?.path())?;
        }
    }
    Ok(())
}
