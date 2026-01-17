#![allow(clippy::uninlined_format_args)]

extern crate bindgen;

use cmake::Config;
use std::env;
use std::path::PathBuf;

fn main() {
    // Iterate over all environment variables
    for (key, value) in env::vars() {
        if key.starts_with("CARGO_FEATURE_") {
            let feature = key
                .strip_prefix("CARGO_FEATURE_")
                .unwrap()
                .to_lowercase()
                .replace("_", "-");
            println!("Enabled feature: {}", feature);
        }
    }

    let target = env::var("TARGET").unwrap();
    let arch: &str = target.split('-').nth(0).expect("Invalid TARGET format");
    let is_android = target.contains("android");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");
    let cc_root = PathBuf::from(manifest_dir.to_string()).join("cc");

    let mut config = Config::new(&cc_root);

    // Link C++ standard library
    if let Some(cpp_stdlib) = get_cpp_link_stdlib(&target) {
        println!("cargo:rustc-link-lib=dylib={}", cpp_stdlib);
    }

    // Android-specific configuration
    if is_android {
        // Ensure NDK environment variables are set
        let ndk_home = env::var("ANDROID_NDK_HOME")
            .or_else(|_| env::var("NDK_HOME"))
            .expect("ANDROID_NDK_HOME or NDK_HOME must be set for Android builds");
        let ndk = PathBuf::from(ndk_home);

        // Map Rust target to Android ABI
        let android_abi = match arch {
            "aarch64" => "arm64-v8a",
            _ => panic!("Unsupported Android architecture: {}", arch),
        };

        if arch.contains("aarch64") {
            config.cflag("-march=armv8.7a");
            config.cxxflag("-march=armv8.7a");
        }  else {
            // Rather than guessing just fail.
            panic!("Unsupported Android target {arch}");
        }

        let toolchain_cmake = ndk
            .join("build")
            .join("cmake")
            .join("android.toolchain.cmake");

        // Set Android-specific flags
        config.define("CMAKE_TOOLCHAIN_FILE", toolchain_cmake);

        config.define("ANDROID_ABI", android_abi);
        config.define("ANDROID_PLATFORM", format!("android-28"));
        config.define("GGML_LLAMAFILE", "OFF");
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=c++_shared");
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
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-link-lib=cuda");
        cfg_if::cfg_if! {
            if #[cfg(target_os = "windows")] {
                let cuda_path = PathBuf::from(env::var("CUDA_PATH").unwrap()).join("lib/x64");
                println!("cargo:rustc-link-search={}", cuda_path.display());
            } else {
                println!("cargo:rustc-link-lib=culibos");
                println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
                println!("cargo:rustc-link-search=/usr/local/cuda/lib64/stubs");
                println!("cargo:rustc-link-search=/opt/cuda/lib64");
                println!("cargo:rustc-link-search=/opt/cuda/lib64/stubs");
            }
        }
    }

    #[cfg(feature = "hipblas")]
    {
        if is_android {
            panic!("HIPBLAS is not supported on Android targets");
        }
        println!("cargo:rustc-link-lib=hipblas");
        println!("cargo:rustc-link-lib=rocblas");
        println!("cargo:rustc-link-lib=amdhip64");
        cfg_if::cfg_if! {
            if #[cfg(target_os = "windows")] {
                panic!("Due to a problem with the last revision of the ROCm 5.7 library, it is not possible to compile the library for the windows environment.\nSee https://github.com/ggerganov/ggml.cpp/issues/2202 for more details.")
            } else {
                println!("cargo:rerun-if-env-changed=HIP_PATH");
                let hip_path = match env::var("HIP_PATH") {
                    Ok(path) => PathBuf::from(path),
                    Err(_) => PathBuf::from("/opt/rocm"),
                };
                let hip_lib_path = hip_path.join("lib");
                println!("cargo:rustc-link-search={}", hip_lib_path.display());
            }
        }
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
    let bindings = bindings.header(
        cc_root
            .join("ggml/include/ggml-metal.h")
            .display()
            .to_string(),
    );

    let bindings = bindings
        .clang_arg(format!("-I{}", cc_root.join("models/").display()))
        .clang_arg(format!("-I{}", cc_root.display()))
        .clang_arg(format!("-I{}", cc_root.join("ggml/include/").display()))
        // Add Android-specific include paths for bindgen
        .clang_arg(if is_android {
            let sysroot = PathBuf::from(
                env::var("ANDROID_NDK_HOME")
                    .or_else(|_| env::var("NDK_HOME"))
                    .expect("ANDROID_NDK_HOME or NDK_HOME must be set"),
            )
            .join("toolchains/llvm/prebuilt")
            .join(if cfg!(target_os = "windows") {
                "windows-x86_64"
            } else if cfg!(target_os = "macos") {
                "darwin-x86_64"
            } else {
                "linux-x86_64"
            })
            .join("sysroot");
            format!("--sysroot={}", sysroot.display())
        } else {
            "".to_string()
        })
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
        .prepend_enum_name(false)
        .generate()
        .expect("Failed to generate bindings");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!(
        "cargo:rerun-if-changed={}",
        cc_root.join("ggml/src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cc_root.join("models/llama.cpp/src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cc_root.join("models/whisper.cpp/src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cc_root.join("models/SenseVoice.cpp/src").display()
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

    if cfg!(target_os = "windows") {
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
                println!("cargo:rustc-link-search={}", vulkan_path.display());
            }
        } else {
            config.define("GGML_VULKAN", "ON");
            if cfg!(windows) {
                println!("cargo:rerun-if-env-changed=VULKAN_SDK");
                println!("cargo:rustc-link-lib=vulkan-1");
                let vulkan_path = match env::var("VULKAN_SDK") {
                    Ok(path) => PathBuf::from(path),
                    Err(_) => panic!(
                        "Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set"
                    ),
                };
                let vulkan_lib_path = vulkan_path.join("Lib");
                println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
            } else if cfg!(target_os = "macos") {
                println!("cargo:rerun-if-env-changed=VULKAN_SDK");
                println!("cargo:rustc-link-lib=vulkan");
                let vulkan_path = match env::var("VULKAN_SDK") {
                    Ok(path) => PathBuf::from(path),
                    Err(_) => panic!(
                        "Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set"
                    ),
                };
                let vulkan_lib_path = vulkan_path.join("lib");
                println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
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
        if is_useful_flag || is_cmake_flag {
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
}

fn get_cpp_link_stdlib(target: &str) -> Option<&'static str> {
    if target.contains("msvc") {
        None
    } else if target.contains("apple") || target.contains("freebsd") || target.contains("openbsd") {
        Some("c++")
    } else if target.contains("android") {
        Some("c++_shared") // Already correctly set for Android
    } else {
        Some("stdc++")
    }
}

fn add_link_search_path(dir: &std::path::Path) -> std::io::Result<()> {
    if dir.is_dir() {
        println!("cargo:rustc-link-search={}", dir.display());
        for entry in std::fs::read_dir(dir)? {
            add_link_search_path(&entry?.path())?;
        }
    }
    Ok(())
}
