workspace(name = "nexus_core")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Rules for C++
http_archive(
    name = "rules_cc",
    urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.0.9/rules_cc-0.0.9.tar.gz"],
    sha256 = "2037875b9a4456dce4a79d112a8ae101e479d507",
)

# Rules for Rust
http_archive(
    name = "rules_rust",
    urls = ["https://github.com/bazelbuild/rules_rust/releases/download/0.42.1/rules_rust-v0.42.1.tar.gz"],
    sha256 = "c25cb7bb8cc2b5757d23d8c1e40d04bde6edcfa81c",
)

load("@rules_rust//rust:repositories.bzl", "rules_rust_dependencies", "rust_register_toolchains")

rules_rust_dependencies()

rust_register_toolchains(
    edition = "2021",
    versions = ["1.75.0"],
)

# Rules for Python
http_archive(
    name = "rules_python",
    urls = ["https://github.com/bazelbuild/rules_python/releases/download/0.31.0/rules_python-0.31.0.tar.gz"],
    sha256 = "c68bdc4fbec207d5718a7a92ccff8b6af97e8b835e",
)
