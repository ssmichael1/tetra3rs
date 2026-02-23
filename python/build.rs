fn main() {
    pyo3_build_config::add_extension_module_link_args();

    // Embed git hash at compile time (falls back to "unknown" for sdist builds)
    let git_hash = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=TETRA3RS_GIT_HASH={git_hash}");

    // Re-run if HEAD changes (new commit)
    if let Ok(git_dir) = std::process::Command::new("git")
        .args(["rev-parse", "--git-dir"])
        .output()
    {
        if git_dir.status.success() {
            if let Ok(path) = String::from_utf8(git_dir.stdout) {
                let head = format!("{}/HEAD", path.trim());
                println!("cargo:rerun-if-changed={head}");
            }
        }
    }
}
