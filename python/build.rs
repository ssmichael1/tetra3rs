fn main() {
    pyo3_build_config::add_extension_module_link_args();

    // Embed git hash + dirty flag at compile time. Falls back to "unknown"
    // for sdist builds (no .git directory available).
    let hash = run_git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let dirty = run_git(&["status", "--porcelain"])
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let revision = if dirty {
        format!("{hash}-dirty")
    } else {
        hash
    };

    println!("cargo:rustc-env=TETRA3RS_GIT_HASH={revision}");

    // Re-run when HEAD changes (new commits, branch switches) and when the
    // index changes (staging — covers most edits that affect the dirty
    // flag without requiring a full re-run on every working-tree write).
    if let Some(git_dir) = run_git(&["rev-parse", "--git-dir"]) {
        println!("cargo:rerun-if-changed={git_dir}/HEAD");
        println!("cargo:rerun-if-changed={git_dir}/index");
    }
}

fn run_git(args: &[&str]) -> Option<String> {
    let output = std::process::Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_string())
}
