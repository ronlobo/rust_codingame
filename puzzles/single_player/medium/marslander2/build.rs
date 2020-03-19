/*!
This file bundles all rust bin and lib source code into a singlefile.rs in the
output directory.
*/
extern crate rustsourcebundler;

use std::path::Path;
use rustsourcebundler::Bundler;

fn main() {
    let mut bundler: Bundler = Bundler::new(
        Path::new("examples/main.rs"),
        Path::new("examples/bundle/main.rs"),
    );

    bundler.crate_name("marslander2");
    bundler.run();
}
