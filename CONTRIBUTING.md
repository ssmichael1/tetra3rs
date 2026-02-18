# Contributing to tetra3rs

Contributions are welcome! Whether it's bug reports, feature requests, documentation improvements, or code changes to improve speed and robustness

## Getting started

1. Fork the repository and clone your fork
2. Install Rust (stable toolchain)
3. Download the Hipparcos catalog:
   ```sh
   mkdir -p data
   curl -o data/hip2.dat.gz "http://cdsarc.u-strasbg.fr/ftp/I/311/hip2.dat.gz"
   gunzip data/hip2.dat.gz
   ```
4. Run the tests:
   ```sh
   cargo test                          # unit tests
   cargo test --features image         # integration tests (downloads test data on first run)
   ```

## Submitting changes

1. Create a branch for your changes
2. Make your changes and ensure all tests pass
3. Open a pull request against `main` with a clear description of what you changed and why

## Reporting issues

Open an issue on GitHub. For bugs, please include:

- What you expected to happen
- What actually happened
- Steps to reproduce
- Rust version (`rustc --version`)

## Areas where help is appreciated

- Support for additional star catalogs
- Distortion model support (SIP, polynomial) in the solver
- Performance improvements
- Documentation and examples
- Python binding improvements

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project (MIT and Apache 2.0).
