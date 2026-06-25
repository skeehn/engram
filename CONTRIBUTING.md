# Contributing to engram

Thank you for your interest in contributing to engram! 🧠

engram is a multi-modal knowledge database for AI agents. Your contributions help improve knowledge persistence for the entire AI agent ecosystem.

## How to Contribute

### 1. Report Bugs

Found a bug? [Open an issue](https://github.com/skeehn/engram/issues) with:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Rust version, engram version)
- Relevant logs or error messages

### 2. Suggest Features

Have an idea? [Open a feature request](https://github.com/skeehn/engram/issues) with:
- Problem statement
- Proposed solution
- Example usage
- Performance implications

### 3. Submit Pull Requests

**Before starting:**
- Check existing issues and PRs
- For large changes, open an issue first

**PR Guidelines:**
- Fork the repo and create a branch from `master`
- Follow Rust conventions (rustfmt, clippy)
- Add tests for new features
- Update documentation
- Ensure `cargo test --all` passes
- Ensure `cargo clippy --all` passes

**Commit message format:**
```
<type>: <description>

Examples:
feat: Add temporal time-travel queries
fix: Handle missing JINA_API_KEY gracefully
docs: Update README with installation steps
perf: Optimize RRF fusion scoring
```

### 4. Improve Documentation

Documentation PRs are highly valued:
- README improvements
- Code comments
- Architecture diagrams
- Usage examples
- API documentation

## Development Setup

```bash
# Clone repo
git clone https://github.com/skeehn/engram.git
cd engram

# Build
cargo build

# Run tests
cargo test --all

# Lint
cargo clippy --all

# Format
cargo fmt --all

# Build release
cargo build --release

# Install locally
cargo install --path engram-cli
```

## Project Structure

```
engram/
├── engram-core/        # Core types and traits
├── engram-store/       # Sled key-value storage
├── engram-fts/         # Tantivy full-text search
├── engram-vector/      # Flat vector index
├── engram-embed/       # Jina v3 embeddings
├── engram-rerank/      # Jina reranker
├── engram-graph/       # Graph operations
├── engram-query/       # RRF fusion
├── engram-extract/     # Jina Reader (URL ingestion)
├── engram-temporal/    # Temporal queries
├── engram-ingest/      # Ingestion pipeline
└── engram-cli/         # CLI binary
```

## Code Style

- **Rust 2021 edition**
- **Follow Rustfmt**: `cargo fmt --all`
- **Pass Clippy**: `cargo clippy --all`
- **Naming**: 
  - Functions: `snake_case`
  - Types: `PascalCase`
  - Constants: `SCREAMING_SNAKE_CASE`

## Testing

- Unit tests: `cargo test --lib`
- Integration tests: `cargo test --test '*'`
- Doc tests: `cargo test --doc`
- All tests: `cargo test --all`

**Test coverage expectations:**
- Core logic: >80%
- CLI: >60%
- Happy path + error cases

## Performance

engram is performance-critical. When adding features:
- Benchmark with `cargo bench` (if applicable)
- Profile with `cargo flamegraph`
- Minimize allocations in hot paths
- Use `Arc` / `Rc` appropriately
- Document time/space complexity

## Documentation

Update docs when you:
- Add a new crate
- Add public API
- Change CLI commands
- Change configuration format

**Documentation requirements:**
- Public API: rustdoc comments
- Modules: Module-level docs
- Examples: Usage examples in docs
- README: User-facing guide

## Environment Variables

engram requires:
- `JINA_API_KEY`: For embeddings and reranking (get at jina.ai)

Optional:
- `ENGRAM_DB`: Custom database path (default: `~/.engram/knowledge`)

## Community

- **Discord**: [grain.ai/discord](https://grain.ai/discord) (shared with grain)
- **Issues**: [GitHub Issues](https://github.com/skeehn/engram/issues)

## Code of Conduct

- Be respectful and constructive
- Help newcomers
- No harassment, discrimination, or spam
- Focus on improving engram for everyone

## Recognition

Contributors are recognized in:
- Release notes
- Project README
- `CONTRIBUTORS.md` (coming soon)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for making engram better!** 🧠

Every contribution makes knowledge persistence better for all AI agents.

Let's build the future of AI memory together.
