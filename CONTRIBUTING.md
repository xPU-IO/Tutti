# Contributing Guide

This document defines the collaboration rules for human contributors and AI contributors working in this repository.

Use this file together with:

- [`README.md`](README.md)
- [`Roadmap.md`](Roadmap.md)

## First Principles

This repository is being refactored from a historical `Tutti` implementation toward a `Unified Storage Runtime`.

Contributors must optimize for:

- clear layering
- stable and replaceable interfaces
- explicit ownership and lifecycle
- deployment realism
- reviewability

Do not optimize for short-term convenience at the cost of locking future `Local NVMe`, `GDS`, `RDMA`, or vendor-specific backends into the wrong abstraction.

## Required Reading Before Changes

Before making meaningful changes, read:

1. [`Roadmap.md`](Roadmap.md)
2. [`README.md`](README.md)
3. Relevant subsystem documents for the area you are changing

If you are changing architecture, interfaces, deployment flow, or naming, you must read the relevant current documents first and align with them.

## Coding Rules

### General

- Keep code ASCII unless the file already intentionally uses another character set
- Avoid hardcoded local absolute paths
- Avoid personal environment assumptions in source files, examples, and docs
- Prefer explicit ownership and explicit error handling over hidden side effects
- Keep comments focused on invariants, lifecycle, concurrency, hardware assumptions, and tricky logic
- Do not add comments that merely restate obvious syntax

### Layering

- Do not mix `device manager`, `IO engine`, `memory`, and `backend` concerns in one new interface
- Do not expose backend-private implementation types through future public runtime APIs
- Do not bind core abstractions to the current file-based implementation model
- Do not introduce new top-level abstractions that permanently hardcode the `Tutti` name unless maintainers explicitly choose that path

### Include Paths

Two include roots are registered, and the spelling tells you which side of the
API boundary a header lives on:

| Header | Root | Form | Example |
| --- | --- | --- | --- |
| Public API / SPI (`csrc/include/tutti/`) | `csrc/include` | angle brackets | `#include <tutti/spi/data_path.h>` |
| Implementation (everything else under `csrc/`) | repository root | double quotes | `#include "csrc/data_paths/..."` |

- These two are the only allowed spellings. Do not write bare implementation
  paths (`"data_paths/..."`), `csrc/include/...`, or relative `"../foo.h"`
  paths; the only sanctioned exception is the kernel-shared UAPI shim
  `libnvm/include/ioctl.h`, which must resolve from its own directory for both
  the kernel module and userspace builds.
- First-line path comments: public headers state their **include name**
  (`// tutti/spi/data_path.h`); implementation files state their **source path**
  (`// csrc/data_paths/...`). Keep them in sync when a file moves.
- Implementation-only helpers must not live under `csrc/include/tutti/`:
  `install(DIRECTORY csrc/include/tutti/)` ships that whole tree, so a helper
  placed there silently becomes installed API surface. Cross-cutting helpers
  belong in `csrc/common/`.
- Public headers must not include implementation headers, and must not name
  backend-private types or constants.
- New backend packages follow
  `csrc/payloads/<name>/payload.h` (payload contract) +
  `csrc/resolvers/<name>/resolver.h` + `csrc/data_paths/<name>/`, and register
  their identity constants in `csrc/common/backend_ids.h`. See
  [doc/extending_tutti.md](doc/extending_tutti.md).

### Refactoring Strategy

This repository is mid-refactor. All code movement must follow a **zero-risk, additive-first** discipline:

- **Create new, keep old**: When extracting or reorganizing code, always create the new file or header in the target location first. Do not delete or modify the existing file as part of the same step.
- **Legacy files are backups**: the historical monolithic GPU-file implementation has been moved out of the repository as an external backup; it is no longer part of the build.
- **No forced cutover**: The new layer structure and the legacy path may coexist in the same build until the maintainer explicitly decides the new path is complete and verified.
- **Deletion only at the end**: Legacy file removal is deferred until the full refactoring is complete and has been validated. Deletion must be a separate, explicit commit after that decision is made by the maintainer.
- **One step at a time**: Each refactor step should produce at most one new file or one new directory boundary. Do not batch multiple extractions into one commit.

### Hardware and System Constraints

- Treat deployment, boot ordering, permissions, module readiness, and Linux compatibility as design inputs, not post-hoc operations work
- Kernel-facing changes must consider Linux version drift
- Public runtime semantics must stay separate from backend driver capabilities
- Do not expose `cooperative submit` as a runtime contract in `v0.1`

### Configuration and Examples

- Prefer repository-relative config examples
- Example programs must not require a contributor's personal filesystem layout
- Changes that affect configuration shape must be reflected in the relevant docs

## Interface Rules

### Public Interface Discipline

Public or near-public interfaces must be:

- stable enough to test
- backend-agnostic where intended to be shared
- explicit about ownership
- explicit about sync vs async behavior
- explicit about thread safety
- explicit about failure semantics

At minimum, interface changes should define:

- input/output model
- lifecycle
- error behavior
- concurrency model
- capability assumptions
- fallback or unsupported behavior

### What Must Not Leak Into Public APIs

Do not expose the following directly in future runtime-facing APIs unless a maintainer has approved it as an explicit contract:

- raw backend controller internals
- kernel ioctl details
- queue implementation details
- PRP layout details
- CUDA IPC internals
- filesystem-specific persistence internals

### When an RFC or Design Discussion Is Required

Open a design discussion before landing major changes to:

- public API shape
- directory structure
- backend SPI
- device-manager/io-engine boundary
- memory model
- config format
- wire protocol
- deployment model
- project/runtime naming

If the change alters long-term architecture, do not hide it inside a normal implementation patch.

## Documentation Rules

If you change any of the following, update docs in the same change or explain why not:

- architecture boundaries
- public interfaces
- config semantics
- deployment flow
- kernel module expectations
- known bugs
- roadmap priorities

Versioned planning must continue to preserve:

- `Feature Snapshot`
- `Known Bugs Snapshot`

## Testing and Validation Rules

When applicable, include or describe validation for:

- functional correctness
- concurrency behavior
- deployment impact
- config compatibility
- device-manager/io-engine interaction

If tests are not added, state the reason in the commit or review notes.

## AI Contributor Rules

AI contributors must:

- read the active roadmap and task list before large edits
- avoid inventing new version numbers
- avoid silently renaming the project/product
- avoid hardcoding local machine paths
- keep changes scoped and reviewable
- update docs when changing interfaces, naming, or architecture intent
- avoid mixing unrelated refactors into one patch

AI contributors must not assume that the current repository layout is the final architecture.

## Commit Rules

### Commit Scope

Each commit should do one coherent thing:

- one feature
- one fix
- one refactor step
- one documentation change
- one test addition

Do not mix unrelated cleanup, renames, and behavior changes into a single commit.

### Commit Message Format

Use this format:

```text
<type>(<scope>): <summary>
```

Rules:

- `type` is required
- `scope` is required
- `summary` must be short, specific, and in imperative mood
- keep the first line concise
- do not end the summary with a period
- avoid vague messages such as `update`, `fix bug`, `misc`, `cleanup`

### Allowed Commit Types

- `feat`
- `fix`
- `refactor`
- `docs`
- `test`
- `build`
- `ci`
- `perf`
- `chore`
- `revert`

### Recommended Scopes

- `api`
- `runtime`
- `memory`
- `device_manager`
- `io_engine`
- `backend/local_nvme`
- `kernel`
- `nvmeservice`
- `adapter/lmcache`
- `adapter/mooncake`
- `docs`
- `build`
- `tests`
- `examples`
- `repo`

### Good Commit Examples

```text
docs(repo): define contribution and commit rules
refactor(memory): separate registration from allocation semantics
fix(nvmeservice): validate queue lease release on expired client
feat(api): add runtime capability query skeleton
test(io_engine): cover cpu submit request validation
build(kernel): document module install prerequisites
```

### Bad Commit Examples

```text
update
fix bug
misc cleanup
refactor: many changes
docs: update files
```

## Review Expectations

Every change should be reviewable against:

- layer boundaries
- interface clarity
- deployment impact
- naming quality
- testability
- future backend extensibility

If a change makes one of these worse, the contributor should justify the tradeoff explicitly.
