---
title: Fast Python Package and Project Management with uv
author: yo3nglau
date: '2025-11-12'
categories:
  - Computer Technology
tags:
  - Guide
toc: true
---

## Preface

In today’s Python ecosystem, managing dependencies, virtual environments, project configuration and publishing can involve a variety of tools—`pip`, `virtualenv/venv`, `pyenv`, `poetry`, `pip‑tools`, and more. The tool `uv` is designed to unify and simplify that workflow into a single high-performance solution.

## Introduction

`uv` is an extremely fast Python package and project manager, written in `Rust`.
It aims to serve as a drop-in replacement for `pip` and related tooling, offering additional features such as managed virtual environments, Python version handling, universal lock-files, and project scaffolding.

## Install uv

You can install via a standalone installer, via `pip` or `pipx`, via `Homebrew` or `WinGet`, or even via `Cargo`.
Example:

```bash
pipx install uv
# or
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Basic usage

### Creating a New Project

You can initialize a project in the working directory:

```bash
cd working_dir
uv init
```

### Create a Virtual Environment

```bash
uv venv .venv --python 3.12
```

Here `uv` automatically manages and creates the environment.

### Manage Dependencies

You can add dependencies to your `pyproject.toml` with the `uv add` command. This will also update the lockfile and project environment:

```bash
uv add requests
```

You can also specify version constraints or alternative sources:

```bash
# Specify a version constraint
uv add 'requests==2.31.0'

# Add a git dependency
uv add git+https://github.com/psf/requests
```

If you're migrating from a `requirements.txt` file, you can use `uv add` with the `-r` flag to add all dependencies from the file:

```bash
# Add all dependencies from `requirements.txt`.
uv add -r requirements.txt
```

To remove a package, you can use `uv remove`:

```bash
uv remove requests
```

### Install Dependencies (pip-style interface)

```bash
uv pip install requests
```

## Commands

---

### `uv init`

Create a new project.

#### Usage

```bash
uv init [OPTIONS] [PATH]
```

#### Arguments

##### PATH

The path to use for the project/script.

#### Options

##### `--bare`

Only create a `pyproject.toml`.

This will disable creating extra files like `README.md`, the `src/` tree, `.python-version` files, etc.

##### `--python`

Determine the minimum supported Python version for the Python interpreter.

Common supported Python version request formats are:

- `<version>` e.g. `3`, `3.12`, `3.12.3`
- `<version-specifier>` e.g. `>=3.12,<3.13`

---

### `uv add`

Add dependencies to the project.

Dependencies are added to the project's `pyproject.toml` file.

The lockfile and project environment will be updated to reflect the added dependencies. To skip updating the lockfile, use `--frozen`. To skip updating the environment, use `--no-sync`.

#### Usage

```bash
uv add [OPTIONS] <PACKAGES|--requirements <REQUIREMENTS>>
```

#### Arguments

##### PACKAGES

The packages to add, as PEP 508 requirements (e.g., `ruff==0.5.0`)

#### Options

##### `--requirements`, `--requirement`, `-r` *requirements*

Add the packages listed in the given files. The following formats are supported: `requirements.txt`, `.py` files with inline metadata, `pylock.toml`, `pyproject.toml`, `setup.py`, and `setup.cfg`.

##### `--frozen`

Add dependencies without re-locking the project.

##### `--no-sync`

Avoid syncing the virtual environment.

## Conclusion

If you’re working on Python projects and find yourself using multiple tools for environments, dependencies, version management and publishing, `uv` offers a compelling alternative. Its speed, unified interface and reproducible workflow make it worthy of evaluation. Start with a small project, get familiar with `uv venv`, `uv pip`, `uv lock`, `uv run`, and see whether it can streamline your development experience.

## Resources

[Astral/uv](https://docs.astral.sh/uv/)
