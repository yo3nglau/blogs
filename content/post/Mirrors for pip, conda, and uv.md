---
title: Mirrors for pip, conda, and uv
author: yo3nglau
date: '2025-11-16'
categories:
  - Computer Technology
tags:
  - Guide
  - Mirror
toc: true
---

## Preface

Efficient Python development often relies on fast and stable package downloads. Network latency or regional restrictions can slow down installations from default repositories. Configuring **mirror sources** for `pip` and `conda` is a simple and powerful way to speed up package installations, improve reliability, and standardize environments.

This post provides a concise introduction to why mirrors matter and how to configure, add, remove, and inspect them using practical commands.

## Advantages of Mirrors

Mirror sources are synchronized replicas of official package repositories. Common benefits include:

* **Faster download speeds** when the mirror is geographically closer
* **Higher reliability** in regions with unstable access to default servers
* **Improved accessibility** in constrained networks
* **Better reproducibility** across teams and CI pipelines

## Mirrors for `pip`

### Temporary Use (one-time installation)

```bash
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Persistent Configuration (recommended)

#### Edit the `pip.conf`/`pip.ini` file

Create or update:

* **Linux/macOS:** `~/.pip/pip.conf`
* **Windows:** `%USERPROFILE%\pip\pip.ini`

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

#### Use Commands

```bash
# Add
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# Remove
pip config unset global.index-url
pip config unset global.trusted-host
```

### Show Current Configuration

```bash
pip config list
```

or show a specific key:

```bash
pip config get global.index-url
```

## Mirrors for `conda`

`conda` stores its configuration in the `.condarc` file. You can manipulate mirror channels using simple commands.

### Temporary Use (one-time installation)

You can define the channel during installation:

```bash
conda install numpy -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
```

### Persistent Configuration (recommended)

#### Edit the `.condarc` file (recommended)

Create or update:

* **Linux/macOS:** `~/.condarc`
* **Windows:** `%USERPROFILE%\.condarc`

```yaml
channels:
  - defaults
show_channel_urls: true

default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2

custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

#### Use commands

Example: adding TUNA mirrors.

```bash
# Add
conda config --add default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2

conda config --add custom_channels.conda-forge https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
conda config --add custom_channels.pytorch https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud

# Remove
conda config --remove default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --remove-key custom_channels

# Reset
conda config --remove-key default_channels
conda config --remove-key custom_channels
conda config --append channels defaults
```

### Show current configuration

```bash
conda config --show
conda config --show channels
```

## Mirrors for `uv`

### Temporary Use (one-time installation)

```bash
uv add pandas --index https://pypi.tuna.tsinghua.edu.cn/simple
```

### Persistent Configuration (recommended)

#### Edit the `pyproject.toml` or `uv.toml` File

##### Project-level Configuration

Create or update `pyproject.toml`:

```toml
[[tool.uv.index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
```

##### User- and System-level Configuration

Create or update `uv.toml`:

* **Linux/macOS:** `~/.config/uv/uv.toml` or `/etc/uv/uv.toml`
* **Windows:** `%APPDATA%\uv\uv.toml` or `%SYSTEMDRIVE%\ProgramData\uv\uv.toml`

```toml
[[index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
```

`uv.toml` files take precedence over `pyproject.toml` files.

User- and system-level configuration must use the `uv.toml` format.

**Priority**: UV merges configuration from **project-level** (e.g., `pyproject.toml`) and **user-/system-level** (`uv.toml`). Project-level settings take precedence.

### No Showing Current Configuration

**uv’s design**: Configuration is read from `.toml` files (project or user/system) instead of being managed with CLI config commands.

## Resources

[pip/config](https://pip.pypa.io/en/stable/cli/pip_config/)

[Conda/config](https://docs.conda.io/projects/conda/en/stable/commands/config.html)

[uv/configuration-files](https://docs.astral.sh/uv/concepts/configuration-files/)
