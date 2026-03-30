---
title: Long-Running Process Management with screen Command
author: yo3nglau
date: '2025-11-10'
categories:
  - Computer Technology
tags:
  - Guide
  - Linux Command
  - Remote Server
toc: true
---

## Preface

When managing long-running processes in a Linux environment, maintaining persistent terminal sessions becomes essential—especially when working over SSH connections. Accidental disconnections can interrupt tasks, terminate important jobs, and disrupt your workflow. This is where the `screen` command proves invaluable. It allows users to create detachable terminal sessions that continue running even after the connection is closed.

## Introduction

`screen` is a terminal multiplexer commonly used in Linux systems. It enables multiple shell sessions to run within a single terminal window. More importantly, it provides the ability to detach from a session without stopping the underlying processes, and later reattach to it seamlessly.

## Basic Operations

### Install the Screen Command

```bash
sudo apt install screen  # for Debian/Ubuntu
```

### Start a New Naming Screen Session for Organization

```bash
screen -S mytask
```

Custom names make it easier to identify and manage tasks later.

### List Existing Sessions

```bash
screen -ls
```

### Detach from a Session

```bash
screen -d <session_id>
```

Or press the shortcut key:

```
Ctrl + A then D
```

The session continues running in the background.

### Reattach to a Session

```bash
screen -r <session_id>
```

This restores the terminal view of a running session.

### Terminating a Screen Session

When finished, exit the shell within the session:

```bash
exit
```

This closes the associated screen window.

## Conclusion

The `screen` command is a lightweight yet powerful tool for maintaining persistence on remote servers and managing multiple shell sessions efficiently.

If you frequently work with SSH or run processes that must not be interrupted, integrating `screen` into your daily toolbox is highly recommended.

## Resources

[geeksforgeeks/screen-command-in-linux-with-examples](https://www.geeksforgeeks.org/linux-unix/screen-command-in-linux-with-examples/)
