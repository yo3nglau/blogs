# Design: Common Shortcuts for the Linux Command Line

**Date:** 2026-03-31  
**Status:** Approved

## Overview

A blog post covering common Linux command line keyboard shortcuts, targeting both beginners and intermediate users. Comprehensive coverage across cursor movement, text editing, command history, process control, and terminal control. Each section includes a quick-reference table and a brief practical explanation. A full summary table is provided at the end.

## Frontmatter

```yaml
title: Common Shortcuts for the Linux Command Line
author: yo3nglau
date: '2026-03-31'
categories:
  - Computer Technology
tags:
  - Guide
  - Linux Command
toc: true
```

## Article Structure

### Preface
Explain that Bash uses the readline library by default, and shortcuts follow the Emacs keybinding style. Mention that Vi mode is available as an alternative (`set -o vi`), but this post focuses on the default Emacs mode.

### Cursor Movement
Quick-reference table covering:
- `Ctrl+A` — Move to beginning of line
- `Ctrl+E` — Move to end of line
- `Alt+F` — Move forward one word
- `Alt+B` — Move backward one word
- `Ctrl+F` — Move forward one character
- `Ctrl+B` — Move backward one character

Typical use case: repositioning the cursor to fix a typo in a long command.

### Text Editing
Quick-reference table covering:
- `Ctrl+H` / `Backspace` — Delete character before cursor
- `Alt+D` — Delete word after cursor
- `Ctrl+W` — Delete word before cursor
- `Ctrl+K` — Delete from cursor to end of line
- `Ctrl+U` — Delete from cursor to beginning of line
- `Ctrl+Y` — Paste (yank) last deleted text
- `Ctrl+_` — Undo last edit
- `Alt+U` — Uppercase word after cursor
- `Alt+L` — Lowercase word after cursor

Typical use case: quickly clearing a partially typed command or fixing a word.

### Command History
Quick-reference table covering:
- `Ctrl+P` / `↑` — Previous command
- `Ctrl+N` / `↓` — Next command
- `Ctrl+R` — Reverse incremental search through history
- `Ctrl+G` — Abort history search
- `!!` — Repeat last command
- `!$` — Last argument of previous command
- `Alt+.` — Insert last argument of previous command

Typical use case: re-running a previous command or reusing arguments.

### Process & Job Control
Quick-reference table covering:
- `Ctrl+C` — Interrupt (kill) current process
- `Ctrl+Z` — Suspend current process (send to background)
- `Ctrl+D` — Send EOF / exit shell (only when the line is empty; deletes character under cursor otherwise)
- `fg` — Resume suspended job in foreground
- `bg` — Resume suspended job in background
- `jobs` — List current jobs

Typical use case: suspending a process to run another command, then resuming.

### Terminal Control
Quick-reference table covering:
- `Ctrl+L` — Clear the terminal screen
- `Ctrl+S` — Pause terminal output (XOFF)
- `Ctrl+Q` — Resume terminal output (XON)
- `Ctrl+Alt+T` — Open new terminal (desktop environment shortcut, not readline)

Brief note: `Ctrl+S` / `Ctrl+Q` are flow control shortcuts and may feel like the terminal is frozen.

### Quick Reference
A consolidated table of all shortcuts from all sections, organized as: Shortcut | Action | Category.

### Conclusion
Encourage readers to practice the shortcuts gradually rather than memorizing all at once. Suggest starting with cursor movement and history search (`Ctrl+R`) as the highest-value shortcuts.

### Resources
- GNU Readline documentation: https://www.gnu.org/software/bash/manual/bash.html#Command-Line-Editing
- Bash manual (Bindable Readline Commands): https://www.gnu.org/software/bash/manual/bash.html#Bindable-Readline-Commands

## Constraints

- Language: English
- Style: consistent with existing blog posts (concise, practical, no filler)
- Format: Markdown, Hugo-compatible, with YAML frontmatter
- Output file: `content/post/Common Shortcuts for the Linux Command Line.md`
- Site rebuild required after writing: `hugo` command
