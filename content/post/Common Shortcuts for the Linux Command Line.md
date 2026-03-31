---
title: Common Shortcuts for the Linux Command Line
author: yo3nglau
date: '2026-03-31'
categories:
  - Computer Technology
tags:
  - Guide
  - Linux Command
toc: true
---

## Preface

Bash uses the GNU Readline library to handle command-line input. By default, it follows an Emacs-style keybinding scheme, which makes a rich set of shortcuts available for editing commands, navigating history, and controlling the terminal—without touching the mouse.

This post covers the most useful shortcuts across five areas: cursor movement, text editing, command history, process and job control, and terminal control. A consolidated quick-reference table is provided at the end.

> **Note:** These shortcuts apply to Bash in its default Emacs mode. If you have switched to Vi mode (`set -o vi`), the bindings differ.

## Cursor Movement

| Shortcut | Action |
|----------|--------|
| `Ctrl+A` | Move to beginning of line |
| `Ctrl+E` | Move to end of line |
| `Alt+F`  | Move forward one word |
| `Alt+B`  | Move backward one word |
| `Ctrl+F` | Move forward one character |
| `Ctrl+B` | Move backward one character |

These shortcuts shine when editing long commands. For example, after pressing `↑` to recall a previous command, use `Ctrl+A` to jump to the beginning, then `Alt+F` to hop word-by-word to the typo you need to fix—far faster than holding the arrow key.

## Text Editing

| Shortcut | Action |
|----------|--------|
| `Backspace` / `Ctrl+H` | Delete character before cursor |
| `Alt+D`  | Delete word after cursor |
| `Ctrl+W` | Delete word before cursor |
| `Ctrl+K` | Delete from cursor to end of line |
| `Ctrl+U` | Delete from cursor to beginning of line |
| `Ctrl+Y` | Paste (yank) last deleted text |
| `Ctrl+_` | Undo last edit |
| `Alt+U`  | Uppercase word after cursor |
| `Alt+L`  | Lowercase word after cursor |

`Ctrl+K` and `Ctrl+U` are especially useful together: they act as a cut operation. Delete part of a command with `Ctrl+K`, correct what remains, then restore the deleted portion with `Ctrl+Y`.

## Command History

| Shortcut | Action |
|----------|--------|
| `↑` / `Ctrl+P` | Previous command in history |
| `↓` / `Ctrl+N` | Next command in history |
| `Ctrl+R` | Reverse incremental history search |
| `Ctrl+G` | Abort history search |
| `!!`     | Repeat last command |
| `!$`     | Last argument of the previous command |
| `Alt+.`  | Insert last argument of the previous command |

Most of the above are Readline keybindings; `!!` and `!$` are Bash history expansions typed literally at the prompt.

`Ctrl+R` is arguably the most powerful shortcut in this list. Press it, start typing any part of a previous command, and Bash finds the most recent match instantly. Press `Ctrl+R` again to cycle to older matches.

`!$` and `Alt+.` save you from retyping long file paths. After running `vim /etc/nginx/nginx.conf`, you can run `cat !$` to print the same file.

## Process and Job Control

| Shortcut / Command | Action |
|--------------------|--------|
| `Ctrl+C` | Interrupt (terminate) the current process |
| `Ctrl+Z` | Suspend the current process |
| `Ctrl+D` | Send EOF / exit the shell (only when the line is empty) |
| `fg`     | Resume the most recent suspended job in the foreground |
| `bg`     | Resume the most recent suspended job in the background |
| `jobs`   | List all current jobs with their status |

A common workflow: you are running a command interactively and need to quickly check something else. Press `Ctrl+Z` to suspend it, do your work, then run `fg` to resume exactly where you left off.

## Terminal Control

| Shortcut | Action |
|----------|--------|
| `Ctrl+L` | Clear the terminal screen (equivalent to `clear`) |
| `Ctrl+S` | Pause terminal output (XOFF) |
| `Ctrl+Q` | Resume terminal output (XON) |

If your terminal appears frozen and stops responding to input, `Ctrl+S` is often the culprit—it pauses all output. Press `Ctrl+Q` to unfreeze it.

## Quick Reference

| Shortcut / Command | Action | Category |
|----------|--------|----------|
| `Ctrl+A` | Move to beginning of line | Cursor |
| `Ctrl+E` | Move to end of line | Cursor |
| `Alt+F`  | Move forward one word | Cursor |
| `Alt+B`  | Move backward one word | Cursor |
| `Ctrl+F` | Move forward one character | Cursor |
| `Ctrl+B` | Move backward one character | Cursor |
| `Backspace` / `Ctrl+H` | Delete character before cursor | Editing |
| `Alt+D`  | Delete word after cursor | Editing |
| `Ctrl+W` | Delete word before cursor | Editing |
| `Ctrl+K` | Delete from cursor to end of line | Editing |
| `Ctrl+U` | Delete from cursor to beginning of line | Editing |
| `Ctrl+Y` | Paste (yank) last deleted text | Editing |
| `Ctrl+_` | Undo last edit | Editing |
| `Alt+U`  | Uppercase word after cursor | Editing |
| `Alt+L`  | Lowercase word after cursor | Editing |
| `↑` / `Ctrl+P` | Previous command | History |
| `↓` / `Ctrl+N` | Next command | History |
| `Ctrl+R` | Reverse incremental history search | History |
| `Ctrl+G` | Abort history search | History |
| `!!`     | Repeat last command | History |
| `!$`     | Last argument of previous command | History |
| `Alt+.`  | Insert last argument of previous command | History |
| `Ctrl+C` | Interrupt current process | Process/Job |
| `Ctrl+Z` | Suspend current process | Process/Job |
| `Ctrl+D` | Send EOF / exit shell (empty line only) | Process/Job |
| `fg`     | Resume suspended job in foreground | Process/Job |
| `bg`     | Resume suspended job in background | Process/Job |
| `jobs`   | List current jobs | Process/Job |
| `Ctrl+L` | Clear terminal screen | Terminal |
| `Ctrl+S` | Pause terminal output | Terminal |
| `Ctrl+Q` | Resume terminal output | Terminal |

## Conclusion

You don't need to memorize all of these at once. Start with the highest-return shortcuts: `Ctrl+R` for history search, `Ctrl+A`/`Ctrl+E` for line navigation, and `Ctrl+K`/`Ctrl+Y` for cut-and-paste editing. The rest will become natural over time as you encounter the situations they solve.

## Resources

- [GNU Bash Manual — Command-Line Editing](https://www.gnu.org/software/bash/manual/bash.html#Command-Line-Editing)
- [GNU Bash Manual — Bindable Readline Commands](https://www.gnu.org/software/bash/manual/bash.html#Bindable-Readline-Commands)
