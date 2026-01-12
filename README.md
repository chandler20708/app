# AED Scheduling and Analytics App

Streamlit application for operator scheduling and AED analytics.

## Prerequisites
- Python 3.10+
- Graphviz (for tree visualisations)

## Quick Start
```sh
make install
make run
```

The app will start at `http://localhost:8501`.

## Graphviz
`make install` attempts to install Graphviz automatically:
- Linux: `apt` or `dnf`
- macOS: Homebrew or MacPorts
- Windows: Chocolatey or Winget

If Graphviz is already installed, it is skipped. If the installer does not update PATH, restart your terminal.

## Make Targets
```sh
make venv
make install
make run
make clean
```
