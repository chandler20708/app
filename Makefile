.PHONY: help venv install run clean

VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

help:
	@echo "Targets:"
	@echo "  make venv    - Create virtual environment"
	@echo "  make install - Install dependencies (and Graphviz)"
	@echo "  make run     - Run Streamlit app"
	@echo "  make clean   - Remove virtual environment"

venv:
	python3 -m venv $(VENV_DIR)

install: venv
	@if command -v dot >/dev/null 2>&1; then \
		echo "Graphviz already installed."; \
	elif [ "$$(uname -s)" = "Linux" ]; then \
		if command -v apt >/dev/null 2>&1; then \
			sudo apt install graphviz; \
		elif command -v dnf >/dev/null 2>&1; then \
			sudo dnf install graphviz; \
		else \
			echo "No supported Linux package manager found. Install Graphviz from https://gitlab.com/graphviz/graphviz/-/releases"; \
			exit 1; \
		fi; \
	elif [ "$$(uname -s)" = "Darwin" ]; then \
		if command -v brew >/dev/null 2>&1; then \
			brew install graphviz; \
		elif command -v port >/dev/null 2>&1; then \
			sudo port install graphviz; \
		else \
			/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
			if command -v brew >/dev/null 2>&1; then \
				brew install graphviz; \
			else \
				echo "Homebrew install failed. Install Homebrew manually, then run: brew install graphviz"; \
				exit 1; \
			fi; \
		fi; \
	elif [ "$$OS" = "Windows_NT" ] || echo "$$(uname -s)" | grep -qiE "mingw|msys|cygwin"; then \
		if command -v choco >/dev/null 2>&1; then \
			choco install graphviz -y; \
		elif command -v winget >/dev/null 2>&1; then \
			winget install graphviz --accept-package-agreements --accept-source-agreements; \
		else \
			echo "No Chocolatey or Winget found. Install Graphviz from https://graphviz.org/download/"; \
			exit 1; \
		fi; \
	else \
		echo "Unsupported OS for automatic Graphviz install. See https://graphviz.org/download/"; \
		exit 1; \
	fi; \
	echo "If 'dot' is not found, restart your terminal to refresh PATH."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: install
	$(PYTHON) -m streamlit run Overview.py

clean:
	rm -rf $(VENV_DIR)
