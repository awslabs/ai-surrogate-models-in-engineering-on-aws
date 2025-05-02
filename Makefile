# Usage examples:
#
# To use with virtual, run:
#
#	virtualenv .venv
# 	source .venv/bin/activate
#
# To install the package in editable mode:
#   make install
#
# To run unit tests:
#   make test
#
# To create a wheel distribution:
#   make wheel

# Variables
PYTHON := python3
PIP := pip3
RUFF := ruff
PYTEST := pytest
COVERAGE := coverage

# Project paths
SRC_DIR := src
DOCS_DIR := docs
TESTS_DIR := tst

# Package name (replace with your actual package name)
PACKAGE_NAME := mlsimkit

# Targets
.PHONY: all clean install lint test coverage docs wheel

all: install ruff test coverage docs wheel
ruff: lint format

clean:
	rm -rf  dist build docs/_build *.egg-info .eggs .pytest_cache

install:
	pip install -e .

lint:
	$(RUFF) check . --fix

format:
	$(RUFF) format .

test:
	$(PYTEST) $(TESTS_DIR)

coverage:
	$(COVERAGE) run -m pytest $(TESTS_DIR)
	$(COVERAGE) report
	$(COVERAGE) xml

docs:
	cd $(DOCS_DIR) && $(MAKE) html

pdf:
	cd $(DOCS_DIR) && $(MAKE) latexpdf

sdist:
	$(PYTHON) -m build --sdist
	@SDIST_ARCHIVE=$$(ls -Art dist/$(PACKAGE_NAME)-*.tar.gz | tail -n1); \
	scripts/make_src_zip "$$SDIST_ARCHIVE" dist/

wheel:
	$(PYTHON) -m build --wheel
	@WHEEL_FILE=$$(ls -Art dist/$(PACKAGE_NAME)-*.whl | tail -n1); \
	echo ""; \
	echo "To install the wheel, run:"; \
	echo ""; \
	echo "		pip install $$WHEEL_FILE"; \
	echo ""; \
	echo " or to a custom directory: "; \
	echo ""; \
	echo "		pip install $$WHEEL_FILE" --prefix /opt/mlsimkit; \
	echo ""
