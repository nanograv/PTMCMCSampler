.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help
define BROWSER_PYSCRIPT
import os, webbrowser, sys
try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

init:
	@python3 -m venv venv
	@./venv/bin/python3 -m pip install -U pip setuptools wheel
	@./venv/bin/python3 -m pip install -r requirements.txt -U
	@./venv/bin/python3 -m pip install -r requirements_dev.txt -U
	@./venv/bin/python3 -m pre_commit install --install-hooks --overwrite
	@./venv/bin/python3 -m pip install -e .

format:
	pre-commit run --all-files

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

test: format ## run tests quickly with the default Python
	pytest -v --durations=10 --full-trace --cov-report html --cov-config .coveragerc --cov=PTMCMCSampler tests

coverage: test ## check code coverage quickly with the default Python
	$(BROWSER) htmlcov/index.html


dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

test-sdist: ## Test source distribution
	mkdir tmp
	cd tmp
	python -m venv venv-sdist
	venv-sdist/bin/python -m pip install --upgrade pip setuptools wheel
	venv-sdist/bin/python -m pip install dist/ptmcmcsampler*.tar.gz
	venv-sdist/bin/python -c "import PTMCMCSampler;print(PTMCMCSampler.__version__)"
	rm -rf tmp venv-sdist

test-wheel: ## Test wheel
	mkdir tmp2
	cd tmp2
	python -m venv venv-wheel
	venv-wheel/bin/python -m pip install --upgrade pip setuptools
	venv-wheel/bin/python -m pip install dist/ptmcmcsampler*.whl
	venv-wheel/bin/python -c "import PTMCMCSampler;print(PTMCMCSampler.__version__)"
	rm -rf tmp2 venv-wheel
