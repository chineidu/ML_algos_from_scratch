.PHONY: setup_venv run_lint run_style run_checks

setup_venv:
	python3 -m venv venv && . venv/bin/activate \
	&& pip install --upgrade pip \
	&& python3 -m pip install -e ".[dev]"

run_lint:
	. venv/bin/activate \
	&& black src && isort src

run_style:
	. venv/bin/activate \
	&& pylint --recursive=y src

run_checks: run_lint run_style