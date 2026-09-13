.PHONY: install install-dev lint typecheck test train clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

lint:
	ruff check .

typecheck:
	mypy .

test:
	pytest tests/

train:
	python model/src/train_baseline.py

clean:
	rm -rf .mypy_cache .ruff_cache