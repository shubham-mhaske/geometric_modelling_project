.PHONY: help install test train run clean docs

help:
	@echo "High-Fidelity Mesh Improvement Pipeline"
	@echo "========================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make install     Install dependencies"
	@echo "  make download    Download sample dataset"
	@echo "  make train       Train ML model"
	@echo "  make test        Run test suite"
	@echo "  make run         Launch Streamlit app"
	@echo "  make clean       Clean generated files"
	@echo "  make structure   Show project structure"
	@echo ""

install:
	pip install -r requirements.txt
	pip install -e .

download:
	python scripts/download_data.py

train:
	python scripts/train_ml_model.py --samples 200 --epochs 50

test:
	python tests/test_pipeline.py

run:
	streamlit run app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/
	@echo "Cleaned up temporary files"

structure:
	tree -I '__pycache__|*.pyc|.git|*.egg-info' -L 3

format:
	black src/ tests/ scripts/ app.py
	@echo "Code formatted with black"

lint:
	flake8 src/ tests/ scripts/ app.py
	@echo "Linting complete"

package:
	python setup.py sdist bdist_wheel
	@echo "Package built in dist/"

all: install download train test
	@echo "Full setup complete!"
