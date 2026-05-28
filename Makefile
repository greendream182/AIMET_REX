# AIMET RX — common fixed-point targets (spec 13–14).
ROOT := $(shell pwd)
export PYTHONPATH := $(ROOT)

.PHONY: help test-fixed-point test-fixed-point-fast coverage baseline compare-modes report-fast report-full

help:
	@echo "Targets:"
	@echo "  test-fixed-point-fast  pytest (excl. slow, imagenet)"
	@echo "  test-fixed-point       full pytest (excl. imagenet)"
	@echo "  coverage               pytest + coverage report (>=85%)"
	@echo "  baseline               fast quality report + baseline check"
	@echo "  compare-modes          dual_linear compare_modes JSON"
	@echo "  report-fast / report-full  quality reports"

test-fixed-point-fast:
	python3 -m pytest tests/fixed_point -m "not slow and not imagenet" -q

test-fixed-point:
	python3 -m pytest tests/fixed_point -m "not imagenet" -q

coverage:
	python3 -m coverage run -m pytest tests/fixed_point -m "not slow and not imagenet" -q
	python3 -m coverage report

baseline:
	python3 scripts/fixed_point/run_baseline.py

compare-modes:
	python3 scripts/fixed_point/compare_quant_modes.py \
		--model dual_linear \
		--report scripts/fixed_point/reports/quant_mode_report.json \
		--per-layer-csv scripts/fixed_point/reports/per_layer.csv

report-fast:
	bash scripts/fixed_point/run_report_fast.sh

report-full:
	bash scripts/fixed_point/run_report.sh
