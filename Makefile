ifeq ($(IN_DIR),)
IN_DIR_PARAM=
else
IN_DIR_PARAM=--in_dir $(IN_DIR)
endif

ifeq ($(OUT_DIR),)
OUT_DIR_PARAM=
else
OUT_DIR_PARAM=--out_dir $(OUT_DIR)
endif

ifeq ($(WORK_DIR),)
WORK_DIR_PARAM=
else
WORK_DIR_PARAM=--work_dir $(WORK_DIR)
endif

PARAMS=$(IN_DIR_PARAM) $(OUT_DIR_PARAM) $(WORK_DIR_PARAM)

.PHONY: run
run: yolov11m-face.pt sync
	uv run main.py $(PARAMS)

.PHONY: detect
detect: yolov11m-face.pt sync
	uv run main.py detect $(PARAMS)

.PHONY: review
review: sync
	uv run main.py review $(PARAMS)

.PHONY: render
render: sync
	uv run main.py render $(PARAMS)

yolov11m-face.pt:
	@echo "Downloading face detection model..."
	@wget -O $@ https://github.com/akanametov/yolo-face/releases/download/v0.0.0/$@

.PHONY: sync
sync:
	uv sync

.PHONY: format
format:
	uv run ruff format

.PHONY: lint
lint:
	uv run ruff check

.PHONY: lint-fix
lint-fix:
	uv run ruff check --fix
