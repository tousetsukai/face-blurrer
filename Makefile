yolov11m-face.pt:
	@echo "Downloading face detection model..."
	@wget -O $@ https://github.com/akanametov/yolo-face/releases/download/v0.0.0/$@

.PHONY: format
format:
	uv run ruff format

.PHONY: lint
lint:
	uv run ruff check

.PHONY: lint-fix
lint-fix:
	uv run ruff check --fix
