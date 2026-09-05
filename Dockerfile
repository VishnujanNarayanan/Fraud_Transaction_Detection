# Pinned to the same minor version CI runs, so a container failure is never a
# Python-version difference the test suite could not have caught.
FROM python:3.10-slim

# Keeps the image small and the logs unbuffered, so `docker logs` shows training
# progress as it happens rather than in one burst at the end.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies are copied and installed before the source, so editing a .py file
# does not invalidate the layer that compiles numpy, pandas and scikit-learn.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip && pip install -r requirements-dev.txt

COPY pyproject.toml ./
COPY src/ ./src/
COPY tests/ ./tests/

# Fraud.csv is a 470 MB download and is never baked into the image. Mount it:
#   docker run --rm -v "$PWD":/data fraud-detection python -m src.train --csv /data/Fraud.csv
VOLUME ["/data"]

# Default to the test suite, which needs no data and proves the image is sound.
CMD ["pytest"]
