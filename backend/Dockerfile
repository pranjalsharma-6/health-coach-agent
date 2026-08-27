# Kaya API
#
# Slim rather than alpine: the wheels for bcrypt and pydantic-core are prebuilt
# for glibc, and alpine's musl forces a source build that is both slow and
# fragile.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Run as a non-root user.
RUN useradd --create-home --uid 1000 kaya && chown -R kaya:kaya /app
USER kaya

EXPOSE 8000

# Hosts inject $PORT; default to 8000 for a plain `docker run`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
