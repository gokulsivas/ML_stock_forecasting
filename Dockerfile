FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    postgresql \
    postgresql-contrib \
    libpq-dev \
    gcc \
    git \
    curl \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ── Working directory ──────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ── Copy project files ─────────────────────────────────
COPY . .

# ── PostgreSQL setup ───────────────────────────────────
USER postgres
RUN /etc/init.d/postgresql start && \
    psql -c "CREATE USER stockuser WITH PASSWORD 'stockpass';" && \
    psql -c "CREATE DATABASE stockdb OWNER stockuser;" && \
    psql -c "GRANT ALL PRIVILEGES ON DATABASE stockdb TO stockuser;"
USER root

# ── Ports ──────────────────────────────────────────────
EXPOSE 8001 5432

# ── Entrypoint ────────────────────────────────────────
CMD ["python", "orchestrator.py"]