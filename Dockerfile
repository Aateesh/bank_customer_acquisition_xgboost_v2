# ============================================================
# Base Image
# ============================================================
FROM python:3.13-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# ============================================================
# Install Java 17 - required by PySpark
# ============================================================
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       openjdk-17-jre-headless \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Java Configuration
# ============================================================
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ============================================================
# Application Working Directory
# ============================================================
WORKDIR /app

# ============================================================
# Install Python Dependencies
# ============================================================
COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# ============================================================
# Copy Application Source Code
# ============================================================
COPY src ./src

# ============================================================
# Copy Configuration
# ============================================================
COPY config ./config
# ============================================================
# Copy Data
# Only the two runtime inputs that are not in S3:
#   - stage 1 source CSV (SOURCE_DATA_FILE, loaded into Snowflake)
#   - stage 5 evaluation dataset
# Processed features / predictions live in S3, not in the image.
# ============================================================
COPY data/customer_acquisition_scoring_data.csv ./data/
COPY data/customer_acquisition_test_data.csv ./data/

# ============================================================
# Copy Model Artifact
# ============================================================
COPY xgb_customer_acquisition_pipeline.joblib .

# ============================================================
# Copy Pipeline Runner
# ============================================================
COPY run_pipeline.py .

# ============================================================
# Python Runtime Configuration
# ============================================================
ENV PYTHONUNBUFFERED=1

# ============================================================
# Build traceability (set by CI: --build-arg GIT_SHA=<commit>)
# ============================================================
ARG GIT_SHA=unknown
ENV APP_GIT_SHA=${GIT_SHA}
LABEL org.opencontainers.image.revision=${GIT_SHA} \
      org.opencontainers.image.title="bank-customer-acquisition-ml"

# ============================================================
# Start Pipeline
# ============================================================
CMD ["python", "run_pipeline.py"]