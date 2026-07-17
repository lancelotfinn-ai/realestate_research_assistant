FROM python:3.11-slim

# Install R and jsonlite (prebuilt Debian packages — no slow compilation)
RUN apt-get update && \
    apt-get install -y --no-install-recommends r-base r-cran-jsonlite r-cran-sf r-cran-httr r-cran-dplyr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
