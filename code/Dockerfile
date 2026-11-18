FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends postgresql ca-certificates \
 && PGV=$(ls /usr/lib/postgresql | sort -n | tail -1) \
 && apt-get install -y --no-install-recommends postgresql-${PGV}-pgvector \
 && ln -s /usr/lib/postgresql/${PGV}/bin /usr/lib/postgresql/current \
 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -m spacy download en_core_web_sm
COPY *.py ./
USER app
ENV HOME=/home/app PATH="/usr/lib/postgresql/current:${PATH}"
CMD ["python", "harness.py"]
