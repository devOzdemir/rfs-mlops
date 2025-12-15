# KARARLI VE MODERN SÜRÜM: 2.10.3
FROM apache/airflow:2.10.3

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         build-essential \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt

# Constraint linkini 2.10.3 sürümüne uygun hale getirdik
RUN pip install --no-cache-dir \
    -r /requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.3/constraints-3.10.txt"