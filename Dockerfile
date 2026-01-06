FROM python:3.11-slim

# Needed for LightGBM (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 10000
CMD ["streamlit", "run", "app/AFMO.py", "--server.port=10000", "--server.address=0.0.0.0"]
