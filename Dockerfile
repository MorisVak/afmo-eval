FROM python:3.11-slim

WORKDIR /app
COPY . /app

# Install locked Linux deps to avoid ABI issues
RUN pip install --no-cache-dir --upgrade pip setuptools wheel &&     pip install --no-cache-dir -r requirements-locked/linux-py311.txt &&     pip install --no-cache-dir -e .[ui]

EXPOSE 8501
CMD ["streamlit", "run", "app/AFMo.py", "--server.address=0.0.0.0"]
