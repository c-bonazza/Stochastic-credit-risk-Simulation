FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml data_generation.py simulator.py visualize.py main.py ./

CMD ["python", "main.py"]
