FROM python:3.12.7-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# shared data folder
VOLUME ["/app/data"]

COPY . .

ARG DATAWAREHOUSE

CMD ["bash", "start_app.sh"]
