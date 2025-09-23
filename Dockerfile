FROM python:3.12.7-slim

ENV PYTHONUNBUFFERED=1

# RUN apt-get update && apt-get install -y curl git wget unzip

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# in addition to requirements.txt install package pandas-ta from repo
# COPY pandas-ta.tar.gz .
# RUN tar -xzf pandas-ta.tar.gz
# RUN pip install ./pandas-ta

# shared data folder
VOLUME ["/app/data"]

COPY . .

ARG DATAWAREHOUSE

CMD ["bash", "start_app.sh"]
