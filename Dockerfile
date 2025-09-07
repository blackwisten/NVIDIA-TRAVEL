FROM python:3.12-slim

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-chache-dir -r requirements.txt && \
	apt-get update && apt-get install -y --no-install-recommends wget && \
	rm -rf /var/lib/apt/lists/*

COPY . .

CMD["sh","-c","python app.py"]

