FROM python:3.10.6

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
     mariadb-client \
  && rm -rf /var/lib/apt/lists/*

# Get necessary python libraries
COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt

# Copy over code
COPY ./code/hw6/hw6.sql .
# COPY models models

## Run app
#CMD gunicorn --bind :$PORT --timeout 500 --workers 1 --threads 2 flask_examples.simple_pickled_model:app
