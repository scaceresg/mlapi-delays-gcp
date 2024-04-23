# syntax=docker/dockerfile:1.2
FROM python:3.10-slim

# Set current working directory to /code
WORKDIR /code/app

# Copy the requirements.txt file
COPY ./requirements.txt /code/app/requirements.txt
COPY ./requirements-test.txt /code/app/requirements-test.txt
COPY ./requirements-dev.txt /code/app/requirements-dev.txt

# Install requirements.txt package
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements-test.txt
RUN python3 -m pip install --no-cache-dir -r requirements-dev.txt

# Copy /challenge and /data directories to /code
COPY ./challenge /code/app
COPY ./data /code/data

EXPOSE 8080

# Run uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]