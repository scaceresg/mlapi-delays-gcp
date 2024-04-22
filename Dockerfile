# syntax=docker/dockerfile:1.2
FROM python:3.10-slim

# Set current working directory to /code
WORKDIR /code

# Copy the requirements.txt file
COPY ./requirements.txt /code/requirements.txt

# Install requirements.txt package
RUN pip install -r /code/requirements.txt
RUN pip install -r /code/requirements-test.txt
RUN pip install -r /code/requirements-dev.txt
RUN pip cache purge

# Copy /challenge and /data directories to /code
COPY ./challenge /code/app
COPY ./data /code/data

# Run uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]