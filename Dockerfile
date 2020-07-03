# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

ENV APP_HOME /covid-simulation
WORKDIR $APP_HOME

# need git for machine_version
RUN apt-get update && apt-get install -y \
    git 
# Install production dependencies.
RUN pip install \
    google-cloud-storage \
    google-cloud-datastore \
    pandas \ 
    jsonschema \
    altair \
    altair_saver \
    pyyaml \
    flask \
    gunicorn

# Copy local code to the container image.

COPY . ./

# helps code understand its running on cloud, so starts server
ENV LOCAL False
# only for running locally. google cloud sets port on it's own
ENV PORT 8080

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# CMD python -m simulation.simulation

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 simulation.simulation:app
