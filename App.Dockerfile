# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7

ENV APP_HOME /covid-simulation
WORKDIR $APP_HOME

# need libpq-dev and python3-dev for psycopg2
RUN apt-get update && apt-get install -y \
    libpq-dev \ 
    python3-dev 

# Install production dependencies.
RUN python -m pip install -U pip
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy local code to the container image.
COPY . .

# helps code understand its running on cloud, so starts server
ENV LOCAL False
# only for running locally. google cloud sets port on it's own

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# CMD python -m simulation.simulation a

CMD streamlit run --server.port $PORT --server.enableCORS false app.py
