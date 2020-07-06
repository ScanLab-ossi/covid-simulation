### Running locally

Use `Python>=3.7`

Make sure to install all dependencies: `pip install -r requirements.txt`

Run simulation locally: `python -m simulation.simulation`

Run streamlit app locally: `streamlit run app.py`

Run tests: `nose2 --with-coverage`

Run tests with nice HTML reports: `nose2 --with-coverage --coverage-report html`

#### Run dockerized

To run docker containers, make sure you have docker and docker-compose installed, running linux containers. 

Run docker containers of simulation and streamlit app: `docker-compose up --build [simulation/app]`

### Google Cloud

1. Login, auth and so on.
2. Set default region: `gcloud config set run/region europe-west4`
3. Build images and deploy on Google Cloud:
```
gcloud builds submit
gcloud run deploy --image gcr.io/temporal-dynamics/[simulation/app] --platform managed
```