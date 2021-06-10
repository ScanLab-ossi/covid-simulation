### Running locally

Use `Python==3.7`

Make sure to install all dependencies: `pip install -r requirements.txt`

<!-- `cd` into `covid-simulation` and install this as a package before working on it. It makes imports easier: `pip install -e .` -->

Run simulation locally: `python -m simulation.simulation`

Run streamlit app locally: `streamlit run app.py`

#### Run dockerized

To run docker containers, make sure you have docker and docker-compose installed, running linux containers. 

Run docker containers of simulation and streamlit app: `docker-compose up --build [simulation/app]`

### Run on Google Compute Engine

`ssh` into the instance (currently named `instance-2`). 

Run using `python3 -m simulation.simulation`. Don't forget to `git pull` before running!

Run in `screen -S screen-name` so process won't die. Rename is possible with `Ctrl+A , :` followed by `sessionname name`.

### Running Tests

Make sure to install packages needed for testing: `pip install black nose2[coverage_plugin] mypy`

Run tests: `nose2 --with-coverage`

Run tests with nice HTML reports: `nose2 --with-coverage --coverage-report html`

Run type-checking: `mypy -p simulation --ignore-missing-imports`

### Google Cloud

1. Login, auth and so on.
2. Set default region: `gcloud config set run/region europe-west4`
3. Build images and deploy on Google Cloud:

```
docker push eu.gcr.io/temporal-dynamics/simulation:latest
gcloud run deploy simulation --image eu.gcr.io/temporal-dynamics/simulation:latest --region europe-west4 --platform managed
```

```
docker push eu.gcr.io/temporal-dynamics/app:latest
gcloud app deploy --image-url=eu.gcr.io/temporal-dynamics/app:latest
```

### Colors and their meaning

| color  | shorthand | meaning                                             |
| ------ | --------- | --------------------------------------------------- |
| green  | g         | susceptible                                         |
| blue   | b         | asymptomatic, or light symptoms that aren't noticed |
| purple | p         | pre-symptomatic                                     |
| pink   | v         | light symptoms, enough to be noticed                |
| red    | r         | heavy, potentially life threatening sickness        |
| white  | w         | recovered                                           |
| black  | k         | deceased                                            |
