#!/bin/bash
# Initialize the Airflow database
poetry run airflow db init

# Create an Admin user
poetry run airflow users create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin

# Start the Airflow webserver and scheduler
poetry run airflow webserver -D -p 8999
poetry run airflow scheduler
