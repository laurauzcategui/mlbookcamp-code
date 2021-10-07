FROM agrigorev/zoomcamp-model:3.8.12-slim

# Install pipenv
RUN pip install pipenv

# Workdir 
ARG appdir=/app

WORKDIR $appdir

# Copy the files
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies with pipenv
RUN pipenv install --system --deploy

# Copy the files
COPY ["homework_q4.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "homework_q4:app"]