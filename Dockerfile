FROM public.ecr.aws/docker/library/python:3.11

# Set environmental variables to prevent Python from writing .pyc files and to buffer stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file into the container at /app/
COPY requirements.txt /app/

# Install dependencies: upgrade pip, setuptools, and wheel, then install the packages listed in requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Copy the entire contents of the current directory into the container at /app/
COPY . /app/

# Copy the pre-trained model file into the container at /app/model/
COPY model/iris_model.joblib /app/model/iris_model.joblib

# Expose port 8002 to allow communication to/from the container
EXPOSE 8002

# Define the command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]