# Use the official Python 3.8 runtime as the base image
FROM python:3.8 

# Set the working directory inside the container
WORKDIR /workspace

# Copy the current local directory contents into the container at /app
COPY . /workspace
RUN apt-get update && apt-get install -y \
    python3-pip python3-full \
    && rm -rf /var/lib/apt/lists/* \

RUN pip install -r requirements.txt

CMD ["/bin/bash"]