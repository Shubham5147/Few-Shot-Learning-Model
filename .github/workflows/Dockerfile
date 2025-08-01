FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy environment and requirements files
COPY environment.yml .
COPY requirements.txt .

# Create conda environment
RUN conda env create -f environment.yml

# Activate environment and install pip dependencies
RUN /bin/bash -c "source activate fewshot-env && pip install -r requirements.txt"

# Copy the rest of the code
COPY . .

# Use conda environment for all following commands
SHELL ["conda", "run", "-n", "fewshot-env", "/bin/bash", "-c"]

# Default command
CMD ["python", "train.py", "--help"]