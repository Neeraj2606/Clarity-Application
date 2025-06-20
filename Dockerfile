# 1. Specify the Base Image
# Use an official, lightweight Python image for a consistent environment.
FROM python:3.9-slim

# 2. Set the Working Directory
# All subsequent commands will be run from this '/app' directory inside the container.
WORKDIR /app

# 3. Copy and Install Dependencies Efficiently
# Copy only the requirements file first to leverage Docker's layer caching.
# This step is only re-run if requirements.txt changes.
COPY requirements.txt .

# Install the Python dependencies. The --no-cache-dir flag keeps the image size smaller.
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the Application Code
# Copy all the remaining files from the local project directory into the container.
COPY . .

# 5. Expose the Necessary Port
# Inform Docker that the container will listen on port 8501, which is Streamlit's default.
EXPOSE 8501

# 6. Set the Command to Run the Application
# This is the default command that will be executed when the container starts.
# It runs the Streamlit app on the correct port and in a container-friendly headless mode.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"] 