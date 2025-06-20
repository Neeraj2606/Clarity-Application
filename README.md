# Clarity ✨ - Interactive Data Quality Assistant

Clarity is a web-based, interactive data quality auditing tool designed to help users quickly identify and resolve common data issues. Upload a CSV file, and Clarity provides a comprehensive suite of tools to profile, clean, and validate your data, including an intelligent fuzzy duplicate detection system.

![Clarity Application Demo](https://user-images.githubusercontent.com/12345/placeholder.png)  <!-- Replace with a real screenshot of your app -->

---

## 🚀 Key Features

*   **Interactive Data Profiling:** Get an instant overview of your dataset, including row/column counts, data types, and missing value summaries.
*   **Descriptive Statistics:** Automatically calculate and display descriptive statistics for all numerical columns.
*   **Value Distribution Analysis:** Interactively explore the value counts for categorical columns.
*   **Advanced Quality Reporting:** Generate a detailed, column-by-column report covering null values, outlier detection, range consistency, and more.
*   **Interactive Cleaning Workbench:**
    *   **Handle Missing Values:** Easily drop rows with missing data or fill them using common strategies (mean, median, mode) or a custom value.
    *   **Resolve Fuzzy Duplicates:** An intelligent algorithm identifies potential duplicates that aren't exact matches (e.g., "John Smith" vs. "J. Smith"). Users can then interactively merge these duplicate clusters.
*   **Download Clean Data:** Once you're done cleaning, download the final, polished dataset as a new CSV file.
*   **Dockerized for Portability:** The entire application is containerized with a `Dockerfile`, allowing it to be run on any machine with a single command.

## 🛠️ Tech Stack

*   **Backend:** Python
*   **Web Framework:** [Streamlit](https://streamlit.io/)
*   **Data Manipulation:** [Pandas](https://pandas.pydata.org/)
*   **Fuzzy String Matching:** [thefuzz](https://github.com/seatgeek/thefuzz)
*   **Containerization:** [Docker](https://www.docker.com/)

---

## 🏁 Getting Started

There are two ways to run the Clarity application: using Docker (recommended for ease of use) or setting up a local Python environment.

### 1. Running with Docker (Recommended)

This is the simplest way to run Clarity. It ensures all dependencies are handled correctly within a self-contained environment.

**Prerequisites:**
*   [Docker](https://www.docker.com/get-started) installed on your machine.

**Steps:**
1.  **Clone the repository:**
    ```sh
    git clone https://github.com/Neeraj2606/Clarity-Application.git
    cd Clarity-Application
    ```

2.  **Build the Docker image:**
    This command reads the `Dockerfile` and builds the container image, tagging it as `clarity-app`.
    ```sh
    docker build -t clarity-app .
    ```

3.  **Run the Docker container:**
    This command starts the container and maps port `8501` from the container to port `8501` on your local machine.
    ```sh
    docker run -p 8501:8501 clarity-app
    ```

4.  **Access the application:**
    Open your web browser and navigate to **http://localhost:8501**.

### 2. Running Locally

If you prefer to run the application in a local Python environment.

**Prerequisites:**
*   Python 3.9+
*   `pip` (Python package installer)

**Steps:**
1.  **Clone the repository:**
    ```sh
    git clone https://github.com/Neeraj2606/Clarity-Application.git
    cd Clarity-Application
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```sh
    streamlit run app.py
    ```

5.  **Access the application:**
    Streamlit will provide a local URL in your terminal, which you can open in your web browser (usually http://localhost:8501).

---

## 📖 How to Use Clarity

1.  **Upload Data:** Start by uploading your dataset using the file uploader on the main page. Only CSV files are supported.
2.  **Profile Data:** Review the automated data profile to get a quick understanding of your data's characteristics.
3.  **Clean Data:**
    *   Use the **Cleaning Workbench** in the sidebar to handle missing values in any column.
    *   In the main panel, use the **Detect Potential Duplicates** section to find and resolve fuzzy duplicates. Select the records you wish to keep and merge them.
4.  **Download:** Once all cleaning operations are complete, use the **Download Clean Data** button to save your final dataset. 