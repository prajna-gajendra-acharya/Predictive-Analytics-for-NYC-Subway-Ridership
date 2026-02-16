# Predictive Analytics for NYC Subway Ridership

## Overview
NYC Subway Insights is a big data-driven web application that analyzes and predicts ridership patterns for the top 20 busiest stations of the New York City MTA subway system. Built with PySpark for distributed data processing, a LightGBM model for predictive analytics, and Streamlit for an interactive interface, the app visualizes historical ridership trends, predicts future ridership up to 2030, and maps station locations. The dashboard integrates weather and U.S. holiday data for enhanced predictions and features official MTA subway icons for a thematic design.

Data sources : 
- NYC Subway Risership - https://catalog.data.gov/dataset/mta-subway-hourly-ridership-beginning-february-2022
- Weather data - https://open-meteo.com

---
## Prerequisites
Before running the app, ensure you have the following installed:

- **Python**: Version 3.12 (exact version tested)
- **Java**: Version 17.0.15 (required for PySpark)
- **Apache Hadoop**: Version 3.3.6 recommended (required for PySpark on all operating systems)
- **Web Browser**: A modern browser (e.g., Chrome, Firefox) to view the Streamlit app
- **PySpark**: Version 3.5.5

---

## Setup Instructions

### 1. Clone the Repository
Clone the project from GitHub to your local machine:

```bash
cd <Predictive_Analytics_for_NYC_Subway_Ridership>
```

### 2. Install Dependencies
Create a virtual environment (recommended) to isolate the project dependencies:

```bash
python3.12 -m venv venv
source venv/Scripts/activate  # On Windows
source venv/bin/activate  # On macOS/Linux
```

Install the required Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment
#### Install Java
- Ensure Java 17.0.15 is installed. Download from https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html if needed.
- Set the `JAVA_HOME` environment variable to your Java installation directory (e.g., `C:\Program Files\Java\jdk-17` on Windows, `/usr/lib/jvm/java-17-openjdk` on Linux).

#### Install Hadoop
- Download Hadoop 3.3.6 from https://hadoop.apache.org/releases.html and extract it (e.g., to `C:/hadoop` on Windows, `/usr/local/hadoop` on macOS/Linux).
- Set the `HADOOP_HOME` environment variable to the Hadoop directory.
- Add Hadoop’s `bin` directory to your `PATH`:
  - On Windows: Add `C:\hadoop\bin` to the `Path` environment variable.
  - On macOS/Linux: Add `export PATH=$PATH:/usr/local/hadoop/bin` to your shell configuration (e.g., `~/.bashrc` or `~/.zshrc`).

### 4. Prepare Required Files
Ensure the following files are in the project directory:

- `lightgbm_ridership_model_3.txt`: Pre-trained LightGBM model file.
- `scaler.pkl`: Pre-trained scaler file for numerical features.
- `mta_top20_joined_1/`: Directory containing Parquet files with MTA subway ridership data.

### 5. Running the App

**Start the Streamlit Server**:
   Navigate to the project directory and run:
   ```bash
   streamlit run Dashboard.py
   ```
   This will start the Streamlit server and open the app in your default browser at `http://localhost:8501`.

**Interact with the App**:
   - **Filter by Borough**: Use the sidebar to select a borough (e.g., Manhattan, Brooklyn, Queens, Bronx, or "All").
   - **Predict Ridership**: Select a station, fare class, payment method, date (up to 2030), time, and weather conditions, then click "Predict Ridership" to get a prediction.
   - **Ridership Trends**: Select a station, fare class, payment method, year, and month, then click "Generate Ridership Chart" to view historical trends. Download the chart using the "Download Chart" button.
   - **Station Locations Map**: View an interactive map of the top 20 stations.


