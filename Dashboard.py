import os

java_home = "/usr/lib/jvm/java-11-openjdk-amd64"
if os.path.exists(java_home):
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = java_home + "/bin:" + os.environ.get("PATH", "")

import subprocess
import streamlit as st

# Debug: find where Java actually got installed
result = subprocess.run(["find", "/usr", "-name", "java", "-type", "f"], 
                       capture_output=True, text=True)
st.code(result.stdout or "Java not found!")
st.stop()  # Stop here so we can see the output

import pyspark as sp
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, time
from pyspark.sql.functions import col, year, to_timestamp, sum as _sum, desc, date_format, countDistinct, collect_set, first
from pyspark.sql import Row
from pyspark.sql import functions as F
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
from calendar import month_name
import numpy as np
import lightgbm as lgb
import joblib
import holidays
import re
from math import ceil

st.set_page_config(page_title="NYC Subway Ridership", page_icon="🚇")

# Load the pre-trained LightGBM model and scaler for ridership prediction
# The model file and scaler are expected to be in the same directory as the script
model = lgb.Booster(model_file="./lightgbm_ridership_model_3.txt")
scaler = joblib.load("./scaler.pkl")

# Define the categorical variables expected by the LightGBM model
# These lists are based on the training data used to build the model
station_complexes = [
    "Jamaica Center-Parsons/Archer (E,J,Z)",
    "Fordham Rd (4)",
    "34 St-Penn Station (1,2,3)",
    "Myrtle-Wyckoff Avs (L,M)",
    "Flushing-Main St (7)",
    "Junction Blvd (7)",
    "161 St-Yankee Stadium (B,D,4)",
    "74-Broadway (7)/Jackson Hts-Roosevelt Av (E,F,M,R)",
    "Bedford Av (L)",
    "Times Sq-42 St (N,Q,R,W,S,1,2,3,7)/42 St (A,C,E)",
    "Kings Hwy (B,Q)",
    "3 Av-149 St (2,5)",
    "34 St-Herald Sq (B,D,F,M,N,Q,R,W)",
    "Grand Central-42 St (S,4,5,6,7)",
    "Hunts Point Av (6)",
    "103 St-Corona Plaza (7)",
    "34 St-Penn Station (A,C,E)",
    "Parkchester (6)",
    "Atlantic Av-Barclays Ctr (B,D,N,Q,R,2,3,4,5)",
    "Crown Hts-Utica Av (3,4)",
]

fare_class_categories = [
    "Metrocard - Fair Fare",
    "OMNY - Fair Fare",
    "OMNY - Seniors & Disability",
    "OMNY - Full Fare",
    "Metrocard - Unlimited 7-Day",
    "Metrocard - Unlimited 30-Day",
    "Metrocard - Full Fare",
    "Metrocard - Other",
    "Metrocard - Seniors & Disability",
    "Metrocard - Students",
    "OMNY - Other",
    "OMNY - Students",
]

payment_methods = [
    "metrocard", "omny",
]

# Define the numerical columns expected by the scaler
# These are the feature names used during training, retrieved from the scaler
numerical_cols = scaler.feature_names_in_.tolist()

def predict_ridership(station, dt, temp, humidity, precipitation, fare_class, payment_method):
    """
    Predict ridership for a given station, timestamp, and conditions using the LightGBM model.

    Args:
        station (str): Station complex name
        dt (datetime or str): Timestamp (datetime object or string in 'YYYY-MM-DD HH:MM:SS' format)
        temp (float): Temperature in Celsius
        humidity (float): Humidity percentage
        precipitation (float): Precipitation in mm
        fare_class (str): Fare class category
        payment_method (str): Payment method

    Returns:
        int: Ceiling of predicted ridership
    """
    # Convert dt to datetime if it's a string, ensuring proper format
    if isinstance(dt, str):
        try:
            dt = pd.to_datetime(dt)
        except ValueError:
            raise ValueError("Invalid timestamp format. Use 'YYYY-MM-DD HH:MM:SS'.")
    elif not isinstance(dt, datetime):
        raise ValueError("dt must be a datetime object or a string in 'YYYY-MM-DD HH:MM:SS' format.")

    # Validate categorical inputs against the model's expected values
    if station not in station_complexes:
        raise ValueError(f"Station '{station}' not in training data: {station_complexes}")
    if fare_class not in fare_class_categories:
        raise ValueError(f"Fare class '{fare_class}' not in training data: {fare_class_categories}")
    if payment_method not in payment_methods:
        raise ValueError(f"Payment method '{payment_method}' not in training data: {payment_methods}")

    # Prepare numerical features for scaling, setting lag features to 0 since they’re unavailable during prediction
    numerical_data = {
        'temperature_C': temp,
        'humidity_%': humidity,
        'precipitation_mm': precipitation,
        'hour': dt.hour,
        'day_of_week': dt.weekday(),
        'month': dt.month,
        'ridership_lag1': 0,  # No lag data for prediction
        'ridership_lag24': 0,  # No lag data for prediction
    }

    # Create a DataFrame for numerical features in the order expected by the scaler
    numerical_df = pd.DataFrame([numerical_data], columns=numerical_cols)

    # Scale the numerical features using the pre-trained scaler
    numerical_scaled = scaler.transform(numerical_df)

    # Map scaled values to their respective indices
    scaled_indices = {col: idx for idx, col in enumerate(numerical_cols)}
    temp_scaled = numerical_scaled[0][scaled_indices['temperature_C']]
    humidity_scaled = numerical_scaled[0][scaled_indices['humidity_%']]
    precip_scaled = numerical_scaled[0][scaled_indices['precipitation_mm']]

    # Prepare the input dictionary for the LightGBM model
    # Include temporal features, binary indicators, and scaled numerical features
    input_dict = {
        'hour': dt.hour,
        'day_of_week': dt.weekday(),
        'month': dt.month,
        'is_weekend': 1 if dt.weekday() in [5, 6] else 0,
        'is_holiday': 1 if dt.date() in holidays.US() else 0,
        'ridership_lag1': 0,
        'ridership_lag24': 0,
        'heavy_rain': 1 if precipitation > 10 else 0,
        'high_temp': 1 if temp > 30 else 0,
        'temperature_C': temp_scaled,
        'humidity_%': humidity_scaled,
        'precipitation_mm': precip_scaled,
    }

    # Add one-hot encoded categorical features for station, fare class, and payment method
    model_features = set(model.feature_name())
    station_encoded = f"station_complex_{re.sub(r'[ /(),-]', '_', station)}"
    fare_class_encoded = f'fare_class_category_{fare_class.replace(" ", "_").replace("/", "_").replace("-", "_")}'
    payment_method_encoded = f'payment_method_{payment_method.replace(" ", "_").replace("/", "_").replace("-", "_")}'

    for col in model_features:
        if col.startswith('station_complex_'):
            input_dict[col] = 1 if col == station_encoded else 0
        elif col.startswith('fare_class_category_'):
            input_dict[col] = 1 if col == fare_class_encoded else 0
        elif col.startswith('payment_method_'):
            input_dict[col] = 1 if col == payment_method_encoded else 0
        elif col not in input_dict:
            input_dict[col] = 0

    # Create input DataFrame with features in the order expected by the LightGBM model
    input_df = pd.DataFrame([input_dict])[model.feature_name()]

    # Predict using the LightGBM model, undo the log1p transformation, and return the ceiling of the prediction
    return ceil(np.expm1(model.predict(input_df)[0]))

# Load and preprocess the MTA data using PySpark
@st.cache_data
def load_data():
    # Initialize a Spark session for distributed data processing
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, to_date, year, month, hour

    spark = SparkSession.builder \
    .appName("Load Top 20 Stations") \
    .master("local[1]") \
    .config("spark.ui.enabled", "false") \
    .config("spark.driver.host", "localhost") \
    .config("spark.driver.memory", "1g") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

    # Load the Parquet files containing MTA data
    top_20_data = spark.read.parquet("./mta_top20_joined_1")

    # Parse the transit_timestamp column into a proper timestamp format
    top_20_data = top_20_data.withColumn("parsed_timestamp", to_timestamp("transit_timestamp", "MM/dd/yyyy hh:mm:ss a"))

    # Aggregate daily ridership by station, borough, fare class, payment method, year, and date
    daily_ridership = top_20_data \
        .groupBy(
            "station_complex",
            "borough",
            "fare_class_category",
            "payment_method",
            year(col("parsed_timestamp")).alias("year"),
            to_date(col("parsed_timestamp")).alias("date")
        ) \
        .agg(_sum("ridership").alias("daily_ridership"))

    # Aggregate total ridership by station and borough, including latitude and longitude for mapping
    top_20_stations = top_20_data \
        .groupBy("station_complex", "borough", "latitude", "longitude") \
        .agg(_sum("ridership").alias("total_ridership")) \
        .select("station_complex", "borough", "latitude", "longitude", "total_ridership")

    # Convert PySpark DataFrames to Pandas DataFrames for use in Streamlit
    top_20_stations_df = top_20_stations.toPandas()
    daily_ridership_df = daily_ridership.toPandas()

    # Convert the "date" column to a datetime type for easier filtering and plotting
    daily_ridership_df["date"] = pd.to_datetime(daily_ridership_df["date"])

    return top_20_stations_df, daily_ridership_df

# Load the data into Pandas DataFrames
top_20_stations_df, daily_ridership_df = load_data()

# Extract unique stations and boroughs for filtering
stations = top_20_stations_df["station_complex"].unique()
boroughs = top_20_stations_df["borough"].unique()

# Sidebar: Borough Filter
# Allow users to filter stations by borough
st.sidebar.header("Filter by Borough")
borough = st.sidebar.selectbox("Select Borough", ["All"] + list(boroughs))

# Filter stations based on the selected borough
if borough == "All":
    stations_in_borough = stations
else:
    stations_in_borough = top_20_stations_df[top_20_stations_df["borough"] == borough]["station_complex"].unique()

st.markdown(
    """
    <div style="text-align: center; margin-bottom: 10px;">
        <span style="background-color: #EE352E; color: white; border-radius: 50%; padding: 8px 16px; margin: 0 5px; font-weight: bold; font-size: 30px;">2</span>
        <span style="background-color: #00AF3F; color: white; border-radius: 50%; padding: 8px 16px; margin: 0 5px; font-weight: bold; font-size: 30px;">4</span>
        <span style="background-color: #00AF3F; color: white; border-radius: 50%; padding: 8px 16px; margin: 0 5px; font-weight: bold; font-size: 30px;">5</span>
        <span style="background-color: #FCCC0A; color: black; border-radius: 50%; padding: 8px 16px; margin: 0 5px; font-weight: bold; font-size: 30px;">N</span>
        <span style="background-color: #FCCC0A; color: black; border-radius: 50%; padding: 8px 16px; margin: 0 5px; font-weight: bold; font-size: 30px;">R</span>
        <span style="background-color: #FCCC0A; color: black; border-radius: 50%; padding: 8px 16px; margin: 0 5px; font-weight: bold; font-size: 30px;">Q</span>
        <span style="background-color: #0039A6; color: white; border-radius: 50%; padding: 8px 16px; margin: 0 5px; font-weight: bold; font-size: 30px;">A</span>
        <span style="background-color: #0039A6; color: white; border-radius: 50%; padding: 8px 16px; margin: 0 5px; font-weight: bold; font-size: 30px;">C</span>
        <span style="background-color: #FF6319; color: white; border-radius: 50%; padding: 8px 16px; margin: 0 5px; font-weight: bold; font-size: 30px;">F</span>
        <span style="background-color: #FF6319; color: white; border-radius: 50%; padding: 8px 16px; margin: 0 5px; font-weight: bold; font-size: 30px;">D</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Main Title and Description
st.title("Predictive Analytics for NYC Subway Ridership")

# Section 1: Prediction with LightGBM
st.header("Predict Ridership")
with st.expander("Prediction Inputs", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        # Allow users to select a station, fare class, and payment method
        station_pred = st.selectbox("Select Station", stations_in_borough, key="pred_station")
        fare_class_pred = st.selectbox("Select Fare Class Category", fare_class_categories, key="pred_fare")
        payment_method_pred = st.selectbox("Select Payment Method", payment_methods, key="pred_payment")
    
    with col2:
        # Allow users to select a date, time, and weather conditions
        # Date range extends to 2030 for future predictions
        date_pred = st.date_input("Select Date", value=datetime(2025, 5, 6), min_value=datetime(2023, 1, 1), max_value=datetime(2030, 12, 31))
        time_pred = st.time_input("Select Time", value=time(8, 0))
        temp_f = st.number_input("Temperature (°F)", min_value=-20.0, max_value=120.0, value=70.0)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
        precipitation = st.number_input("Precipitation (inches)", min_value=0.0, max_value=10.0, value=0.0)

    # Predict ridership using the LightGBM model when the button is clicked
    if st.button("Predict Ridership"):
        try:
            # Convert temperature from Fahrenheit to Celsius for the model
            temp_c = (temp_f - 32) * 5 / 9

            # Convert precipitation from inches to mm (1 inch = 25.4 mm) for the model
            precipitation_mm = precipitation * 25.4

            # Combine date and time into a single timestamp string in the format expected by predict_ridership
            dt = datetime.combine(date_pred, time_pred)
            dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")

            # Call the predict_ridership function with user inputs
            predicted_ridership = predict_ridership(
                station=station_pred,
                dt=dt_str,
                temp=temp_c,
                humidity=humidity,
                precipitation=precipitation_mm,
                fare_class=fare_class_pred,
                payment_method=payment_method_pred
            )
            st.success(f"Predicted Ridership: {predicted_ridership} passengers")
        except ValueError as e:
            st.error(str(e))

# Define the function to plot historical ridership trends as a bar chart
def plot_station_ridership(pandas_df, fare_category, payment_method, year, month=None, station=None, output_dir="station_charts"):
    """
    Generate a bar chart for ridership at a specific station complex for a specific fare category,
    payment method, year, and optional month. Returns the matplotlib figure for Streamlit to display.

    Parameters:
    - pandas_df: Pandas DataFrame with the specified schema
    - fare_category: String, e.g., 'Full Fare'
    - payment_method: String, e.g., 'OMNY'
    - year: Integer, e.g., 2023
    - month: Integer, 1-12 to filter a specific month (optional, defaults to None for all months)
    - station: String, station_complex to filter a specific station (required)
    - output_dir: Directory to save the chart
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Filter the data based on user selections
    filtered_df = pandas_df[
        (pandas_df["station_complex"] == station) &
        (pandas_df["year"] == year)
    ]
    
    # Apply fare class filter if not "All"
    if fare_category != "All":
        filtered_df = filtered_df[filtered_df["fare_class_category"] == fare_category]
    
    # Apply payment method filter if not "All"
    if payment_method != "All":
        filtered_df = filtered_df[filtered_df["payment_method"] == payment_method]

    # Apply month filter if specified
    if month is not None:
        if not 1 <= month <= 12:
            st.error("Month must be between 1 and 12.")
            return None
        filtered_df = filtered_df[filtered_df["date"].dt.month == month]

    # Check if there’s data after filtering
    record_count = len(filtered_df)
    if record_count == 0:
        st.warning("No data available after filtering. Please check the filter parameters against available values logged above.")
        return None

    # Aggregate daily ridership by date for the bar chart
    agg_df = filtered_df.groupby(
        filtered_df["date"].dt.strftime("%Y-%m-%d").rename("date")
    )["daily_ridership"].sum().reset_index(name="total_ridership").sort_values("date")

    # Check if there’s data to plot after aggregation
    if agg_df.empty:
        st.warning(f"No data for station: {station}")
        return None

    # Create the bar chart with a monochromatic design that blends with the Streamlit background
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='none')
    fig.patch.set_alpha(0)  
    ax.set_facecolor('none') 
    ax.bar(agg_df["date"], agg_df["total_ridership"], color='#FFFFFF', label="Ridership")
    # Set title and labels with monochromatic gray tones for a cohesive look
    title_month = f"{month_name[month]} " if month else ""
    ax.set_title(f"Daily Ridership at {station}\n{fare_category}, {payment_method}, {title_month}{year}", color='#FFFFFF')
    ax.set_xlabel("Date", color='#FFFFFF')
    ax.set_ylabel("Total Ridership", color='#FFFFFF')

    # Adjust x-axis ticks based on the number of days for better readability
    num_days = len(agg_df)
    if num_days > 0:
        step = max(1, num_days // 5)
        ax.set_xticks(agg_df["date"][::step])
        ax.tick_params(axis='x', rotation=45, colors='#FFFFFF')

    # Set y-axis scale based on the maximum ridership value
    max_ridership = agg_df["total_ridership"].max()
    if max_ridership <= 0:
        st.warning(f"Max ridership for {station} is {max_ridership}. Using default y-axis scaling.")
        step_size = 1000
    else:
        # Calculate a step size as roughly 1/5 of the max value, rounded to a clean number
        step_size = max(1000, np.round(max_ridership / 5, -3))
    y_ticks = np.arange(0, max_ridership + step_size, step_size)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='y', colors='#FFFFFF')

    # Style the legend with monochromatic gray to match the chart
    ax.legend(facecolor='none', edgecolor='none', labelcolor='#FFFFFF')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the chart as a PNG file with a transparent background
    safe_station_name = station.replace("/", "_").replace(" ", "_")
    filename_month = f"_{month_name[month].lower()}" if month else ""
    output_path = os.path.join(output_dir, f"{safe_station_name}_{year}{filename_month}_barchart.png")
    plt.savefig(output_path, dpi=300, transparent=True)  # Save with transparent background

    return fig

# Section 2: Plotting Historical Ridership Trends
st.header("Ridership Trends")
with st.expander("Plotting Options", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        # Allow users to select a station and fare class for plotting
        station_plot = st.selectbox("Select Station", stations_in_borough, key="plot_station")
        fare_class_options = ["All"] + list(daily_ridership_df["fare_class_category"].unique())
        fare_class_plot = st.selectbox("Select Fare Class Category", fare_class_options, key="plot_fare")
    
    with col2:
        # Allow users to select a payment method, year, and month for plotting
        payment_method_options = ["All"] + list(daily_ridership_df["payment_method"].unique())
        payment_method_plot = st.selectbox("Select Payment Method", payment_method_options, key="plot_payment")
        available_years = sorted(daily_ridership_df["year"].unique())
        year_plot = st.selectbox("Select Year", available_years, key="plot_year")
        month_plot = st.selectbox("Select Month", ["All"] + list(month_name)[1:], key="plot_month")

    # Convert the selected month to a month number (1-12) for filtering
    month_num = None if month_plot == "All" else list(month_name).index(month_plot)

    # Adjust fare class and payment method for "All" selections
    fare_class_input = fare_class_plot if fare_class_plot != "All" else daily_ridership_df["fare_class_category"].unique()[0]
    payment_method_input = payment_method_plot if payment_method_plot != "All" else daily_ridership_df["payment_method"].unique()[0]

    # Generate the bar chart when the button is clicked
    if st.button("Generate Ridership Chart"):
        output_dir = "station_charts"
        fig = plot_station_ridership(
            pandas_df=daily_ridership_df,
            fare_category=fare_class_input,
            payment_method=payment_method_input,
            year=year_plot,
            month=month_num,
            station=station_plot,
            output_dir=output_dir
        )

        # Display the chart and provide a download option if generated
        if fig is not None:
            st.pyplot(fig)
            safe_station_name = station_plot.replace("/", "_").replace(" ", "_")
            filename_month = f"_{month_plot.lower()}" if month_plot != "All" else ""
            chart_filename = f"{safe_station_name}_{year_plot}{filename_month}_barchart.png"
            chart_path = os.path.join(output_dir, chart_filename)
            with open(chart_path, "rb") as file:
                st.download_button(
                    label="Download Chart",
                    data=file,
                    file_name=chart_filename,
                    mime="image/png"
                )

# Section 3: Interactive Map of Station Locations
st.header("Station Locations Map")
# Create a Folium map centered on NYC (latitude 40.7128, longitude -74.0060)
m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
marker_cluster = MarkerCluster().add_to(m)
# Add markers for each station with its total ridership in a popup
for _, row in top_20_stations_df.iterrows():
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=f"{row['station_complex']} (Ridership: {row['total_ridership']})",
        icon=folium.Icon(color="blue")
    ).add_to(marker_cluster)
# Save the map as HTML and display it in Streamlit
m.save("stations_map.html")
st.components.v1.html(open("stations_map.html", "r").read(), height=600)
