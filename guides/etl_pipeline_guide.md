# Build an ETL Pipeline - Step-by-Step Guide

**Time:** 4-6 hours
**Difficulty:** Intermediate
**Prerequisites:** Python basics, PostgreSQL installed, basic SQL knowledge

## What You'll Build

An automated pipeline that:
1. **E**xtracts weather data from OpenWeather API
2. **T**ransforms it with Pandas (clean, aggregate, enrich)
3. **L**oads it into PostgreSQL database
4. Runs on a schedule automatically

---

## Step 1: Setup and Prerequisites

### Create Project Structure

```bash
mkdir weather-etl-pipeline
cd weather-etl-pipeline

# Create folder structure
mkdir data
mkdir logs
mkdir src

# Create files
touch src/extract.py
touch src/transform.py
touch src/load.py
touch src/pipeline.py
touch config.py
touch requirements.txt
touch .env
touch .gitignore
```

### Setup `.gitignore`

```
# .gitignore
.env
data/
logs/
__pycache__/
*.pyc
.DS_Store
```

### Install Dependencies

```bash
# requirements.txt content:
requests==2.31.0
pandas==2.1.0
python-dotenv==1.0.0
psycopg2-binary==2.9.7
sqlalchemy==2.0.20
schedule==1.2.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## Step 2: Get API Access

### Sign Up for OpenWeather API

1. Go to [openweathermap.org](https://openweathermap.org/api)
2. Click "Sign Up" (it's free!)
3. Verify your email
4. Go to "API keys" tab
5. Copy your API key

### Store Credentials Securely

Create `.env` file:
```bash
# .env
OPENWEATHER_API_KEY=your_api_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=weather_db
DB_USER=postgres
DB_PASSWORD=your_postgres_password
```

---

## Step 3: Extract - Get Data from API

### Create `config.py`

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_KEY = os.getenv('OPENWEATHER_API_KEY')
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# Database Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

# Cities to track
CITIES = [
    'New York',
    'London',
    'Tokyo',
    'Paris',
    'Sydney',
    'Mumbai',
    'Cairo',
    'São Paulo'
]
```

### Create `src/extract.py`

```python
# src/extract.py
import requests
import json
from datetime import datetime
import config

def fetch_weather_data(city):
    """
    Fetch current weather data for a city from OpenWeather API

    Args:
        city (str): City name

    Returns:
        dict: Weather data or None if error
    """
    params = {
        'q': city,
        'appid': config.API_KEY,
        'units': 'metric'  # Celsius
    }

    try:
        response = requests.get(config.BASE_URL, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        print(f"✓ Successfully fetched data for {city}")
        return data

    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching data for {city}: {e}")
        return None

def extract_all_cities():
    """
    Extract weather data for all configured cities

    Returns:
        list: List of weather data dictionaries
    """
    print(f"\n{'='*50}")
    print(f"EXTRACT: Fetching weather data for {len(config.CITIES)} cities")
    print(f"{'='*50}\n")

    all_data = []

    for city in config.CITIES:
        data = fetch_weather_data(city)
        if data:
            all_data.append(data)

    print(f"\n✓ Successfully extracted data for {len(all_data)}/{len(config.CITIES)} cities\n")

    # Save raw data (optional, for debugging)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'data/raw_weather_{timestamp}.json', 'w') as f:
        json.dump(all_data, f, indent=2)

    return all_data

if __name__ == "__main__":
    # Test extraction
    data = extract_all_cities()
    print(f"Extracted {len(data)} records")
```

**Test it:**
```bash
python src/extract.py
```

---

## Step 4: Transform - Clean and Process Data

### Create `src/transform.py`

```python
# src/transform.py
import pandas as pd
from datetime import datetime

def transform_weather_data(raw_data):
    """
    Transform raw API data into clean, structured format

    Args:
        raw_data (list): List of raw weather dictionaries from API

    Returns:
        pd.DataFrame: Cleaned and transformed data
    """
    print(f"\n{'='*50}")
    print(f"TRANSFORM: Processing {len(raw_data)} records")
    print(f"{'='*50}\n")

    # Extract relevant fields
    transformed = []

    for record in raw_data:
        try:
            row = {
                'city_name': record['name'],
                'country': record['sys']['country'],
                'latitude': record['coord']['lat'],
                'longitude': record['coord']['lon'],
                'temperature': record['main']['temp'],
                'feels_like': record['main']['feels_like'],
                'temp_min': record['main']['temp_min'],
                'temp_max': record['main']['temp_max'],
                'pressure': record['main']['pressure'],
                'humidity': record['main']['humidity'],
                'weather_main': record['weather'][0]['main'],
                'weather_description': record['weather'][0]['description'],
                'wind_speed': record['wind']['speed'],
                'cloudiness': record['clouds']['all'],
                'timestamp': datetime.fromtimestamp(record['dt']),
                'extracted_at': datetime.now()
            }
            transformed.append(row)

        except KeyError as e:
            print(f"✗ Error transforming record for {record.get('name', 'Unknown')}: Missing key {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(transformed)

    # Data Quality Checks
    print("\n📊 Data Quality Report:")
    print(f"   Total records: {len(df)}")
    print(f"   Missing values:\n{df.isnull().sum()}")
    print(f"\n   Temperature range: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")
    print(f"   Humidity range: {df['humidity'].min()}% to {df['humidity'].max()}%")

    # Add derived columns
    df['temperature_f'] = df['temperature'] * 9/5 + 32  # Convert to Fahrenheit
    df['is_hot'] = df['temperature'] > 30  # Flag hot weather
    df['is_humid'] = df['humidity'] > 70   # Flag humid conditions
    df['comfort_index'] = df['feels_like'] - df['temperature']  # Feels like difference

    print(f"\n✓ Transformation complete: {len(df)} records processed\n")

    return df

if __name__ == "__main__":
    # Test transformation
    from extract import extract_all_cities

    raw_data = extract_all_cities()
    df = transform_weather_data(raw_data)

    print("\nSample data:")
    print(df.head())

    # Save to CSV for inspection
    df.to_csv('data/transformed_weather.csv', index=False)
    print("\n✓ Saved to data/transformed_weather.csv")
```

**Test it:**
```bash
python src/transform.py
```

---

## Step 5: Load - Insert into PostgreSQL

### Create Database and Table

Connect to PostgreSQL:
```bash
psql -U postgres
```

Run these SQL commands:
```sql
-- Create database
CREATE DATABASE weather_db;

-- Connect to it
\c weather_db

-- Create table
CREATE TABLE weather_data (
    id SERIAL PRIMARY KEY,
    city_name VARCHAR(100),
    country VARCHAR(10),
    latitude DECIMAL(10, 6),
    longitude DECIMAL(10, 6),
    temperature DECIMAL(5, 2),
    feels_like DECIMAL(5, 2),
    temp_min DECIMAL(5, 2),
    temp_max DECIMAL(5, 2),
    pressure INTEGER,
    humidity INTEGER,
    weather_main VARCHAR(50),
    weather_description VARCHAR(200),
    wind_speed DECIMAL(5, 2),
    cloudiness INTEGER,
    temperature_f DECIMAL(5, 2),
    is_hot BOOLEAN,
    is_humid BOOLEAN,
    comfort_index DECIMAL(5, 2),
    timestamp TIMESTAMP,
    extracted_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster queries
CREATE INDEX idx_city_timestamp ON weather_data(city_name, timestamp);

-- Verify
\d weather_data
```

### Create `src/load.py`

```python
# src/load.py
import pandas as pd
from sqlalchemy import create_engine
import config

def create_db_engine():
    """Create SQLAlchemy engine for PostgreSQL connection"""
    conn_string = (
        f"postgresql://{config.DB_CONFIG['user']}:{config.DB_CONFIG['password']}"
        f"@{config.DB_CONFIG['host']}:{config.DB_CONFIG['port']}/{config.DB_CONFIG['database']}"
    )
    return create_engine(conn_string)

def load_to_postgres(df):
    """
    Load transformed data into PostgreSQL

    Args:
        df (pd.DataFrame): Transformed weather data
    """
    print(f"\n{'='*50}")
    print(f"LOAD: Inserting {len(df)} records into PostgreSQL")
    print(f"{'='*50}\n")

    try:
        engine = create_db_engine()

        # Insert data
        df.to_sql(
            name='weather_data',
            con=engine,
            if_exists='append',  # Append to existing table
            index=False,
            method='multi'  # Fast multi-row insert
        )

        print(f"✓ Successfully loaded {len(df)} records to PostgreSQL\n")

        # Verify insertion
        with engine.connect() as conn:
            result = conn.execute("SELECT COUNT(*) FROM weather_data")
            total_rows = result.scalar()
            print(f"📊 Total records in database: {total_rows}")

        engine.dispose()

    except Exception as e:
        print(f"✗ Error loading data to PostgreSQL: {e}")
        raise

if __name__ == "__main__":
    # Test loading
    df = pd.read_csv('data/transformed_weather.csv')
    load_to_postgres(df)
```

**Test it:**
```bash
python src/load.py
```

---

## Step 6: Complete Pipeline

### Create `src/pipeline.py`

```python
# src/pipeline.py
from datetime import datetime
from extract import extract_all_cities
from transform import transform_weather_data
from load import load_to_postgres

def run_etl_pipeline():
    """
    Execute complete ETL pipeline
    """
    start_time = datetime.now()

    print(f"\n{'='*70}")
    print(f"  WEATHER ETL PIPELINE - Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    try:
        # 1. Extract
        raw_data = extract_all_cities()

        if not raw_data:
            print("✗ No data extracted. Aborting pipeline.")
            return

        # 2. Transform
        df = transform_weather_data(raw_data)

        # 3. Load
        load_to_postgres(df)

        # Success!
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\n{'='*70}")
        print(f"  ✓ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Records processed: {len(df)}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"  ✗ PIPELINE FAILED")
        print(f"  Error: {e}")
        print(f"{'='*70}\n")
        raise

if __name__ == "__main__":
    run_etl_pipeline()
```

**Run the complete pipeline:**
```bash
python src/pipeline.py
```

---

## Step 7: Schedule Automation

### Windows (Task Scheduler)

Create `run_pipeline.bat`:
```batch
@echo off
cd C:\path\to\weather-etl-pipeline
C:\path\to\python.exe src\pipeline.py >> logs\pipeline.log 2>&1
```

**Schedule:**
1. Open Task Scheduler
2. Create Basic Task
3. Name: "Weather ETL Pipeline"
4. Trigger: Daily at 8:00 AM
5. Action: Start a program
6. Program: `C:\path\to\run_pipeline.bat`

### Mac/Linux (cron)

Add to crontab:
```bash
crontab -e

# Add this line (runs every day at 8 AM)
0 8 * * * cd /path/to/weather-etl-pipeline && /usr/bin/python3 src/pipeline.py >> logs/pipeline.log 2>&1
```

### Python Scheduler (Cross-platform)

Create `scheduler.py`:
```python
# scheduler.py
import schedule
import time
from src.pipeline import run_etl_pipeline

# Run every day at 8:00 AM
schedule.every().day.at("08:00").do(run_etl_pipeline)

# Or run every 6 hours
# schedule.every(6).hours.do(run_etl_pipeline)

print("Scheduler started. Press Ctrl+C to stop.")

while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
```

Run in background:
```bash
python scheduler.py
```

---

## Step 8: Query Your Data

### Useful SQL Queries

```sql
-- View latest weather for all cities
SELECT
    city_name,
    country,
    temperature,
    humidity,
    weather_description,
    timestamp
FROM weather_data
WHERE timestamp IN (
    SELECT MAX(timestamp)
    FROM weather_data
    GROUP BY city_name
)
ORDER BY temperature DESC;

-- Average temperature by city (last 7 days)
SELECT
    city_name,
    ROUND(AVG(temperature), 2) as avg_temp,
    ROUND(AVG(humidity), 2) as avg_humidity,
    COUNT(*) as num_readings
FROM weather_data
WHERE timestamp > CURRENT_DATE - INTERVAL '7 days'
GROUP BY city_name
ORDER BY avg_temp DESC;

-- Hottest and coldest readings
SELECT
    'Hottest' as type,
    city_name,
    temperature,
    timestamp
FROM weather_data
ORDER BY temperature DESC
LIMIT 5

UNION ALL

SELECT
    'Coldest' as type,
    city_name,
    temperature,
    timestamp
FROM weather_data
ORDER BY temperature ASC
LIMIT 5;

-- Weather trends over time for a specific city
SELECT
    DATE(timestamp) as date,
    MIN(temperature) as min_temp,
    MAX(temperature) as max_temp,
    AVG(temperature) as avg_temp,
    AVG(humidity) as avg_humidity
FROM weather_data
WHERE city_name = 'New York'
GROUP BY DATE(timestamp)
ORDER BY date DESC
LIMIT 30;
```

---

## Verification Checklist

- [ ] API key works (extract runs successfully)
- [ ] Data transforms correctly (no errors in logs)
- [ ] PostgreSQL table created with correct schema
- [ ] Data loads into database (verify with SELECT query)
- [ ] Complete pipeline runs end-to-end
- [ ] Scheduler configured and tested
- [ ] Can query data and see results

---

## Common Issues

### Issue: "No API key" error
**Solution:** Make sure `.env` file exists and contains `OPENWEATHER_API_KEY=your_key`

### Issue: "psycopg2 module not found"
**Solution:**
```bash
pip install psycopg2-binary
```

### Issue: "Can't connect to PostgreSQL"
**Solution:**
1. Check PostgreSQL is running
2. Verify credentials in `.env`
3. Test connection: `psql -U postgres -h localhost`

### Issue: Rate limit error (429)
**Solution:** Free tier has 60 calls/minute. Add delay between cities:
```python
import time
time.sleep(1)  # 1 second delay
```

---

## Next Steps

1. **Add more cities** - Edit `config.py` CITIES list
2. **Visualize data** - Create dashboard with Matplotlib/Seaborn
3. **Add alerts** - Send email if temperature > threshold
4. **Expand sources** - Add more APIs (pollution, traffic, etc.)
5. **Deploy** - Run on cloud VM (AWS EC2, DigitalOcean)

---

## Resources

- [OpenWeather API Docs](https://openweathermap.org/api)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [SQLAlchemy Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/)
- [Schedule Library](https://schedule.readthedocs.io/)
