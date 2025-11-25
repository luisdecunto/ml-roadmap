# SQL Window Functions Exercises - Flight Data (Mode Public Warehouse)

These exercises use Mode's real flight dataset: `tutorial.flights` to master SQL Window Functions

💡 **Solutions included - Click to expand under each exercise!**

---

## 📊 Database Schema

### `tutorial.flights` Table
**Complete schema:**
- `airline_code` - Airline code (string)
- `airline_name` - Full airline name (string)
- `flight_number` - Flight number (float)
- `day` - Flight date (timestamp)
- `day_of_week` - Day of week (string)
- `origin_airport`, `origin_city`, `origin_state` - Departure details
- `destination_airport`, `destination_city`, `destination_state` - Arrival details
- `scheduled_departure_time`, `acutal_departure_time` - Departure times
- `scheduled_arrival_time`, `actual_arrival_time` - Arrival times
- `departure_delay`, `arrival_delay` - Delay in minutes (float)
- `scheduled_flight_time`, `actual_flight_time`, `air_time` - Duration measures
- `distance` - Flight distance (float)
- `was_cancelled` - Boolean cancellation flag
- `cancellation_reason` - Reason for cancellation (string)
- `carrier_delay`, `weather_delay`, `late_aircraft_delay`, `air_traffic_delay`, `security_delay` - Delay breakdown (float)
- `wheels_off_time`, `wheels_on_time` - Takeoff/landing times

---

## ✅ Progress Tracker

### Easy Exercises (ROW_NUMBER, RANK, Basic PARTITION BY)
- [ ] Exercise 1: Rank Flights by Delay Within Each Airline (ROW_NUMBER)
- [ ] Exercise 2: Rank Airlines by Average Delay (RANK vs DENSE_RANK)
- [ ] Exercise 3: Number Flights per Airline per Day (PARTITION BY)

### Medium Exercises (LAG/LEAD, Running Totals, Moving Averages)
- [ ] Exercise 4: Compare Flight Delay to Previous Flight on Same Route (LAG/LEAD)
- [ ] Exercise 5: Calculate Cumulative Delays by Airline Over Time (Running Totals)
- [ ] Exercise 6: Calculate 7-Day Moving Average of Delays by Airline (Moving Averages)

### Hard Exercises (NTILE, Complex Frames, ROWS/RANGE)
- [ ] Exercise 7: Divide Flights into Quartiles by Distance and Analyze Delays (NTILE)
- [ ] Exercise 8: Calculate Delays Relative to Route Average (ROWS/RANGE)

### Challenge Exercises (Advanced Window Functions)
- [ ] Challenge 1: Find Consecutive Days an Airline Operated (Gaps and Islands)
- [ ] Challenge 2: Find Flights in Top 10% of Delays (PERCENT_RANK)
- [ ] Challenge 3: Compare Each Flight to Route Best/Worst Performance (FIRST_VALUE/LAST_VALUE)

---

## 🎯 Easy Exercises

### Exercise 1: Rank Flights by Delay Within Each Airline
**Task**: For each airline, rank all flights by arrival delay (highest delay = rank 1). Show the top 3 most delayed flights for each airline.

**Database**: `tutorial.flights`

**Your Solution:**
```sql
-- Write your window function query here
```

**Hint**:
- Use ROW_NUMBER() OVER (PARTITION BY airline_name ORDER BY arrival_delay DESC)
- PARTITION BY creates separate rankings for each airline
- Filter for row_number <= 3 to get top 3
- Consider using a CTE or subquery

**Expected Output**: Should show top 3 delayed flights for each airline with their rank

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH ranked_flights AS (
    SELECT
        airline_name,
        flight_number,
        day,
        origin_airport,
        destination_airport,
        arrival_delay,
        ROW_NUMBER() OVER (
            PARTITION BY airline_name
            ORDER BY arrival_delay DESC
        ) as delay_rank
    FROM tutorial.flights
    WHERE arrival_delay IS NOT NULL
)
SELECT
    delay_rank,
    airline_name,
    flight_number,
    day,
    origin_airport,
    destination_airport,
    arrival_delay
FROM ranked_flights
WHERE delay_rank <= 3
ORDER BY airline_name, delay_rank;
```

**Explanation:**
1. **ROW_NUMBER()** assigns a unique sequential number to each row
2. **PARTITION BY airline_name** creates separate rankings for each airline
3. **ORDER BY arrival_delay DESC** ensures highest delays get rank 1
4. **WHERE delay_rank <= 3** filters for top 3 in each airline
5. Results show most delayed flights per airline

**Key Concepts:**
- ROW_NUMBER() assigns unique numbers (no ties)
- PARTITION BY creates separate "windows" for each group
- Window functions don't reduce rows like GROUP BY does
- Must use CTE or subquery to filter on window function results

**Expected Output Pattern:**
```
delay_rank | airline_name      | flight_number | arrival_delay
1          | American Airlines | 1523          | 1425
2          | American Airlines | 892           | 1398
3          | American Airlines | 2341          | 1267
1          | Delta Air Lines   | 1045          | 1502
2          | Delta Air Lines   | 3421          | 1456
3          | Delta Air Lines   | 892           | 1234
```

</details>

---

### Exercise 2: Rank Airlines by Average Delay
**Task**: Calculate average arrival delay for each airline and rank them. Show the difference between RANK() and DENSE_RANK() when airlines have the same average delay.

**Database**: `tutorial.flights`

**Your Solution:**
```sql
-- Write your query here
```

**Hint**:
- First aggregate: GROUP BY airline_name and calculate AVG(arrival_delay)
- Use both RANK() and DENSE_RANK() in window functions
- RANK() skips numbers after ties (1, 2, 2, 4)
- DENSE_RANK() doesn't skip (1, 2, 2, 3)
- No PARTITION BY needed (ranking across all airlines)

**Expected Output**: Airlines ranked by average delay showing both rank types

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH airline_avg_delays AS (
    SELECT
        airline_name,
        COUNT(*) as total_flights,
        ROUND(AVG(arrival_delay), 2) as avg_arrival_delay,
        ROUND(AVG(departure_delay), 2) as avg_departure_delay
    FROM tutorial.flights
    WHERE arrival_delay IS NOT NULL
    GROUP BY airline_name
)
SELECT
    airline_name,
    total_flights,
    avg_arrival_delay,
    avg_departure_delay,
    RANK() OVER (ORDER BY avg_arrival_delay DESC) as rank_with_gaps,
    DENSE_RANK() OVER (ORDER BY avg_arrival_delay DESC) as dense_rank_no_gaps,
    ROW_NUMBER() OVER (ORDER BY avg_arrival_delay DESC) as row_number_unique
FROM airline_avg_delays
ORDER BY avg_arrival_delay DESC;
```

**Explanation:**
1. **airline_avg_delays CTE** aggregates flights by airline
2. **RANK()** assigns same rank to ties, then skips numbers (1,2,2,4)
3. **DENSE_RANK()** assigns same rank to ties, no gaps (1,2,2,3)
4. **ROW_NUMBER()** always assigns unique numbers, even for ties
5. All three functions ordered by avg_arrival_delay DESC

**Key Concepts:**
- **ROW_NUMBER()**: Always unique (1,2,3,4,5...)
- **RANK()**: Ties get same rank, gaps after ties (1,2,2,4,5)
- **DENSE_RANK()**: Ties get same rank, no gaps (1,2,2,3,4)
- When to use each:
  - ROW_NUMBER: Need unique identifier
  - RANK: Traditional ranking (like sports rankings)
  - DENSE_RANK: Categories/levels without gaps

**Expected Output Pattern:**
```
airline_name      | total_flights | avg_arrival_delay | rank | dense_rank | row_number
SkyWest Airlines  | 15000         | 28.45             | 1    | 1          | 1
Spirit Airlines   | 8500          | 26.82             | 2    | 2          | 2
Southwest         | 22000         | 24.10             | 3    | 3          | 3
Alaska Airlines   | 9200          | 24.10             | 3    | 3          | 4  (tie broken)
Frontier          | 6800          | 22.50             | 5    | 4          | 5  (gap after tie)
```

**Notice**: Airlines with same avg delay (24.10) get same RANK (3) and DENSE_RANK (3), but different ROW_NUMBER. After the tie, RANK skips to 5, but DENSE_RANK goes to 4.

</details>

---

### Exercise 3: Number Flights per Airline per Day
**Task**: For each airline and date combination, assign sequential flight numbers showing which flight was 1st, 2nd, 3rd, etc. of the day for that airline.

**Database**: `tutorial.flights`

**Your Solution:**
```sql
-- Write your query here
```

**Hint**:
- Use PARTITION BY airline_name, CAST(day AS DATE)
- ORDER BY scheduled_departure_time within each partition
- ROW_NUMBER() will give sequential numbers
- Show only first 5 flights per airline per day for readability

**Expected Output**: Each flight numbered sequentially within its airline-day combination

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH daily_flight_sequence AS (
    SELECT
        airline_name,
        CAST(day AS DATE) as flight_date,
        flight_number,
        origin_airport,
        destination_airport,
        scheduled_departure_time,
        actual_departure_time,
        departure_delay,
        ROW_NUMBER() OVER (
            PARTITION BY airline_name, CAST(day AS DATE)
            ORDER BY scheduled_departure_time
        ) as daily_flight_sequence
    FROM tutorial.flights
    WHERE day IS NOT NULL
        AND scheduled_departure_time IS NOT NULL
)
SELECT
    airline_name,
    flight_date,
    daily_flight_sequence,
    flight_number,
    origin_airport,
    destination_airport,
    scheduled_departure_time,
    departure_delay
FROM daily_flight_sequence
WHERE daily_flight_sequence <= 5  -- Show first 5 flights per airline per day
ORDER BY airline_name, flight_date, daily_flight_sequence;
```

**Explanation:**
1. **PARTITION BY airline_name, CAST(day AS DATE)** creates separate sequences for each airline-day
2. **ORDER BY scheduled_departure_time** sequences flights by departure time
3. **ROW_NUMBER()** assigns 1, 2, 3... to each flight in the sequence
4. Each airline-day combination restarts at 1
5. Filtering for <= 5 shows sample results

**Key Concepts:**
- PARTITION BY with multiple columns creates finer groupings
- ROW_NUMBER() restarts at 1 for each partition
- ORDER BY determines the sequence within partition
- Useful for "nth occurrence" problems

**Example Use Cases:**
- First flight of the day for each airline
- Last flight before midnight
- 10th customer transaction per month
- 3rd order by each customer

**Expected Output Pattern:**
```
airline_name      | flight_date | daily_flight_sequence | flight_number | scheduled_departure_time
American Airlines | 2015-01-01  | 1                     | 123           | 05:30
American Airlines | 2015-01-01  | 2                     | 456           | 06:15
American Airlines | 2015-01-01  | 3                     | 789           | 07:00
American Airlines | 2015-01-01  | 4                     | 234           | 07:45
American Airlines | 2015-01-01  | 5                     | 567           | 08:20
American Airlines | 2015-01-02  | 1                     | 890           | 05:35  (resets for new day)
```

</details>

---

## 📊 Medium Exercises

### Exercise 4: Compare Flight Delay to Previous/Next Flight on Same Route
**Task**: For each route (origin-destination pair), show each flight's delay alongside the previous flight's delay and next flight's delay on the same route. Calculate the difference between current and previous delay.

**Database**: `tutorial.flights`

**Your Solution:**
```sql
-- Write your query here
```

**Hint**:
- Use LAG() to get previous flight's delay
- Use LEAD() to get next flight's delay
- PARTITION BY origin_airport, destination_airport
- ORDER BY day, scheduled_departure_time
- Calculate difference: arrival_delay - LAG(arrival_delay)

**Expected Output**: Flights with previous/next delays and comparison

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH route_delay_comparison AS (
    SELECT
        airline_name,
        flight_number,
        day,
        origin_airport,
        destination_airport,
        arrival_delay,
        LAG(arrival_delay, 1) OVER (
            PARTITION BY origin_airport, destination_airport
            ORDER BY day, scheduled_departure_time
        ) as previous_flight_delay,
        LEAD(arrival_delay, 1) OVER (
            PARTITION BY origin_airport, destination_airport
            ORDER BY day, scheduled_departure_time
        ) as next_flight_delay,
        LAG(airline_name, 1) OVER (
            PARTITION BY origin_airport, destination_airport
            ORDER BY day, scheduled_departure_time
        ) as previous_airline
    FROM tutorial.flights
    WHERE arrival_delay IS NOT NULL
        AND origin_airport IS NOT NULL
        AND destination_airport IS NOT NULL
)
SELECT
    airline_name,
    flight_number,
    day,
    origin_airport,
    destination_airport,
    arrival_delay as current_delay,
    previous_flight_delay,
    next_flight_delay,
    previous_airline,
    ROUND(arrival_delay - previous_flight_delay, 2) as delay_change_from_previous,
    CASE
        WHEN previous_flight_delay IS NULL THEN 'First flight on route'
        WHEN arrival_delay > previous_flight_delay THEN 'Worse than previous'
        WHEN arrival_delay < previous_flight_delay THEN 'Better than previous'
        ELSE 'Same as previous'
    END as comparison_to_previous
FROM route_delay_comparison
WHERE origin_airport = 'LAX'
    AND destination_airport = 'JFK'  -- Focus on one route for clarity
ORDER BY day, arrival_delay DESC
LIMIT 20;
```

**Explanation:**
1. **LAG(arrival_delay, 1)** gets the delay from the previous row in the window
2. **LEAD(arrival_delay, 1)** gets the delay from the next row in the window
3. **PARTITION BY origin_airport, destination_airport** groups by route
4. **ORDER BY day, scheduled_departure_time** sequences flights chronologically
5. **Offset of 1** means immediately previous/next (use 2 for 2 rows back, etc.)
6. First row in each partition: LAG returns NULL
7. Last row in each partition: LEAD returns NULL

**Key Concepts:**
- **LAG(column, offset)**: Access previous row's value
- **LEAD(column, offset)**: Access next row's value
- Default offset is 1 (immediate previous/next)
- NULL when no previous/next row exists
- Useful for:
  - Time series comparisons
  - Sequential analysis
  - Trend detection
  - Change tracking

**Expected Output Pattern:**
```
airline_name | flight_number | day        | current_delay | previous_flight_delay | delay_change | comparison
Delta        | 123           | 2015-01-05 | 45            | NULL                  | NULL         | First flight
American     | 456           | 2015-01-05 | 30            | 45                    | -15          | Better than previous
United       | 789           | 2015-01-05 | 60            | 30                    | +30          | Worse than previous
```

**Practical Use:**
- Did this flight improve on-time performance compared to the previous flight?
- Are delays getting better or worse on this route?
- Identify patterns in consecutive flights

</details>

---

### Exercise 5: Calculate Cumulative Delays by Airline Over Time
**Task**: For each airline, calculate the running total of arrival delays over time. Show how total delay accumulates day by day.

**Database**: `tutorial.flights`

**Your Solution:**
```sql
-- Write your query here
```

**Hint**:
- First aggregate: total delay per airline per day
- Use SUM() OVER with PARTITION BY airline_name
- ORDER BY date within partition
- Add ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW (or omit for default)
- Running totals reset for each airline

**Expected Output**: Daily cumulative delay totals for each airline

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH daily_airline_delays AS (
    SELECT
        airline_name,
        CAST(day AS DATE) as flight_date,
        COUNT(*) as flights_count,
        ROUND(SUM(arrival_delay), 2) as total_daily_delay,
        ROUND(AVG(arrival_delay), 2) as avg_daily_delay
    FROM tutorial.flights
    WHERE arrival_delay IS NOT NULL
    GROUP BY airline_name, CAST(day AS DATE)
),
cumulative_delays AS (
    SELECT
        airline_name,
        flight_date,
        flights_count,
        total_daily_delay,
        avg_daily_delay,
        -- Running total of delay minutes
        ROUND(SUM(total_daily_delay) OVER (
            PARTITION BY airline_name
            ORDER BY flight_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ), 2) as cumulative_delay,
        -- Running average of daily delays
        ROUND(AVG(total_daily_delay) OVER (
            PARTITION BY airline_name
            ORDER BY flight_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ), 2) as running_avg_delay,
        -- Running count of flights
        SUM(flights_count) OVER (
            PARTITION BY airline_name
            ORDER BY flight_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as cumulative_flights
    FROM daily_airline_delays
)
SELECT
    airline_name,
    flight_date,
    flights_count,
    total_daily_delay,
    cumulative_delay,
    cumulative_flights,
    running_avg_delay
FROM cumulative_delays
WHERE airline_name IN ('Southwest Airlines', 'Delta Air Lines', 'American Airlines')
ORDER BY airline_name, flight_date
LIMIT 100;
```

**Explanation:**
1. **daily_airline_delays CTE** aggregates delays by airline and date
2. **SUM() OVER** calculates running total
3. **PARTITION BY airline_name** creates separate running totals per airline
4. **ORDER BY flight_date** ensures chronological accumulation
5. **ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW** defines the frame
   - UNBOUNDED PRECEDING = from start of partition
   - CURRENT ROW = up to and including current row
6. Each airline's cumulative total resets independently

**Key Concepts:**
- **Running/Cumulative Totals**: SUM() OVER with ORDER BY
- **Frame Clause**: ROWS BETWEEN defines which rows to include
- **UNBOUNDED PRECEDING**: From the start of the partition
- **CURRENT ROW**: Up to the current row
- Default frame when ORDER BY present: RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW

**Frame Options:**
```sql
-- All rows from start to current
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW

-- All rows in partition (grand total)
ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING

-- Last 7 rows including current (7-day window)
ROWS BETWEEN 6 PRECEDING AND CURRENT ROW

-- Centered 3-row window
ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
```

**Expected Output Pattern:**
```
airline_name         | flight_date | flights_count | total_daily_delay | cumulative_delay | cumulative_flights
American Airlines    | 2015-01-01  | 150           | 2400              | 2400             | 150
American Airlines    | 2015-01-02  | 145           | 2100              | 4500             | 295
American Airlines    | 2015-01-03  | 148           | 2350              | 6850             | 443
American Airlines    | 2015-01-04  | 152           | 2600              | 9450             | 595
```

**Notice**: cumulative_delay keeps adding up, cumulative_flights keeps growing

</details>

---

### Exercise 6: Calculate 7-Day Moving Average of Delays by Airline
**Task**: For each airline, calculate a 7-day moving average of daily delay. This smooths out day-to-day variations to show trends.

**Database**: `tutorial.flights`

**Your Solution:**
```sql
-- Write your query here
```

**Hint**:
- Aggregate to airline-day level first
- Use AVG() OVER with ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
- This creates a 7-day window (6 previous days + current day)
- PARTITION BY airline_name
- ORDER BY date

**Expected Output**: Daily delays with 7-day moving average

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH daily_airline_stats AS (
    SELECT
        airline_name,
        CAST(day AS DATE) as flight_date,
        COUNT(*) as daily_flights,
        ROUND(AVG(arrival_delay), 2) as avg_daily_delay,
        ROUND(SUM(arrival_delay), 2) as total_daily_delay,
        ROUND(MAX(arrival_delay), 2) as max_daily_delay
    FROM tutorial.flights
    WHERE arrival_delay IS NOT NULL
    GROUP BY airline_name, CAST(day AS DATE)
),
moving_averages AS (
    SELECT
        airline_name,
        flight_date,
        daily_flights,
        avg_daily_delay,
        total_daily_delay,
        -- 7-day moving average (current day + 6 preceding)
        ROUND(AVG(avg_daily_delay) OVER (
            PARTITION BY airline_name
            ORDER BY flight_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 2) as moving_avg_7day,
        -- 7-day moving sum
        ROUND(SUM(total_daily_delay) OVER (
            PARTITION BY airline_name
            ORDER BY flight_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 2) as moving_sum_7day,
        -- 3-day moving average for comparison
        ROUND(AVG(avg_daily_delay) OVER (
            PARTITION BY airline_name
            ORDER BY flight_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ), 2) as moving_avg_3day,
        -- Count how many days in the window (for start of series)
        COUNT(*) OVER (
            PARTITION BY airline_name
            ORDER BY flight_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as days_in_window
    FROM daily_airline_stats
)
SELECT
    airline_name,
    flight_date,
    daily_flights,
    avg_daily_delay,
    moving_avg_3day,
    moving_avg_7day,
    days_in_window,
    CASE
        WHEN avg_daily_delay > moving_avg_7day THEN 'Above trend'
        WHEN avg_daily_delay < moving_avg_7day THEN 'Below trend'
        ELSE 'At trend'
    END as vs_trend
FROM moving_averages
WHERE airline_name IN ('Southwest Airlines', 'Delta Air Lines')
ORDER BY airline_name, flight_date
LIMIT 50;
```

**Explanation:**
1. **daily_airline_stats** aggregates to airline-day level
2. **AVG() OVER with ROWS BETWEEN 6 PRECEDING AND CURRENT ROW** creates 7-day window
   - 6 PRECEDING = previous 6 rows
   - CURRENT ROW = current row
   - Total = 7 days
3. **PARTITION BY airline_name** separates airlines
4. **ORDER BY flight_date** ensures chronological order
5. First 6 days have fewer than 7 days in window
6. **days_in_window** shows actual window size (useful for validation)

**Key Concepts:**
- **Moving Average**: Smooths out short-term fluctuations
- **ROWS BETWEEN n PRECEDING AND CURRENT ROW**: Creates sliding window
- **Window Size**: Choose based on data:
  - 7 days: Weekly patterns
  - 30 days: Monthly trends
  - 90 days: Quarterly trends
- First n-1 rows have partial windows

**Common Moving Average Windows:**
```sql
-- 7-day moving average
ROWS BETWEEN 6 PRECEDING AND CURRENT ROW

-- 30-day moving average
ROWS BETWEEN 29 PRECEDING AND CURRENT ROW

-- Centered 7-day average (3 before, current, 3 after)
ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING

-- All previous rows (cumulative average)
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
```

**Expected Output Pattern:**
```
airline_name     | flight_date | avg_daily_delay | moving_avg_7day | days_in_window | vs_trend
Southwest        | 2015-01-01  | 15.5            | 15.5            | 1              | At trend
Southwest        | 2015-01-02  | 18.2            | 16.85           | 2              | Above trend
Southwest        | 2015-01-03  | 12.1            | 15.27           | 3              | Below trend
Southwest        | 2015-01-04  | 16.8            | 15.65           | 4              | Above trend
Southwest        | 2015-01-05  | 14.3            | 15.38           | 5              | Below trend
Southwest        | 2015-01-06  | 17.9            | 15.8            | 6              | Above trend
Southwest        | 2015-01-07  | 13.2            | 15.43           | 7              | Below trend (full window)
Southwest        | 2015-01-08  | 19.5            | 16.0            | 7              | Above trend
```

**Notice**:
- First 6 days have incomplete windows (days_in_window < 7)
- Day 7 onwards has full 7-day windows
- Moving average is smoother than daily values

**Use Cases:**
- Stock prices (50-day, 200-day moving averages)
- Sales trends
- Temperature/weather patterns
- Website traffic
- Customer metrics

</details>

---

## 🔥 Hard Exercises

### Exercise 7: Divide Flights into Quartiles by Distance and Analyze Delays
**Task**: Use NTILE to divide all flights into 4 equal groups (quartiles) based on distance. Then analyze average delay for each distance quartile. Are longer flights more delayed?

**Database**: `tutorial.flights`

**Your Solution:**
```sql
-- Write your query here
```

**Hint**:
- Use NTILE(4) OVER (ORDER BY distance)
- This divides flights into 4 equal-sized groups
- NTILE(4) assigns 1, 2, 3, or 4 to each row
- Group by the quartile and calculate aggregate statistics
- Compare delays across quartiles

**Expected Output**: Delay statistics for each distance quartile

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH flight_distance_quartiles AS (
    SELECT
        airline_name,
        flight_number,
        origin_airport,
        destination_airport,
        distance,
        arrival_delay,
        departure_delay,
        -- Divide into 4 equal groups by distance
        NTILE(4) OVER (ORDER BY distance) as distance_quartile,
        -- Also add percentile rank for comparison
        PERCENT_RANK() OVER (ORDER BY distance) as distance_percentile
    FROM tutorial.flights
    WHERE distance IS NOT NULL
        AND arrival_delay IS NOT NULL
),
quartile_analysis AS (
    SELECT
        distance_quartile,
        COUNT(*) as flights_in_quartile,
        ROUND(MIN(distance), 0) as min_distance,
        ROUND(MAX(distance), 0) as max_distance,
        ROUND(AVG(distance), 0) as avg_distance,
        ROUND(AVG(arrival_delay), 2) as avg_arrival_delay,
        ROUND(AVG(departure_delay), 2) as avg_departure_delay,
        ROUND(STDDEV(arrival_delay), 2) as stddev_arrival_delay,
        ROUND(MAX(arrival_delay), 2) as max_arrival_delay,
        ROUND(MIN(arrival_delay), 2) as min_arrival_delay,
        -- Percentage of flights delayed
        ROUND(100.0 * SUM(CASE WHEN arrival_delay > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_delayed
    FROM flight_distance_quartiles
    GROUP BY distance_quartile
)
SELECT
    CASE distance_quartile
        WHEN 1 THEN '1st Quartile (Shortest)'
        WHEN 2 THEN '2nd Quartile'
        WHEN 3 THEN '3rd Quartile'
        WHEN 4 THEN '4th Quartile (Longest)'
    END as quartile_label,
    distance_quartile,
    flights_in_quartile,
    min_distance,
    max_distance,
    avg_distance,
    avg_arrival_delay,
    avg_departure_delay,
    pct_delayed,
    stddev_arrival_delay,
    max_arrival_delay
FROM quartile_analysis
ORDER BY distance_quartile;
```

**Explanation:**
1. **NTILE(4)** divides flights into 4 approximately equal groups
2. **ORDER BY distance** means groups are based on distance
3. Quartile 1 = shortest 25% of flights
4. Quartile 4 = longest 25% of flights
5. Groups are as equal in size as possible
6. **quartile_analysis** aggregates statistics for each quartile
7. Shows if delay patterns differ by flight distance

**Key Concepts:**
- **NTILE(n)**: Divides rows into n approximately equal groups
- Common values: NTILE(4) for quartiles, NTILE(10) for deciles, NTILE(100) for percentiles
- Groups numbered 1 to n
- If total rows not evenly divisible, some groups get one extra row
- Useful for:
  - Performance segmentation
  - ABC analysis
  - Risk categorization
  - Portfolio allocation

**NTILE vs Other Ranking Functions:**
```sql
-- These operate on the same data differently:
ROW_NUMBER() OVER (ORDER BY value)  -- 1,2,3,4,5,6,7,8...
RANK() OVER (ORDER BY value)        -- 1,2,2,4,5,5,7,8...
DENSE_RANK() OVER (ORDER BY value)  -- 1,2,2,3,4,4,5,6...
NTILE(4) OVER (ORDER BY value)      -- 1,1,2,2,3,3,4,4... (equal-sized groups)
```

**Expected Output Pattern:**
```
quartile_label              | distance_quartile | flights_in_quartile | min_distance | max_distance | avg_distance | avg_arrival_delay | pct_delayed
1st Quartile (Shortest)     | 1                 | 125000              | 31           | 435          | 280          | 8.45              | 42.3
2nd Quartile                | 2                 | 125000              | 436          | 872          | 650          | 11.23             | 48.7
3rd Quartile                | 3                 | 125000              | 873          | 1450         | 1150         | 13.67             | 52.1
4th Quartile (Longest)      | 4                 | 125000              | 1451         | 4983         | 2100         | 15.89             | 55.8
```

**Analysis Questions:**
- Do longer flights have more delays?
- Is the percentage of delayed flights higher for long distances?
- Is delay variability (STDDEV) different across quartiles?

**Sample Extended Query - Per Airline Quartile Analysis:**
```sql
-- Analyze each airline's flights divided by distance
SELECT
    airline_name,
    NTILE(4) OVER (PARTITION BY airline_name ORDER BY distance) as airline_distance_quartile,
    distance,
    arrival_delay
FROM tutorial.flights;
```

</details>

---

### Exercise 8: Calculate Delays Relative to Route Average Using Complex Frames
**Task**: For each flight, calculate how its delay compares to the route average, but only considering flights within +/- 3 days. This uses RANGE frame specification.

**Database**: `tutorial.flights`

**Your Solution:**
```sql
-- Write your query here
```

**Hint**:
- PARTITION BY origin_airport, destination_airport
- ORDER BY day (date column)
- Use RANGE BETWEEN INTERVAL 3 DAY PRECEDING AND INTERVAL 3 DAY FOLLOWING
- Calculate AVG(arrival_delay) over this frame
- Compare current flight to this local average

**Expected Output**: Flights with their delay vs recent route average

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH route_context AS (
    SELECT
        airline_name,
        flight_number,
        CAST(day AS DATE) as flight_date,
        origin_airport,
        destination_airport,
        arrival_delay,
        -- Average delay for this route across all time
        ROUND(AVG(arrival_delay) OVER (
            PARTITION BY origin_airport, destination_airport
        ), 2) as route_overall_avg,
        -- Average delay for this route in a 7-day window (3 days before, current, 3 days after)
        ROUND(AVG(arrival_delay) OVER (
            PARTITION BY origin_airport, destination_airport
            ORDER BY CAST(day AS DATE)
            ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
        ), 2) as route_7day_avg,
        -- Count of flights in the 7-day window
        COUNT(*) OVER (
            PARTITION BY origin_airport, destination_airport
            ORDER BY CAST(day AS DATE)
            ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
        ) as flights_in_7day_window,
        -- Min and max delays in the window
        MIN(arrival_delay) OVER (
            PARTITION BY origin_airport, destination_airport
            ORDER BY CAST(day AS DATE)
            ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
        ) as min_delay_in_window,
        MAX(arrival_delay) OVER (
            PARTITION BY origin_airport, destination_airport
            ORDER BY CAST(day AS DATE)
            ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
        ) as max_delay_in_window
    FROM tutorial.flights
    WHERE arrival_delay IS NOT NULL
        AND origin_airport IS NOT NULL
        AND destination_airport IS NOT NULL
)
SELECT
    airline_name,
    flight_number,
    flight_date,
    origin_airport,
    destination_airport,
    arrival_delay as current_delay,
    route_overall_avg,
    route_7day_avg,
    flights_in_7day_window,
    ROUND(arrival_delay - route_overall_avg, 2) as vs_route_overall,
    ROUND(arrival_delay - route_7day_avg, 2) as vs_route_recent,
    CASE
        WHEN arrival_delay > route_7day_avg + 15 THEN 'Much worse than recent'
        WHEN arrival_delay > route_7day_avg THEN 'Worse than recent'
        WHEN arrival_delay < route_7day_avg - 15 THEN 'Much better than recent'
        WHEN arrival_delay < route_7day_avg THEN 'Better than recent'
        ELSE 'Similar to recent'
    END as performance_vs_recent,
    min_delay_in_window,
    max_delay_in_window
FROM route_context
WHERE origin_airport = 'LAX'
    AND destination_airport = 'SFO'
ORDER BY flight_date, arrival_delay DESC
LIMIT 50;
```

**Explanation:**
1. **PARTITION BY origin_airport, destination_airport** creates separate windows per route
2. **First window (no ORDER BY)**: Calculates overall route average across all time
3. **Second window (with ORDER BY and ROWS BETWEEN)**: Creates sliding 7-day window
   - 3 PRECEDING = 3 rows before current
   - CURRENT ROW = current row
   - 3 FOLLOWING = 3 rows after current
   - Total = 7-row window centered on current flight
4. Compares current flight to both overall and recent averages
5. Shows if this flight is typical or outlier for the route

**Key Concepts:**
- **ROWS BETWEEN**: Counts physical rows (3 rows before/after)
- **RANGE BETWEEN**: Uses logical ranges based on ORDER BY value
- **Symmetric window**: n PRECEDING AND n FOLLOWING centers window on current row
- **No ORDER BY in window**: Calculates over entire partition

**ROWS vs RANGE:**
```sql
-- ROWS: Physical row count (always includes exactly n rows)
ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING

-- RANGE: Logical value range (may include more or fewer rows if duplicates)
RANGE BETWEEN 3 PRECEDING AND 3 FOLLOWING

-- Example: If multiple flights on same day
-- ROWS: Includes exactly 3 rows before
-- RANGE: Includes all rows within 3 days before (could be 0 or 20 flights)
```

**Frame Specification Options:**
```sql
-- No ORDER BY: entire partition (default frame)
AVG(x) OVER (PARTITION BY group)

-- With ORDER BY: from start to current (default frame)
AVG(x) OVER (PARTITION BY group ORDER BY date)

-- Explicit frame: sliding window
AVG(x) OVER (PARTITION BY group ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)

-- Centered window
AVG(x) OVER (PARTITION BY group ORDER BY date ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING)

-- Entire partition explicitly
AVG(x) OVER (PARTITION BY group ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
```

**Expected Output Pattern:**
```
airline_name | flight_number | flight_date | current_delay | route_overall_avg | route_7day_avg | vs_route_recent | performance
United       | 123           | 2015-01-05  | 45            | 15.5              | 18.2           | +26.8           | Much worse
Delta        | 456           | 2015-01-05  | 12            | 15.5              | 18.2           | -6.2            | Better
Southwest    | 789           | 2015-01-06  | 8             | 15.5              | 16.8           | -8.8            | Better
```

**Use Cases:**
- Comparing to recent trends vs long-term average
- Anomaly detection (flights much worse than recent pattern)
- Seasonal pattern analysis
- Quality control (is this data point consistent with recent data?)

</details>

---

## 🏆 Challenge Exercises

### Challenge 1: Find Consecutive Days an Airline Operated (Gaps and Islands)
**Task**: Find sequences of consecutive days where each airline operated flights. This is a "gaps and islands" problem - identifying continuous sequences with breaks.

**Database**: `tutorial.flights`

**Your Solution:**
```sql
-- Write your query here
-- Hint: Use ROW_NUMBER() twice with different ORDER BY to find gaps
```

**Hint**:
- Get distinct airline-date combinations
- Use ROW_NUMBER() to create a sequential number
- Subtract date from row number to find "islands" (consecutive sequences)
- Group by the island identifier to find sequence length

**Expected Output**: Start date, end date, and length of consecutive operating days per airline

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH airline_operating_days AS (
    -- Get distinct days each airline operated
    SELECT DISTINCT
        airline_name,
        CAST(day AS DATE) as operating_date
    FROM tutorial.flights
    WHERE day IS NOT NULL
),
days_with_sequence AS (
    -- Add sequential row number
    SELECT
        airline_name,
        operating_date,
        ROW_NUMBER() OVER (
            PARTITION BY airline_name
            ORDER BY operating_date
        ) as seq_num
    FROM airline_operating_days
),
island_identifier AS (
    -- Create island ID by subtracting sequence from date
    -- Consecutive dates will have same island_id
    SELECT
        airline_name,
        operating_date,
        seq_num,
        DATE_SUB(operating_date, INTERVAL seq_num DAY) as island_id
    FROM days_with_sequence
),
consecutive_sequences AS (
    -- Group by island to find each consecutive sequence
    SELECT
        airline_name,
        island_id,
        MIN(operating_date) as sequence_start,
        MAX(operating_date) as sequence_end,
        COUNT(*) as consecutive_days,
        COUNT(*) - 1 as days_between_start_end
    FROM island_identifier
    GROUP BY airline_name, island_id
)
SELECT
    airline_name,
    sequence_start,
    sequence_end,
    consecutive_days,
    CASE
        WHEN consecutive_days = 1 THEN 'Single day operation'
        WHEN consecutive_days <= 7 THEN 'Short sequence (< 1 week)'
        WHEN consecutive_days <= 30 THEN 'Medium sequence (< 1 month)'
        ELSE 'Long sequence (1+ month)'
    END as sequence_type
FROM consecutive_sequences
WHERE consecutive_days >= 5  -- Show sequences of 5+ days
ORDER BY airline_name, consecutive_days DESC, sequence_start
LIMIT 50;
```

**Alternative Solution (Using LAG for Gap Detection):**
```sql
WITH airline_operating_days AS (
    SELECT DISTINCT
        airline_name,
        CAST(day AS DATE) as operating_date
    FROM tutorial.flights
    WHERE day IS NOT NULL
),
days_with_previous AS (
    SELECT
        airline_name,
        operating_date,
        LAG(operating_date, 1) OVER (
            PARTITION BY airline_name
            ORDER BY operating_date
        ) as previous_operating_date
    FROM airline_operating_days
),
days_with_gap_flag AS (
    SELECT
        airline_name,
        operating_date,
        previous_operating_date,
        -- Flag when there's a gap (not consecutive days)
        CASE
            WHEN previous_operating_date IS NULL THEN 1  -- First day
            WHEN DATEDIFF(operating_date, previous_operating_date) > 1 THEN 1  -- Gap found
            ELSE 0  -- Consecutive
        END as is_new_sequence
    FROM days_with_previous
),
sequences_numbered AS (
    SELECT
        airline_name,
        operating_date,
        is_new_sequence,
        -- Running sum of gap flags creates unique sequence ID
        SUM(is_new_sequence) OVER (
            PARTITION BY airline_name
            ORDER BY operating_date
        ) as sequence_id
    FROM days_with_gap_flag
),
consecutive_sequences AS (
    SELECT
        airline_name,
        sequence_id,
        MIN(operating_date) as sequence_start,
        MAX(operating_date) as sequence_end,
        COUNT(*) as consecutive_days
    FROM sequences_numbered
    GROUP BY airline_name, sequence_id
)
SELECT
    airline_name,
    sequence_start,
    sequence_end,
    consecutive_days
FROM consecutive_sequences
WHERE consecutive_days >= 5
ORDER BY airline_name, consecutive_days DESC;
```

**Explanation:**

**Method 1 (Date Arithmetic):**
1. Get distinct operating dates per airline
2. Assign sequential numbers (1, 2, 3...) to dates
3. Subtract sequence number from date:
   - If dates are consecutive: 2015-01-01 - 1, 2015-01-02 - 2, 2015-01-03 - 3 → all give same result
   - If gap exists: result changes
4. Same result = same "island" (consecutive sequence)
5. Group by island to get sequence details

**Method 2 (Gap Detection):**
1. Use LAG to get previous operating date
2. Flag when gap > 1 day (or first date)
3. Running sum of flags creates unique sequence ID
4. Group by sequence ID

**Key Concepts:**
- **Gaps and Islands**: Finding continuous sequences in data
- **Island**: A continuous sequence without gaps
- **Gap**: Break in the sequence
- **Techniques**:
  - Date arithmetic (date - row_number)
  - LAG with running sum
  - Both create unique identifiers for each sequence

**Why This Works (Method 1):**
```
Date       | RowNum | Date - RowNum  | Island
-----------|--------|----------------|--------
2015-01-01 | 1      | 2014-12-31     | A
2015-01-02 | 2      | 2014-12-31     | A (same)
2015-01-03 | 3      | 2014-12-31     | A (same)
2015-01-05 | 4      | 2015-01-01     | B (gap on 01-04!)
2015-01-06 | 5      | 2015-01-01     | B (same)
```

**Expected Output Pattern:**
```
airline_name         | sequence_start | sequence_end | consecutive_days | sequence_type
Southwest Airlines   | 2015-01-01     | 2015-06-30   | 180              | Long sequence
Southwest Airlines   | 2015-07-05     | 2015-12-31   | 179              | Long sequence
Delta Air Lines      | 2015-01-01     | 2015-03-15   | 73               | Long sequence
American Airlines    | 2015-02-01     | 2015-02-28   | 28               | Medium sequence
```

**Use Cases:**
- Customer activity: consecutive days with purchases
- System uptime: continuous operational periods
- Attendance: consecutive working days
- Subscriptions: continuous membership periods
- IoT: sensor data gaps

**Common Variations:**
```sql
-- Find gaps (non-operating days) instead of islands
-- Find longest sequence per airline
-- Find airlines with most operational consistency
-- Detect service interruptions
```

</details>

---

### Challenge 2: Find Flights in Top 10% of Delays Using PERCENT_RANK
**Task**: Use PERCENT_RANK() to find flights in the top 10% most delayed. Also calculate what percentile each flight falls into.

**Database**: `tutorial.flights`

**Your Solution:**
```sql
-- Write your query here
```

**Hint**:
- PERCENT_RANK() returns value between 0 and 1
- 0 = lowest value, 1 = highest value
- Top 10% means PERCENT_RANK >= 0.9
- Can partition by airline to find top 10% within each airline

**Expected Output**: Highly delayed flights with their percentile ranking

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH flight_delay_percentiles AS (
    SELECT
        airline_name,
        flight_number,
        CAST(day AS DATE) as flight_date,
        origin_airport,
        destination_airport,
        arrival_delay,
        departure_delay,
        distance,
        -- Overall percentile (across all flights)
        PERCENT_RANK() OVER (
            ORDER BY arrival_delay
        ) as overall_delay_percentile,
        -- Percentile within airline
        PERCENT_RANK() OVER (
            PARTITION BY airline_name
            ORDER BY arrival_delay
        ) as airline_delay_percentile,
        -- Percentile within route
        PERCENT_RANK() OVER (
            PARTITION BY origin_airport, destination_airport
            ORDER BY arrival_delay
        ) as route_delay_percentile,
        -- For comparison: CUME_DIST (slightly different calculation)
        CUME_DIST() OVER (
            ORDER BY arrival_delay
        ) as cumulative_distribution,
        -- NTILE for decile (1-10)
        NTILE(10) OVER (
            ORDER BY arrival_delay
        ) as delay_decile
    FROM tutorial.flights
    WHERE arrival_delay IS NOT NULL
),
top_delayed_flights AS (
    SELECT
        airline_name,
        flight_number,
        flight_date,
        origin_airport,
        destination_airport,
        arrival_delay,
        departure_delay,
        distance,
        ROUND(overall_delay_percentile, 4) as overall_pct,
        ROUND(airline_delay_percentile, 4) as airline_pct,
        ROUND(route_delay_percentile, 4) as route_pct,
        delay_decile,
        CASE
            WHEN overall_delay_percentile >= 0.99 THEN 'Top 1%'
            WHEN overall_delay_percentile >= 0.95 THEN 'Top 5%'
            WHEN overall_delay_percentile >= 0.90 THEN 'Top 10%'
            ELSE 'Below top 10%'
        END as delay_category
    FROM flight_delay_percentiles
)
SELECT
    delay_category,
    airline_name,
    flight_number,
    flight_date,
    origin_airport,
    destination_airport,
    arrival_delay,
    overall_pct,
    airline_pct,
    route_pct,
    delay_decile
FROM top_delayed_flights
WHERE overall_pct >= 0.90  -- Top 10% most delayed
ORDER BY overall_pct DESC, arrival_delay DESC
LIMIT 100;
```

**Solution 2 - Find Flights Worse Than 90th Percentile Value:**
```sql
WITH delay_percentiles AS (
    -- Calculate the 90th percentile delay value
    SELECT
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY arrival_delay) as p90_delay,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY arrival_delay) as p95_delay,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY arrival_delay) as p99_delay
    FROM tutorial.flights
    WHERE arrival_delay IS NOT NULL
),
flight_vs_percentiles AS (
    SELECT
        f.airline_name,
        f.flight_number,
        CAST(f.day AS DATE) as flight_date,
        f.origin_airport,
        f.destination_airport,
        f.arrival_delay,
        p.p90_delay,
        p.p95_delay,
        p.p99_delay,
        CASE
            WHEN f.arrival_delay >= p.p99_delay THEN 'Top 1% (99th percentile)'
            WHEN f.arrival_delay >= p.p95_delay THEN 'Top 5% (95th percentile)'
            WHEN f.arrival_delay >= p.p90_delay THEN 'Top 10% (90th percentile)'
            ELSE 'Below 90th percentile'
        END as percentile_category
    FROM tutorial.flights f
    CROSS JOIN delay_percentiles p
    WHERE f.arrival_delay IS NOT NULL
)
SELECT
    percentile_category,
    airline_name,
    flight_number,
    flight_date,
    arrival_delay,
    ROUND(p90_delay, 2) as p90_threshold
FROM flight_vs_percentiles
WHERE arrival_delay >= p90_delay
ORDER BY arrival_delay DESC
LIMIT 100;
```

**Explanation:**

**PERCENT_RANK():**
1. Returns value between 0 and 1
2. Formula: (rank - 1) / (total rows - 1)
3. 0.0 = minimum value
4. 1.0 = maximum value
5. 0.5 = median position
6. 0.9 = 90th percentile (top 10%)

**PERCENT_RANK vs CUME_DIST:**
- **PERCENT_RANK**: (rank - 1) / (total - 1)
- **CUME_DIST**: (rows with value <= current) / total rows
- CUME_DIST typically slightly higher
- Both range from 0 to 1

**Key Concepts:**
- **PERCENT_RANK()**: Relative rank as percentage
- **PERCENTILE_CONT()**: Calculate actual percentile value
- **NTILE()**: Divide into equal groups
- **CUME_DIST()**: Cumulative distribution

**Percentile Functions Comparison:**
```sql
-- Same data set, different functions:
Value | ROW_NUMBER | PERCENT_RANK | CUME_DIST | NTILE(4)
------|------------|--------------|-----------|----------
10    | 1          | 0.00         | 0.11      | 1
15    | 2          | 0.11         | 0.22      | 1
20    | 3          | 0.22         | 0.33      | 1
25    | 4          | 0.33         | 0.44      | 2
30    | 5          | 0.44         | 0.56      | 2
35    | 6          | 0.56         | 0.67      | 3
40    | 7          | 0.67         | 0.78      | 3
45    | 8          | 0.78         | 0.89      | 4
50    | 9          | 0.89         | 1.00      | 4
```

**Expected Output Pattern:**
```
delay_category | airline_name   | flight_number | flight_date | arrival_delay | overall_pct | airline_pct
Top 1%         | SkyWest        | 1234          | 2015-03-15  | 1425          | 0.9987      | 0.9952
Top 1%         | Spirit         | 5678          | 2015-06-22  | 1398          | 0.9981      | 0.9945
Top 5%         | Southwest      | 9012          | 2015-08-10  | 1267          | 0.9745      | 0.9823
Top 10%        | Delta          | 3456          | 2015-12-01  | 1103          | 0.9512      | 0.9601
Top 10%        | United         | 7890          | 2015-02-28  | 1089          | 0.9487      | 0.9534
```

**Interpretation:**
- overall_pct = 0.9987 means this flight is more delayed than 99.87% of all flights
- airline_pct = 0.9952 means most delayed in its airline
- route_pct would show how it compares to other flights on same route

**Use Cases:**
- Performance benchmarking (top 10% performers)
- Outlier detection (top/bottom 5%)
- SLA monitoring (95th percentile response time)
- Salary bands (25th, 50th, 75th percentiles)
- Quality control (identifying extreme values)

**Practical Applications:**
```sql
-- Find median delay (50th percentile)
WHERE PERCENT_RANK() BETWEEN 0.49 AND 0.51

-- Find interquartile range (25th to 75th percentile)
WHERE PERCENT_RANK() BETWEEN 0.25 AND 0.75

-- Find outliers (below 5th or above 95th percentile)
WHERE PERCENT_RANK() < 0.05 OR PERCENT_RANK() > 0.95
```

</details>

---

### Challenge 3: Compare Each Flight to Route Best/Worst Using FIRST_VALUE/LAST_VALUE
**Task**: For each flight, show the best on-time performance (smallest delay) and worst performance (largest delay) ever recorded on that route. Compare current flight to these extremes.

**Database**: `tutorial.flights`

**Your Solution:**
```sql
-- Write your query here
```

**Hint**:
- FIRST_VALUE() gets first row's value in window
- LAST_VALUE() gets last row's value in window
- PARTITION BY origin_airport, destination_airport
- ORDER BY arrival_delay ASC → FIRST_VALUE = best (smallest delay)
- ORDER BY arrival_delay DESC → FIRST_VALUE = worst (largest delay)
- Need ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING for LAST_VALUE

**Expected Output**: Each flight with route's best and worst delays for comparison

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH route_performance_context AS (
    SELECT
        airline_name,
        flight_number,
        CAST(day AS DATE) as flight_date,
        origin_airport,
        destination_airport,
        arrival_delay,
        departure_delay,
        distance,
        -- Best performance on this route (smallest delay)
        FIRST_VALUE(arrival_delay) OVER (
            PARTITION BY origin_airport, destination_airport
            ORDER BY arrival_delay ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) as route_best_delay,
        -- Date of best performance
        FIRST_VALUE(CAST(day AS DATE)) OVER (
            PARTITION BY origin_airport, destination_airport
            ORDER BY arrival_delay ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) as route_best_date,
        -- Airline with best performance
        FIRST_VALUE(airline_name) OVER (
            PARTITION BY origin_airport, destination_airport
            ORDER BY arrival_delay ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) as route_best_airline,
        -- Worst performance on this route (largest delay)
        FIRST_VALUE(arrival_delay) OVER (
            PARTITION BY origin_airport, destination_airport
            ORDER BY arrival_delay DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) as route_worst_delay,
        -- Date of worst performance
        FIRST_VALUE(CAST(day AS DATE)) OVER (
            PARTITION BY origin_airport, destination_airport
            ORDER BY arrival_delay DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) as route_worst_date,
        -- Airline with worst performance
        FIRST_VALUE(airline_name) OVER (
            PARTITION BY origin_airport, destination_airport
            ORDER BY arrival_delay DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) as route_worst_airline,
        -- Alternatively using LAST_VALUE for worst
        LAST_VALUE(arrival_delay) OVER (
            PARTITION BY origin_airport, destination_airport
            ORDER BY arrival_delay ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) as route_worst_delay_alt,
        -- Route average for reference
        ROUND(AVG(arrival_delay) OVER (
            PARTITION BY origin_airport, destination_airport
        ), 2) as route_avg_delay,
        -- Route median using PERCENTILE_CONT (if supported)
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY arrival_delay) OVER (
            PARTITION BY origin_airport, destination_airport
        ) as route_median_delay
    FROM tutorial.flights
    WHERE arrival_delay IS NOT NULL
        AND origin_airport IS NOT NULL
        AND destination_airport IS NOT NULL
),
performance_comparison AS (
    SELECT
        airline_name,
        flight_number,
        flight_date,
        origin_airport,
        destination_airport,
        arrival_delay as current_delay,
        route_best_delay,
        route_worst_delay,
        route_avg_delay,
        route_best_date,
        route_worst_date,
        route_best_airline,
        route_worst_airline,
        -- Calculate position relative to range
        ROUND(
            (arrival_delay - route_best_delay) /
            NULLIF(route_worst_delay - route_best_delay, 0) * 100,
            2
        ) as pct_of_route_range,
        -- Differences
        ROUND(arrival_delay - route_best_delay, 2) as worse_than_best,
        ROUND(route_worst_delay - arrival_delay, 2) as better_than_worst,
        ROUND(arrival_delay - route_avg_delay, 2) as vs_route_avg,
        -- Categorize performance
        CASE
            WHEN arrival_delay <= route_best_delay + (route_worst_delay - route_best_delay) * 0.25 THEN 'Excellent (Top 25%)'
            WHEN arrival_delay <= route_best_delay + (route_worst_delay - route_best_delay) * 0.50 THEN 'Good (Top 50%)'
            WHEN arrival_delay <= route_best_delay + (route_worst_delay - route_best_delay) * 0.75 THEN 'Fair (Top 75%)'
            ELSE 'Poor (Bottom 25%)'
        END as performance_category
    FROM route_performance_context
)
SELECT
    airline_name,
    flight_number,
    flight_date,
    origin_airport,
    destination_airport,
    current_delay,
    route_best_delay,
    route_worst_delay,
    route_avg_delay,
    worse_than_best,
    better_than_worst,
    vs_route_avg,
    pct_of_route_range,
    performance_category,
    route_best_airline,
    route_best_date
FROM performance_comparison
WHERE origin_airport IN ('LAX', 'JFK', 'ORD')  -- Focus on major airports
    AND destination_airport IN ('LAX', 'JFK', 'ORD', 'SFO', 'ATL')
ORDER BY origin_airport, destination_airport, current_delay DESC
LIMIT 100;
```

**Explanation:**

**FIRST_VALUE():**
1. Returns value from first row in window frame
2. Depends on ORDER BY:
   - ORDER BY value ASC → FIRST_VALUE = minimum
   - ORDER BY value DESC → FIRST_VALUE = maximum
3. Need full frame specification for consistent results

**LAST_VALUE():**
1. Returns value from last row in window frame
2. **CRITICAL**: Default frame is "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
3. Must specify full frame: ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
4. Otherwise gets current row, not actual last row!

**Key Concepts:**
- **FIRST_VALUE()**: First row in ordered window
- **LAST_VALUE()**: Last row in ordered window (need full frame!)
- **NTH_VALUE(col, n)**: Nth row in window
- Combined with ORDER BY to get min/max with context

**Why ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING:**
```sql
-- WITHOUT full frame specification:
LAST_VALUE(delay) OVER (ORDER BY delay)
-- Default frame: RANGE UNBOUNDED PRECEDING AND CURRENT ROW
-- Returns current row's value, not last!

-- WITH full frame specification:
LAST_VALUE(delay) OVER (
    ORDER BY delay
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
)
-- Returns actual last row's value
```

**Alternative Approaches:**
```sql
-- Method 1: FIRST_VALUE with different ORDER BY
FIRST_VALUE(delay) OVER (ORDER BY delay ASC)   -- minimum
FIRST_VALUE(delay) OVER (ORDER BY delay DESC)  -- maximum

-- Method 2: LAST_VALUE with proper frame
LAST_VALUE(delay) OVER (
    ORDER BY delay ASC
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
)  -- maximum

-- Method 3: Simple aggregates (no ORDER BY)
MIN(delay) OVER (PARTITION BY route)  -- minimum
MAX(delay) OVER (PARTITION BY route)  -- maximum
```

**Expected Output Pattern:**
```
airline | flight_num | flight_date | origin | dest | current_delay | route_best | route_worst | vs_best | vs_worst | performance
--------|------------|-------------|--------|------|---------------|------------|-------------|---------|----------|-------------
United  | 123        | 2015-03-15  | LAX    | JFK  | 85            | -45        | 1425        | +130    | -1340    | Fair
Delta   | 456        | 2015-03-15  | LAX    | JFK  | 12            | -45        | 1425        | +57     | -1413    | Excellent
Spirit  | 789        | 2015-03-16  | LAX    | JFK  | 1200          | -45        | 1425        | +1245   | -225     | Poor
```

**Interpretation:**
- **route_best_delay = -45**: Best flight was 45 minutes early
- **current_delay = 85**: This flight was 85 minutes late
- **worse_than_best = +130**: This flight is 130 minutes worse than best ever
- **better_than_worst = -1340**: Still 1340 minutes better than worst ever
- **pct_of_route_range**: Position in range (0% = best, 100% = worst)

**Use Cases:**
- Performance benchmarking against historical best/worst
- Goal setting (aim for top quartile)
- Outlier analysis
- Quality control limits
- Sports (compare to record holders)

**Advanced Usage - NTH_VALUE:**
```sql
-- Get 2nd best performance
NTH_VALUE(delay, 2) OVER (
    PARTITION BY route
    ORDER BY delay ASC
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
) as route_second_best

-- Get median (middle value) - if 100 rows, get 50th
NTH_VALUE(delay, 50) OVER (
    PARTITION BY route
    ORDER BY delay ASC
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
) as route_median_approx
```

**Common Pattern - Summary Window:**
```sql
SELECT
    flight_id,
    value,
    -- Get all summary stats in one query
    FIRST_VALUE(value) OVER w as min_value,
    LAST_VALUE(value) OVER w as max_value,
    AVG(value) OVER (PARTITION BY category) as avg_value,
    COUNT(*) OVER (PARTITION BY category) as total_count
FROM flights
WINDOW w AS (
    PARTITION BY category
    ORDER BY value
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
);
```

</details>

---

## 🧪 Verification Queries

Before starting exercises, verify you can access the data:

```sql
-- Check flights table
SELECT COUNT(*) as flight_count FROM tutorial.flights;

-- Check for NULLs in key columns
SELECT
    COUNT(*) as total_flights,
    COUNT(arrival_delay) as flights_with_delay,
    COUNT(origin_airport) as flights_with_origin,
    COUNT(destination_airport) as flights_with_dest
FROM tutorial.flights;

-- Check date range
SELECT
    MIN(day) as earliest_flight,
    MAX(day) as latest_flight
FROM tutorial.flights;

-- Check sample data
SELECT
    airline_name,
    flight_number,
    day,
    origin_airport,
    destination_airport,
    arrival_delay,
    distance
FROM tutorial.flights
LIMIT 10;

-- Check airlines in dataset
SELECT
    airline_name,
    COUNT(*) as flight_count
FROM tutorial.flights
GROUP BY airline_name
ORDER BY flight_count DESC;
```

---

## 🚀 Recommended Learning Path

**Session 1: Basic Window Functions (45-60 min)**
- Read window function basics
- Complete Exercises 1-3 (ROW_NUMBER, RANK, PARTITION BY)
- Understand how PARTITION BY creates separate windows

**Session 2: LAG/LEAD and Frames (45-60 min)**
- Complete Exercises 4-6 (LAG/LEAD, running totals, moving averages)
- Learn about ROWS BETWEEN frame specifications
- Practice with different window sizes

**Session 3: Advanced Functions (45-60 min)**
- Complete Exercises 7-8 (NTILE, complex frames)
- Understand ROWS vs RANGE
- Experiment with different frame specifications

**Session 4: Challenge Exercises (60-90 min)**
- Complete Challenge 1-3
- Combine multiple window functions
- Apply to real-world scenarios

**Session 5: Practice and Exploration (30-45 min)**
- Create your own exercises
- Combine window functions with CTEs
- Optimize complex queries

---

## 💡 Tips for Mode Analytics

### Understanding Window Functions
- Window functions **don't reduce rows** (unlike GROUP BY)
- They add calculated columns based on a "window" of rows
- Each row sees a different window (sliding)
- Use with PARTITION BY to create separate windows per group

### Performance Tips
- Window functions can be slow on large datasets
- Add WHERE filters before window calculations when possible
- Use LIMIT when testing
- Consider materializing intermediate results with CTEs

### Common Patterns
```sql
-- Pattern 1: Ranking within groups
ROW_NUMBER() OVER (PARTITION BY group ORDER BY value DESC)

-- Pattern 2: Running totals
SUM(value) OVER (PARTITION BY group ORDER BY date ROWS UNBOUNDED PRECEDING)

-- Pattern 3: Moving averages
AVG(value) OVER (PARTITION BY group ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)

-- Pattern 4: Comparing to previous/next
LAG(value) OVER (PARTITION BY group ORDER BY date)
LEAD(value) OVER (PARTITION BY group ORDER BY date)

-- Pattern 5: Percentiles
PERCENT_RANK() OVER (ORDER BY value)
NTILE(4) OVER (ORDER BY value)

-- Pattern 6: First/Last in group
FIRST_VALUE(value) OVER (PARTITION BY group ORDER BY date)
LAST_VALUE(value) OVER (PARTITION BY group ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
```

### Running Queries in Mode
1. **Run**: Ctrl+Enter or click "Run Query"
2. **Stop**: Click "Stop" if query takes too long
3. **Format**: Click "Format" to auto-indent SQL
4. **Schema**: Left sidebar shows tables and columns
5. **Results**: Bottom pane shows query output
6. **Export**: Download results as CSV

---

## 🐛 Common Issues and Solutions

### Issue 1: "Column used in window function not in GROUP BY"
**Problem**: Mixing GROUP BY with window functions incorrectly
```sql
-- WRONG:
SELECT airline, flight_number,
       COUNT(*),
       ROW_NUMBER() OVER (ORDER BY COUNT(*))
FROM flights
GROUP BY airline;  -- flight_number not in GROUP BY!

-- RIGHT:
WITH counts AS (
    SELECT airline, COUNT(*) as cnt
    FROM flights
    GROUP BY airline
)
SELECT airline, cnt,
       ROW_NUMBER() OVER (ORDER BY cnt)
FROM counts;
```

### Issue 2: LAST_VALUE returns current row
**Problem**: Not using full frame specification
```sql
-- WRONG:
LAST_VALUE(delay) OVER (ORDER BY delay)

-- RIGHT:
LAST_VALUE(delay) OVER (
    ORDER BY delay
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
)
```

### Issue 3: Moving average has wrong window size
**Problem**: Confusing "N days" with "N PRECEDING"
```sql
-- For 7-day moving average (7 days including today):
-- RIGHT:
ROWS BETWEEN 6 PRECEDING AND CURRENT ROW  -- 6 + 1 = 7 days

-- WRONG:
ROWS BETWEEN 7 PRECEDING AND CURRENT ROW  -- 7 + 1 = 8 days!
```

### Issue 4: NULL values affect window functions
**Solution**: Filter NULLs or use COALESCE
```sql
-- Filter in WHERE clause
WHERE arrival_delay IS NOT NULL

-- Or use COALESCE in calculation
AVG(COALESCE(arrival_delay, 0)) OVER (...)
```

### Issue 5: Performance issues with large windows
**Solution**:
- Add WHERE filters before window calculations
- Use indexes on PARTITION BY and ORDER BY columns
- Consider pre-aggregating data
- Use LIMIT when testing

---

## 📚 Window Function Reference

### Ranking Functions
```sql
ROW_NUMBER() OVER (...)          -- 1, 2, 3, 4 (unique)
RANK() OVER (...)                -- 1, 2, 2, 4 (gaps after ties)
DENSE_RANK() OVER (...)          -- 1, 2, 2, 3 (no gaps)
PERCENT_RANK() OVER (...)        -- 0.0 to 1.0 (relative rank)
NTILE(n) OVER (...)              -- Divide into n buckets
CUME_DIST() OVER (...)           -- Cumulative distribution
```

### Offset Functions
```sql
LAG(col, offset) OVER (...)      -- Previous row's value
LEAD(col, offset) OVER (...)     -- Next row's value
FIRST_VALUE(col) OVER (...)      -- First row in window
LAST_VALUE(col) OVER (...)       -- Last row in window
NTH_VALUE(col, n) OVER (...)     -- Nth row in window
```

### Aggregate Functions as Window Functions
```sql
SUM(col) OVER (...)              -- Running/windowed sum
AVG(col) OVER (...)              -- Running/windowed average
COUNT(col) OVER (...)            -- Running/windowed count
MIN(col) OVER (...)              -- Minimum in window
MAX(col) OVER (...)              -- Maximum in window
STDDEV(col) OVER (...)           -- Standard deviation in window
```

### Frame Specifications
```sql
-- No frame (entire partition)
OVER (PARTITION BY col)

-- Default frame with ORDER BY (start to current)
OVER (PARTITION BY col ORDER BY date)

-- Explicit frames
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW        -- Start to current
ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING -- Entire partition
ROWS BETWEEN 6 PRECEDING AND CURRENT ROW                -- 7-row window
ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING                -- Centered 7-row window
ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING        -- Current to end

-- RANGE vs ROWS
RANGE BETWEEN ...  -- Based on value ranges (logical)
ROWS BETWEEN ...   -- Based on row positions (physical)
```

---

## 🎯 Key Concepts Summary

### 1. PARTITION BY
- Divides data into separate windows
- Like GROUP BY but doesn't reduce rows
- Each partition processed independently

### 2. ORDER BY
- Determines sequence within partition
- Required for: ROW_NUMBER, RANK, LAG, LEAD, running totals
- Optional for: SUM, AVG, MIN, MAX (without ORDER BY = entire partition)

### 3. Frame Specification
- Defines which rows are included in calculation
- Default depends on whether ORDER BY present
- ROWS: Physical row count
- RANGE: Logical value range

### 4. Use Cases by Function Type

**Ranking (ROW_NUMBER, RANK, DENSE_RANK):**
- Top N per group
- Pagination
- Eliminating duplicates
- Assigning priorities

**Offset (LAG, LEAD):**
- Time series analysis
- Comparing consecutive values
- Calculating changes/differences
- Detecting trends

**Running Totals (SUM, AVG with ORDER BY):**
- Cumulative metrics
- Running balances
- Progressive aggregations
- Growth tracking

**Moving Windows (ROWS BETWEEN n PRECEDING):**
- Moving averages
- Smoothing data
- Recent trends
- Local context

**Distribution (PERCENT_RANK, NTILE, CUME_DIST):**
- Percentile analysis
- Performance segmentation
- Outlier detection
- Distribution analysis

---

## 🔗 Additional Resources

### Mode SQL Documentation
- [Mode SQL Tutorial](https://mode.com/sql-tutorial/)
- [Window Functions Guide](https://mode.com/sql-tutorial/sql-window-functions/)
- [Advanced Window Functions](https://mode.com/sql-tutorial/sql-window-functions-advanced/)

### Practice Datasets
- `tutorial.flights` - Airline flight data
- `tutorial.us_flights` - US flight data (alternative)
- Check Mode's public datasets for more practice

### SQL Standards
- Window functions part of SQL:2003 standard
- Syntax may vary slightly between databases (Postgres, MySQL, SQL Server, Oracle)
- Mode uses PostgreSQL syntax

---

## 🎓 Learning Outcomes

After completing these exercises, you should be able to:

✅ Use ROW_NUMBER(), RANK(), and DENSE_RANK() effectively
✅ Understand when to use PARTITION BY
✅ Calculate running totals and moving averages
✅ Compare rows using LAG() and LEAD()
✅ Work with frame specifications (ROWS BETWEEN)
✅ Perform percentile analysis with PERCENT_RANK() and NTILE()
✅ Use FIRST_VALUE() and LAST_VALUE() correctly
✅ Solve "gaps and islands" problems
✅ Combine window functions with CTEs
✅ Apply window functions to real-world business problems

---

Good luck with your window functions journey! Start with Exercise 1 and progressively build your skills. Window functions are one of the most powerful features in SQL for analytical queries. 🛫📊

---

**Pro Tip**: Keep a cheat sheet of common window function patterns handy while practicing. They'll become second nature with practice!
