# SQL CTE Exercises - Flight Data (Mode Public Warehouse)

These exercises use Mode's real flight datasets: `tutorial.us_flights` and `tutorial.flights`

💡 **Solutions included - Click to expand under each exercise!**

---

## 📊 Available Data

### `tutorial.us_flights` Table
- `flight_date` - Date of flight
- `unique_carrier` - Airline code
- `flight_num` - Flight number
- `origin` - Origin airport code
- `dest` - Destination airport code
- `arr_delay` - Arrival delay in minutes
- `cancelled` - Whether flight was cancelled (0/1)
- `distance` - Flight distance
- `carrier_delay` - Delay caused by carrier
- `weather_delay` - Delay caused by weather
- `late_aircraft_delay` - Delay caused by late aircraft
- `nas_delay` - NAS (air traffic system) delay
- `security_delay` - Delay caused by security
- `actual_elapsed_time` - Actual flight duration

### `tutorial.flights` Table
- Similar fields plus additional details
- `airline_name` - Full airline name
- `day_of_week` - Day of week
- `origin_city`, `origin_state` - Departure city/state
- `destination_city`, `destination_state` - Arrival city/state
- `was_cancelled` - Boolean cancellation flag

---

## ✅ Progress Tracker

- [ ] Exercise 1: Flights Above Average Delay
- [ ] Exercise 2: Cancellation Rate by Airline
- [ ] Exercise 3: Routes by Flight Count
- [ ] Exercise 4: Top 5 Airlines by Delay
- [ ] Exercise 5: Multiple Delay Factors
- [ ] Exercise 6: Delay by Day of Week
- [ ] Exercise 7: Number Sequence (Recursive)
- [ ] Exercise 8: Distance Ranges (Window Functions)
- [ ] Challenge 1: Complete Delay Analysis
- [ ] Challenge 2: Route Profitability
- [ ] Challenge 3: Airline Performance Dashboard

---

## 🎯 Easy Exercises

### Exercise 1: Flights Above Average Delay
**Task**: Find all flights with arrival delay greater than the average delay.

**Database**: `tutorial.us_flights`

**Your Solution:**
```sql
-- Write your CTE here
```

**Hint**:
- Calculate average arrival delay
- Filter flights above that average
- Show flight details and delay amount

**Expected Output**: Should show flights with above-average delays

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH avg_delay AS (
    SELECT AVG(arr_delay) as average_delay FROM tutorial.us_flights
)
SELECT
    flight_date,
    unique_carrier,
    flight_num,
    origin,
    dest,
    arr_delay
FROM tutorial.us_flights
CROSS JOIN avg_delay
WHERE arr_delay > avg_delay.average_delay
ORDER BY arr_delay DESC;
```

**Explanation:**
1. `avg_delay` CTE calculates average arrival delay across all flights
2. CROSS JOIN allows us to access the single average value
3. WHERE filters for flights with above-average delays
4. Results ordered by delay (highest first)

**Key Concepts:**
- Single CTE for scalar aggregation
- CROSS JOIN for scalar values
- Comparing values to aggregate

</details>

---

### Exercise 2: Cancellation Rate by Airline
**Task**: Calculate total flights and cancellation count for each airline, then show only airlines with >5% cancellation rate.

**Database**: `tutorial.us_flights`

**Your Solution:**
```sql
-- Write your CTE here
```

**Hint**:
- Group by `unique_carrier`
- Calculate COUNT(*) and SUM(cancelled)
- Calculate cancellation percentage
- Filter for > 5%

**Expected Output Pattern**:
```
unique_carrier | total_flights | cancelled_flights | cancellation_rate
```

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH airline_stats AS (
    SELECT
        unique_carrier,
        COUNT(*) as total_flights,
        SUM(cancelled) as cancelled_flights,
        ROUND(100.0 * SUM(cancelled) / COUNT(*), 2) as cancellation_rate
    FROM tutorial.us_flights
    GROUP BY unique_carrier
)
SELECT
    unique_carrier,
    total_flights,
    cancelled_flights,
    cancellation_rate
FROM airline_stats
WHERE cancellation_rate > 5
ORDER BY cancellation_rate DESC;
```

**Explanation:**
1. `airline_stats` CTE groups flights by airline
2. COUNT(*) counts total flights
3. SUM(cancelled) counts cancelled flights (0/1 values)
4. Calculate percentage: (cancelled / total) * 100
5. HAVING or WHERE filters for >5%
6. ROUND to 2 decimal places

**Key Concepts:**
- GROUP BY for grouping data
- SUM for counting boolean flags
- Percentage calculation: (part / whole) * 100
- Filtering aggregated data

**Expected Output Pattern:**
```
unique_carrier | total_flights | cancelled_flights | cancellation_rate
AA             | 10000         | 600               | 6.00
DL             | 9500          | 475               | 5.00
```

</details>

---

### Exercise 3: Routes by Flight Count
**Task**: Show each route (origin → destination) with the number of flights, ordered by frequency.

**Database**: `tutorial.us_flights`

**Your Solution:**
```sql
-- Write your CTE here
```

**Hint**:
- Group by origin and dest
- Count flights per route
- Order by count descending

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH route_statistics AS (
    SELECT
        origin,
        dest,
        COUNT(*) as flight_count,
        ROUND(AVG(distance), 0) as avg_distance,
        ROUND(AVG(arr_delay), 2) as avg_delay
    FROM tutorial.us_flights
    GROUP BY origin, dest
)
SELECT
    CONCAT(origin, ' → ', dest) as route,
    flight_count,
    avg_distance,
    avg_delay
FROM route_statistics
ORDER BY flight_count DESC;
```

**Explanation:**
1. Group by origin and destination
2. COUNT(*) shows flight frequency per route
3. Include additional metrics (distance, delay)
4. CONCAT creates readable route display
5. Order by frequency descending

**Key Concepts:**
- GROUP BY multiple columns
- Multiple aggregations at once
- STRING concatenation
- Sorting by aggregated values

</details>

---

## 📊 Medium Exercises

### Exercise 4: Top 5 Airlines by Delay
**Task**: Find the top 5 airlines by average arrival delay using two CTEs:
1. `airline_delays` - Average delay per airline
2. `ranked_airlines` - Rank airlines by average delay

**Database**: `tutorial.flights`

**Your Solution:**
```sql
WITH airline_delays AS (
    -- Your query here
),
ranked_airlines AS (
    -- Your query here
)
SELECT * FROM ranked_airlines;
```

**Hint**: Use RANK() or ROW_NUMBER() window function

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH airline_delays AS (
    SELECT
        airline_name,
        COUNT(*) as flight_count,
        ROUND(AVG(arrival_delay), 2) as avg_arrival_delay,
        ROUND(AVG(departure_delay), 2) as avg_departure_delay,
        ROUND(AVG(COALESCE(carrier_delay, 0)), 2) as avg_carrier_delay
    FROM tutorial.flights
    GROUP BY airline_name
),
ranked_airlines AS (
    SELECT
        airline_name,
        flight_count,
        avg_arrival_delay,
        avg_departure_delay,
        avg_carrier_delay,
        RANK() OVER (ORDER BY avg_arrival_delay DESC) as delay_rank
    FROM airline_delays
)
SELECT
    delay_rank,
    airline_name,
    flight_count,
    avg_arrival_delay,
    avg_departure_delay,
    avg_carrier_delay
FROM ranked_airlines
WHERE delay_rank <= 5
ORDER BY delay_rank;
```

**Explanation:**
1. `airline_delays` aggregates by airline, calculates averages
2. COALESCE handles NULL values
3. `ranked_airlines` uses RANK() window function to rank airlines
4. Main query filters for top 5
5. RANK() assigns same rank to ties

**Key Concepts:**
- Two CTEs working together
- Window function RANK() OVER
- ORDER BY in window function
- COALESCE for NULL handling
- Filtering ranked results

**Expected Output:**
```
delay_rank | airline_name | avg_arrival_delay
1          | SkyWest      | 28.45
2          | Spirit       | 26.82
3          | Southwest    | 24.10
4          | Alaska       | 23.95
5          | Frontier     | 22.50
```

</details>

---

### Exercise 5: Flights with Multiple Delay Factors
**Task**: Show flights that had delays from MORE THAN ONE cause (carrier, weather, late aircraft, NAS, security).

**Database**: `tutorial.us_flights`

**Your Solution:**
```sql
WITH delay_analysis AS (
    SELECT
        flight_date,
        unique_carrier,
        flight_num,
        -- Count how many delay factors exist (are not NULL/0)
        -- Your logic here
    FROM tutorial.us_flights
),
multi_factor_delays AS (
    -- Filter for delays with >1 factor
)
SELECT * FROM multi_factor_delays;
```

**Hint**:
- Count non-NULL delay columns
- Or sum them: IF(carrier_delay > 0, 1, 0) + IF(weather_delay > 0, 1, 0) + ...
- Filter for count > 1

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH delay_analysis AS (
    SELECT
        flight_date,
        unique_carrier,
        flight_num,
        origin,
        dest,
        arr_delay,
        carrier_delay,
        weather_delay,
        late_aircraft_delay,
        nas_delay,
        security_delay,
        -- Count how many delay factors contributed
        (
            (CASE WHEN carrier_delay > 0 THEN 1 ELSE 0 END) +
            (CASE WHEN weather_delay > 0 THEN 1 ELSE 0 END) +
            (CASE WHEN late_aircraft_delay > 0 THEN 1 ELSE 0 END) +
            (CASE WHEN nas_delay > 0 THEN 1 ELSE 0 END) +
            (CASE WHEN security_delay > 0 THEN 1 ELSE 0 END)
        ) as delay_factor_count
    FROM tutorial.us_flights
),
multi_factor_delays AS (
    SELECT
        flight_date,
        unique_carrier,
        flight_num,
        origin,
        dest,
        arr_delay,
        delay_factor_count
    FROM delay_analysis
    WHERE delay_factor_count > 1
)
SELECT
    *
FROM multi_factor_delays
ORDER BY arr_delay DESC;
```

**Explanation:**
1. `delay_analysis` creates a new column counting delay factors
2. Use CASE statements to count non-zero delays
3. Add all counts together to get total factors
4. `multi_factor_delays` filters for flights with >1 factor
5. Results show complex delay scenarios

**Key Concepts:**
- CASE statements for conditional counting
- Multiple conditions in one SELECT
- Creating computed columns in CTE
- Filtering on computed values

**What This Shows:**
Flights where delays came from multiple sources (e.g., weather AND carrier delay)

</details>

---

### Exercise 6: Delay Breakdown by Day of Week
**Task**: Show average delay by day of week, sorted logically (Monday → Sunday).

**Database**: `tutorial.flights`

**Your Solution:**
```sql
WITH daily_delays AS (
    SELECT
        day_of_week,
        -- Calculate average arrival_delay
        -- Count flights
        -- Your aggregations here
    FROM tutorial.flights
    GROUP BY day_of_week
)
SELECT * FROM daily_delays
ORDER BY CASE day_of_week
    WHEN 'Monday' THEN 1
    WHEN 'Tuesday' THEN 2
    -- ... etc
END;
```

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH daily_delays AS (
    SELECT
        day_of_week,
        COUNT(*) as flight_count,
        ROUND(AVG(arrival_delay), 2) as avg_arrival_delay,
        ROUND(AVG(departure_delay), 2) as avg_departure_delay,
        ROUND(MAX(arrival_delay), 2) as max_arrival_delay,
        ROUND(MIN(arrival_delay), 2) as min_arrival_delay
    FROM tutorial.flights
    WHERE arrival_delay IS NOT NULL
    GROUP BY day_of_week
)
SELECT
    day_of_week,
    flight_count,
    avg_arrival_delay,
    avg_departure_delay,
    max_arrival_delay,
    min_arrival_delay
FROM daily_delays
ORDER BY
    CASE day_of_week
        WHEN 'Monday' THEN 1
        WHEN 'Tuesday' THEN 2
        WHEN 'Wednesday' THEN 3
        WHEN 'Thursday' THEN 4
        WHEN 'Friday' THEN 5
        WHEN 'Saturday' THEN 6
        WHEN 'Sunday' THEN 7
    END;
```

**Explanation:**
1. Group flights by `day_of_week`
2. Calculate multiple delay statistics
3. Filter out NULL values
4. ORDER BY CASE statement to sort logically (Mon-Sun)
5. Shows which days have worse delays

**Key Concepts:**
- GROUP BY with non-numeric column
- Multiple aggregations (AVG, MAX, MIN)
- CASE for custom sorting
- NULL filtering

**Expected Output Pattern:**
```
day_of_week | flight_count | avg_arrival_delay | avg_departure_delay
Monday      | 3000         | 12.45             | 8.23
Tuesday     | 2950         | 10.32             | 6.54
...
Friday      | 3200         | 18.90             | 14.32
```

</details>

---

## 🔄 Hard Exercises (Recursive CTEs)

### Exercise 7: Simple Number Sequence
**Task**: Generate numbers 1 to 20 using a RECURSIVE CTE.

**Your Solution:**
```sql
WITH RECURSIVE numbers AS (
    -- Base case: start at 1
    SELECT 1 as n

    UNION ALL

    -- Recursive case: increment
    SELECT n + 1
    FROM numbers
    WHERE n < 20
)
SELECT * FROM numbers;
```

**Expected Output**: 1, 2, 3, ..., 20

**Key Concepts**:
- RECURSIVE keyword
- Base case (starting point)
- UNION ALL
- Recursive reference with WHERE termination

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH RECURSIVE numbers AS (
    -- Base case: start at 1
    SELECT 1 as n

    UNION ALL

    -- Recursive case: increment by 1
    SELECT n + 1
    FROM numbers
    WHERE n < 20
)
SELECT * FROM numbers
ORDER BY n;
```

**Explanation:**
1. **Base case**: `SELECT 1` - Starting point
2. **UNION ALL**: Combines base case with recursive results
3. **Recursive case**: `SELECT n + 1 FROM numbers` - References itself
4. **WHERE n < 20**: Termination condition (CRITICAL!)
5. Results: 1, 2, 3, ..., 20

**Key Concepts:**
- RECURSIVE keyword required
- Base case must not reference itself
- UNION ALL required
- Recursive case references the CTE
- WHERE clause for stopping condition

**Output:**
```
n
--
1
2
3
...
20
```

</details>

---

### Exercise 8: Distance Ranges with Row Numbers
**Task**: Rank flights by distance using a window function (not technically recursive, but uses advanced features).

**Database**: `tutorial.us_flights`

**Your Solution:**
```sql
WITH flight_distances AS (
    SELECT
        origin,
        dest,
        distance,
        -- Use ROW_NUMBER() to number flights by distance
        ROW_NUMBER() OVER (ORDER BY distance DESC) as distance_rank
    FROM tutorial.us_flights
)
SELECT * FROM flight_distances
WHERE distance_rank <= 10;
```

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH flight_distances AS (
    SELECT
        origin,
        dest,
        distance,
        unique_carrier,
        flight_date,
        ROW_NUMBER() OVER (ORDER BY distance DESC) as distance_rank,
        NTILE(4) OVER (ORDER BY distance DESC) as distance_quartile
    FROM tutorial.us_flights
    WHERE distance IS NOT NULL
)
SELECT
    distance_rank,
    distance_quartile,
    origin,
    dest,
    distance,
    unique_carrier
FROM flight_distances
WHERE distance_rank <= 10
ORDER BY distance_rank;
```

**Explanation:**
1. `flight_distances` adds window functions to each row
2. ROW_NUMBER() assigns sequential number ordered by distance
3. NTILE(4) divides flights into 4 quartiles by distance
4. Main query filters for top 10 longest flights
5. Shows ranking and distance category

**Key Concepts:**
- ROW_NUMBER() for sequential numbering
- NTILE() for distributing into buckets
- ORDER BY in window function
- Window functions work row-by-row (no GROUP BY needed)

**Expected Output Pattern:**
```
distance_rank | distance_quartile | origin | dest | distance
1             | 1                 | HNL    | JFK  | 5280
2             | 1                 | LAX    | JFK  | 5062
3             | 1                 | SFO    | JFK  | 4965
...
10            | 1                 | PDX    | JFK  | 4480
```

</details>

---

## 🏆 Challenge Exercises

### Challenge 1: Complete Delay Analysis
**Task**: Create a comprehensive delay report with THREE CTEs:
1. `delay_statistics` - Average delays by airline
2. `delay_categorization` - Categorize airlines as "High Delay" (>15 min), "Medium" (5-15 min), "Low" (<5 min)
3. `summary` - Count airlines in each category

**Database**: `tutorial.flights`

**Your Solution:**
```sql
WITH delay_statistics AS (
    -- Your query here
),
delay_categorization AS (
    -- Your query here
),
summary AS (
    -- Your query here
)
SELECT * FROM summary;
```

**Expected Output**:
```
delay_category | airline_count | avg_delay
```

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH delay_statistics AS (
    SELECT
        airline_name,
        COUNT(*) as flight_count,
        ROUND(AVG(arrival_delay), 2) as avg_arrival_delay,
        ROUND(AVG(departure_delay), 2) as avg_departure_delay,
        ROUND(AVG(COALESCE(carrier_delay, 0)), 2) as avg_carrier_delay,
        ROUND(AVG(COALESCE(weather_delay, 0)), 2) as avg_weather_delay,
        ROUND(AVG(COALESCE(late_aircraft_delay, 0)), 2) as avg_late_aircraft_delay
    FROM tutorial.flights
    GROUP BY airline_name
),
delay_categorization AS (
    SELECT
        airline_name,
        flight_count,
        avg_arrival_delay,
        avg_departure_delay,
        avg_carrier_delay,
        avg_weather_delay,
        avg_late_aircraft_delay,
        CASE
            WHEN avg_arrival_delay > 15 THEN 'High Delay'
            WHEN avg_arrival_delay BETWEEN 5 AND 15 THEN 'Medium Delay'
            ELSE 'Low Delay'
        END as delay_category
    FROM delay_statistics
),
summary AS (
    SELECT
        delay_category,
        COUNT(*) as airline_count,
        ROUND(AVG(avg_arrival_delay), 2) as avg_delay_in_category,
        COUNT(DISTINCT airline_name) as unique_airlines
    FROM delay_categorization
    GROUP BY delay_category
)
SELECT
    delay_category,
    airline_count,
    avg_delay_in_category,
    unique_airlines
FROM summary
ORDER BY
    CASE delay_category
        WHEN 'High Delay' THEN 1
        WHEN 'Medium Delay' THEN 2
        WHEN 'Low Delay' THEN 3
    END;
```

**Explanation:**
1. `delay_statistics` - Aggregate all delay metrics by airline
2. `delay_categorization` - Use CASE to classify airlines by delay severity
3. `summary` - Count airlines in each category with category-level statistics
4. Main query orders logically

**Key Concepts:**
- Three CTEs in sequence
- CASE statement for classification
- Multiple levels of aggregation
- CTEs referencing previous CTEs

**Expected Output:**
```
delay_category | airline_count | avg_delay_in_category
High Delay     | 4             | 22.50
Medium Delay   | 8             | 10.20
Low Delay      | 3             | 2.15
```

</details>

---

### Challenge 2: Route Profitability Analysis (Conceptual)
**Task**: For each route, calculate:
1. Number of flights
2. Cancellation rate
3. Average delay
4. "Health Score" (fewer cancellations and delays = higher score)

**Database**: `tutorial.us_flights`

**Your Solution:**
```sql
WITH route_stats AS (
    SELECT
        origin,
        dest,
        COUNT(*) as total_flights,
        SUM(cancelled) as cancelled_flights,
        ROUND(100.0 * SUM(cancelled) / COUNT(*), 2) as cancellation_rate,
        ROUND(AVG(arr_delay), 2) as avg_arrival_delay,
        -- Create a health score
        -- Your formula: (100 - cancellation_rate) - (avg_delay / 10) or similar
    FROM tutorial.us_flights
    GROUP BY origin, dest
)
SELECT * FROM route_stats
ORDER BY health_score DESC
LIMIT 10;
```

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH route_stats AS (
    SELECT
        origin,
        dest,
        CONCAT(origin, ' → ', dest) as route,
        COUNT(*) as total_flights,
        SUM(CASE WHEN cancelled = 1 THEN 1 ELSE 0 END) as cancelled_flights,
        ROUND(100.0 * SUM(CASE WHEN cancelled = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as cancellation_rate,
        ROUND(AVG(arr_delay), 2) as avg_arrival_delay,
        ROUND(AVG(distance), 0) as avg_distance,
        -- Health Score: higher = better (fewer cancellations, shorter delays)
        ROUND((100 - ROUND(100.0 * SUM(CASE WHEN cancelled = 1 THEN 1 ELSE 0 END) / COUNT(*), 2)) -
              (ROUND(AVG(arr_delay), 2) / 10), 2) as health_score
    FROM tutorial.us_flights
    GROUP BY origin, dest
)
SELECT
    route,
    total_flights,
    cancellation_rate,
    avg_arrival_delay,
    avg_distance,
    health_score
FROM route_stats
ORDER BY health_score DESC
LIMIT 20;
```

**Explanation:**
1. Group flights by route (origin-dest pair)
2. Calculate cancellation count and percentage
3. Calculate average arrival delay
4. Create "health score" formula combining both metrics
5. Results show best-performing routes

**Key Concepts:**
- Multiple aggregations for different metrics
- CASE for conditional counting
- Formula-based scoring
- Ordering by computed score

**Health Score Formula:**
- Start with 100 (perfect score)
- Subtract cancellation rate
- Subtract (average delay / 10)
- High score = reliable, on-time route

**Expected Output:**
```
route        | total_flights | cancellation_rate | avg_arrival_delay | health_score
LAX → SFO    | 500           | 2.00              | 5.50              | 92.45
ORD → BOS    | 450           | 1.50              | 6.20              | 92.23
```

</details>

---

### Challenge 3: Airline Performance Dashboard
**Task**: Create a multi-metric performance report with:
1. `airline_flights` - Flight counts by airline
2. `airline_delays` - Delay metrics by airline
3. `airline_cancellations` - Cancellation metrics by airline
4. `airline_performance` - Combine all into one view

**Database**: `tutorial.flights`

**Your Solution:**
```sql
WITH airline_flights AS (
    SELECT
        airline_name,
        COUNT(*) as total_flights,
        COUNT(DISTINCT origin_airport) as origin_airports,
        COUNT(DISTINCT destination_airport) as dest_airports
    FROM tutorial.flights
    GROUP BY airline_name
),
airline_delays AS (
    SELECT
        airline_name,
        ROUND(AVG(arrival_delay), 2) as avg_arrival_delay,
        ROUND(AVG(departure_delay), 2) as avg_departure_delay,
        ROUND(AVG(carrier_delay), 2) as avg_carrier_delay,
        ROUND(AVG(weather_delay), 2) as avg_weather_delay
    FROM tutorial.flights
    GROUP BY airline_name
),
airline_cancellations AS (
    SELECT
        airline_name,
        SUM(CASE WHEN was_cancelled THEN 1 ELSE 0 END) as cancelled_count,
        ROUND(100.0 * SUM(CASE WHEN was_cancelled THEN 1 ELSE 0 END) / COUNT(*), 2) as cancellation_rate
    FROM tutorial.flights
    GROUP BY airline_name
),
airline_performance AS (
    SELECT
        f.airline_name,
        f.total_flights,
        f.origin_airports,
        f.dest_airports,
        d.avg_arrival_delay,
        d.avg_departure_delay,
        c.cancelled_count,
        c.cancellation_rate,
        -- Create a composite score if desired
        ROUND((100 - c.cancellation_rate) - (d.avg_arrival_delay / 10), 2) as performance_score
    FROM airline_flights f
    JOIN airline_delays d ON f.airline_name = d.airline_name
    JOIN airline_cancellations c ON f.airline_name = c.airline_name
)
SELECT * FROM airline_performance
ORDER BY performance_score DESC;
```

<details>
<summary><b>💡 Click to see solution</b></summary>

**Solution:**
```sql
WITH airline_flights AS (
    SELECT
        airline_name,
        COUNT(*) as total_flights,
        COUNT(DISTINCT origin_airport) as origin_airports,
        COUNT(DISTINCT destination_airport) as dest_airports,
        COUNT(DISTINCT CAST(day AS DATE)) as days_operating
    FROM tutorial.flights
    GROUP BY airline_name
),
airline_delays AS (
    SELECT
        airline_name,
        ROUND(AVG(arrival_delay), 2) as avg_arrival_delay,
        ROUND(AVG(departure_delay), 2) as avg_departure_delay,
        ROUND(AVG(COALESCE(carrier_delay, 0)), 2) as avg_carrier_delay,
        ROUND(AVG(COALESCE(weather_delay, 0)), 2) as avg_weather_delay,
        ROUND(AVG(COALESCE(late_aircraft_delay, 0)), 2) as avg_late_aircraft_delay,
        ROUND(MAX(arrival_delay), 2) as max_arrival_delay
    FROM tutorial.flights
    GROUP BY airline_name
),
airline_cancellations AS (
    SELECT
        airline_name,
        SUM(CASE WHEN was_cancelled THEN 1 ELSE 0 END) as cancelled_count,
        ROUND(100.0 * SUM(CASE WHEN was_cancelled THEN 1 ELSE 0 END) / COUNT(*), 2) as cancellation_rate
    FROM tutorial.flights
    GROUP BY airline_name
),
airline_performance AS (
    SELECT
        f.airline_name,
        f.total_flights,
        f.origin_airports,
        f.dest_airports,
        f.days_operating,
        d.avg_arrival_delay,
        d.avg_departure_delay,
        d.avg_carrier_delay,
        d.avg_weather_delay,
        c.cancelled_count,
        c.cancellation_rate,
        -- Performance Score: 0-100
        -- Higher is better
        ROUND(
            100 -
            ((d.avg_arrival_delay / 30) * 50) -  -- 50% weight to arrival delay
            ((c.cancellation_rate / 5) * 50),     -- 50% weight to cancellation rate
            2
        ) as performance_score
    FROM airline_flights f
    JOIN airline_delays d ON f.airline_name = d.airline_name
    JOIN airline_cancellations c ON f.airline_name = c.airline_name
)
SELECT
    airline_name,
    total_flights,
    origin_airports,
    dest_airports,
    avg_arrival_delay,
    avg_departure_delay,
    cancellation_rate,
    performance_score
FROM airline_performance
ORDER BY performance_score DESC;
```

**Explanation:**
1. `airline_flights` - Basic flight statistics per airline
2. `airline_delays` - Multiple delay metrics per airline
3. `airline_cancellations` - Cancellation statistics
4. `airline_performance` - Combines all three CTEs with composite scoring
5. Main query presents comprehensive dashboard

**Key Concepts:**
- Four CTEs in sequence
- Multiple JOINs in final CTE
- Composite scoring formula
- Weighting different metrics

**Performance Score Calculation:**
- Base: 100 (perfect)
- Subtract: (arrival_delay / 30) * 50% weight
- Subtract: (cancellation_rate / 5) * 50% weight
- Result: 0-100 score

**Expected Output:**
```
airline_name | total_flights | avg_arrival_delay | cancellation_rate | performance_score
Southwest    | 15000         | 8.45              | 1.20              | 85.60
Delta        | 14500         | 9.20              | 1.50              | 84.95
United       | 13200         | 11.30             | 2.10              | 82.30
```

</details>

---

## 🧪 Verification Queries

Before starting exercises, verify you can access the data:

```sql
-- Check us_flights table
SELECT COUNT(*) as flight_count FROM tutorial.us_flights;

-- Check first few records
SELECT * FROM tutorial.us_flights LIMIT 5;

-- Check flights table
SELECT COUNT(*) as flight_count FROM tutorial.flights;

-- Check first few records
SELECT * FROM tutorial.flights LIMIT 5;
```

---

## 🚀 Recommended Learning Path

**Session 1 (30 min):**
- Run verification queries
- Complete Exercises 1-3 (basic CTEs with aggregations)

**Session 2 (30 min):**
- Complete Exercises 4-6 (multiple CTEs, window functions)

**Session 3 (30 min):**
- Complete Exercise 7 (recursive CTE concept)
- Attempt Exercise 8 (advanced window functions)

**Session 4 (30 min):**
- Complete Challenge exercises
- Experiment with variations

---

## 💡 Tips for Mode

### Finding Tables and Columns
1. Click **"Schema"** on the left sidebar
2. Browse available tables
3. Click a table to see columns and data types

### Running Queries
- **Run**: Ctrl+Enter or click "Run Query"
- **Stop**: Click "Stop" if query takes too long
- **Format**: Click "Format" to auto-indent

### Working with Large Datasets
- Use `LIMIT` when testing: `SELECT * FROM tutorial.us_flights LIMIT 10;`
- Aggregate before filtering if possible
- Use column aliases for readability

### Window Functions Help
```sql
-- Basic ranking
ROW_NUMBER() OVER (ORDER BY column) as row_num    -- 1,2,3,4...
RANK() OVER (ORDER BY column) as rank              -- 1,2,2,4...
DENSE_RANK() OVER (ORDER BY column) as dense_rank -- 1,2,2,3...

-- With partitions
ROW_NUMBER() OVER (PARTITION BY airline ORDER BY delay DESC) as rank_in_airline
```

---

## 🐛 Common Issues

### "Table not found"
- Verify schema is correct: `tutorial.us_flights` not `tutorial_us_flights`
- Check you're in the right database/warehouse
- Try: `SELECT * FROM information_schema.tables WHERE table_name LIKE '%flight%'`

### CTE Syntax Errors
- Check comma between CTEs: `cte1 AS (...), cte2 AS (...)`
- Verify all CTEs are referenced in final query
- Check parentheses are balanced

### NULL/0 Confusion
- Delay columns might be NULL or 0
- Use `COALESCE(column, 0)` if needed
- Check with: `SELECT * FROM tutorial.us_flights WHERE carrier_delay IS NULL LIMIT 5;`

---

## 🧠 Learning Points by Exercise Type

### Single CTE Exercises (1-3)
- **Learn**: Basic aggregation and filtering
- **Concepts**: GROUP BY, aggregate functions, HAVING/WHERE

### Multiple CTE Exercises (4-6)
- **Learn**: Building complex queries step-by-step
- **Concepts**: Window functions, CASE statements, multiple aggregations

### Recursive CTEs (7)
- **Learn**: Self-referencing queries
- **Concepts**: Base case, recursion, termination conditions

### Advanced Challenges (1-3)
- **Learn**: Real-world analysis patterns
- **Concepts**: Multi-level aggregation, composite scoring, comprehensive reporting

---

## 🎯 Common Patterns to Remember

### Pattern 1: Filter Aggregated Data
```sql
WITH stats AS (
    SELECT group_col, COUNT(*) as cnt, AVG(value) as avg_val
    FROM table
    GROUP BY group_col
)
SELECT * FROM stats WHERE avg_val > 10;
```

### Pattern 2: Rank Within Groups
```sql
WITH ranked AS (
    SELECT
        group_col,
        value,
        ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY value DESC) as rnk
    FROM table
)
SELECT * FROM ranked WHERE rnk <= 5;
```

### Pattern 3: Progressive Filtering
```sql
WITH filtered1 AS (...),
     filtered2 AS (SELECT * FROM filtered1 WHERE condition),
     filtered3 AS (SELECT * FROM filtered2 WHERE condition)
SELECT * FROM filtered3;
```

---

## 📚 References

- [Mode SQL Tutorial](https://mode.com/sql-tutorial/)
- [Window Functions Guide](https://mode.com/sql-tutorial/sql-window-functions/)
- [SQL Aggregate Functions](https://mode.com/sql-tutorial/sql-aggregate-functions/)

---

Good luck! Start with Exercise 1 and build up your CTE skills using real flight data. 🛫
