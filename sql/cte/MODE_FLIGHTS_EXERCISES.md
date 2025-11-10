# SQL CTE Exercises - Flight Data (Mode Public Warehouse)

These exercises use Mode's real flight datasets: `tutorial.us_flights` and `tutorial.flights`

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

## 📚 References

- [Mode SQL Tutorial](https://mode.com/sql-tutorial/)
- [Window Functions Guide](https://mode.com/sql-tutorial/sql-window-functions/)
- [SQL Aggregate Functions](https://mode.com/sql-tutorial/sql-aggregate-functions/)

---

## ✅ Progress Tracking

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

## 💡 Need Help?

Check out the **[Solutions](MODE_FLIGHTS_SOLUTIONS.md)** file for:
- Complete query answers
- Step-by-step explanations
- Key concepts highlighted
- Common mistakes to avoid

---

Good luck! Start with Exercise 1 and build up your CTE skills using real flight data. 🛫
