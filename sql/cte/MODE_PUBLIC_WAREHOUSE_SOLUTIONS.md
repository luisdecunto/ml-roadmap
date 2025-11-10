# SQL CTE Exercises - Solutions (Mode Public Warehouse)

Solutions for all exercises using Mode's public warehouse databases.

---

## Easy Exercises

### Exercise 1: Users Above Average Age

**Solution:**
```sql
WITH avg_age AS (
    SELECT AVG(age) as average_age FROM users
)
SELECT
    u.user_id,
    u.name,
    u.age
FROM users u
CROSS JOIN avg_age
WHERE u.age > avg_age.average_age
ORDER BY u.age DESC;
```

**Explanation:**
1. `avg_age` CTE calculates the average age across all users
2. Main query uses CROSS JOIN to access the single average value
3. WHERE clause filters for users above average
4. ORDER BY shows oldest users first

**Key Concepts:**
- Single CTE for simple aggregation
- CROSS JOIN to access scalar CTE result
- Age comparison with calculated average

---

### Exercise 2: High-Value Orders

**Solution:**
```sql
WITH customer_totals AS (
    SELECT
        o.user_id,
        SUM(o.total) as total_spent
    FROM orders o
    GROUP BY o.user_id
    HAVING SUM(o.total) > 100
)
SELECT
    ct.user_id,
    u.name,
    ct.total_spent
FROM customer_totals ct
JOIN users u ON ct.user_id = u.user_id
ORDER BY ct.total_spent DESC;
```

**Explanation:**
1. `customer_totals` CTE groups orders by user and sums the totals
2. HAVING clause filters for totals > $100
3. Main query joins with users to get names
4. Results ordered by spending amount

**Key Concepts:**
- GROUP BY with SUM aggregation
- HAVING clause for filtering grouped results
- JOIN to enrich CTE data
- Better readability than nested subqueries

---

### Exercise 3: Order Count by User

**Solution:**
```sql
WITH user_order_stats AS (
    SELECT
        o.user_id,
        u.name,
        COUNT(*) as order_count,
        AVG(o.total) as avg_order_value,
        SUM(o.total) as total_spent
    FROM orders o
    JOIN users u ON o.user_id = u.user_id
    GROUP BY o.user_id, u.name
)
SELECT *
FROM user_order_stats
ORDER BY order_count DESC;
```

**Explanation:**
1. CTE calculates multiple aggregations per user
2. COUNT(*) gives total orders
3. AVG(total) shows average order value
4. SUM(total) shows total spending
5. JOIN gets user names

**Key Concepts:**
- Multiple aggregations in single CTE
- GROUP BY multiple columns when needed
- Cleaner than multiple subqueries

---

## Medium Exercises

### Exercise 4: Top 3 Users by Total Spent

**Solution:**
```sql
WITH user_spending AS (
    SELECT
        o.user_id,
        u.name,
        SUM(o.total) as total_spent,
        COUNT(*) as order_count
    FROM orders o
    JOIN users u ON o.user_id = u.user_id
    GROUP BY o.user_id, u.name
),
ranked_users AS (
    SELECT
        user_id,
        name,
        total_spent,
        order_count,
        RANK() OVER (ORDER BY total_spent DESC) as spending_rank
    FROM user_spending
)
SELECT
    user_id,
    name,
    total_spent,
    order_count,
    spending_rank
FROM ranked_users
WHERE spending_rank <= 3
ORDER BY spending_rank;
```

**Explanation:**
1. `user_spending` CTE aggregates order data per user
2. `ranked_users` CTE adds window function RANK() to rank by spending
3. Main query filters for top 3 ranks
4. Results ordered by rank

**Key Concepts:**
- Multiple CTEs chained together
- Window function RANK() OVER
- PARTITION BY (implicit - no partition, so global ranking)
- ORDER BY in OVER clause
- CTEs referencing other CTEs

---

### Exercise 5: Users with Multiple Orders

**Solution:**
```sql
WITH user_order_counts AS (
    SELECT
        o.user_id,
        COUNT(*) as total_orders
    FROM orders o
    GROUP BY o.user_id
),
multi_order_users AS (
    SELECT user_id
    FROM user_order_counts
    WHERE total_orders > 2
)
SELECT
    u.user_id,
    u.name,
    m.user_id as found_in_multi,
    uoc.total_orders
FROM multi_order_users m
JOIN users u ON m.user_id = u.user_id
JOIN user_order_counts uoc ON u.user_id = uoc.user_id
ORDER BY uoc.total_orders DESC;
```

**Explanation:**
1. `user_order_counts` CTE counts orders per user
2. `multi_order_users` CTE filters for users with >2 orders
3. Main query joins both CTEs with users table
4. Shows user info with their order count

**Key Concepts:**
- Filtering in CTEs before main query
- Multiple JOINs in main query
- Reusing CTE results

---

### Exercise 6: Order Items Analysis

**Solution:**
```sql
WITH product_orders AS (
    SELECT
        oi.product_id,
        COUNT(DISTINCT oi.order_id) as order_count,
        AVG(oi.quantity) as avg_quantity_per_order,
        SUM(oi.quantity) as total_quantity
    FROM order_items oi
    GROUP BY oi.product_id
),
popular_products AS (
    SELECT
        product_id,
        order_count,
        avg_quantity_per_order,
        total_quantity
    FROM product_orders
    WHERE order_count > 5
)
SELECT *
FROM popular_products
ORDER BY order_count DESC;
```

**Explanation:**
1. `product_orders` CTE aggregates order items by product
2. COUNT(DISTINCT order_id) counts unique orders
3. AVG(quantity) shows average quantity ordered
4. `popular_products` CTE filters for products in >5 orders
5. Results show product popularity

**Key Concepts:**
- COUNT(DISTINCT) to count unique occurrences
- Filtering aggregated data with WHERE
- Progressive filtering through CTEs

---

## Hard Exercises (Recursive CTEs)

### Exercise 7: Simple Sequence Generator

**Solution:**
```sql
WITH RECURSIVE numbers AS (
    -- Base case: start at 1
    SELECT 1 as n

    UNION ALL

    -- Recursive case: increment
    SELECT n + 1
    FROM numbers
    WHERE n < 10
)
SELECT * FROM numbers;
```

**Explanation:**
1. Base case: SELECT 1 as starting point
2. UNION ALL combines results
3. Recursive case: References the CTE itself (numbers)
4. WHERE n < 10 terminates recursion
5. Results: 1, 2, 3, ..., 10

**Key Concepts:**
- RECURSIVE keyword required
- Base case must be simple
- UNION ALL to combine iterations
- WHERE clause for termination (CRITICAL!)
- Self-referencing CTE

**Output:**
```
n
--
1
2
3
4
5
6
7
8
9
10
```

---

### Exercise 8: User Event Chain (Simplified Version)

**Solution** (if events table has created_at or similar timestamp):
```sql
WITH RECURSIVE event_sequence AS (
    -- Base case: earliest event per user
    SELECT
        user_id,
        event_id,
        event_type,
        created_at,
        1 as event_number
    FROM events e1
    WHERE created_at = (SELECT MIN(created_at) FROM events WHERE user_id = e1.user_id)

    UNION ALL

    -- Recursive case: next events in sequence
    SELECT
        e.user_id,
        e.event_id,
        e.event_type,
        e.created_at,
        es.event_number + 1
    FROM events e
    JOIN event_sequence es ON e.user_id = es.user_id
        AND e.created_at > es.created_at
    WHERE es.event_number < 10
)
SELECT
    user_id,
    event_number,
    event_type,
    created_at
FROM event_sequence
ORDER BY user_id, event_number
LIMIT 50;
```

**Explanation:**
1. Base case: Find earliest event per user
2. Recursive case: Join to find next event chronologically
3. WHERE clause limits recursion depth (event_number < 10)
4. Results show event sequence for each user

**Key Concepts:**
- Complex base case finding minimum per group
- Self-join in recursive case
- Time-based ordering in recursion
- Depth limiting for performance

---

## Challenge Exercises

### Challenge 1: Multi-CTE User Segmentation

**Solution:**
```sql
WITH order_stats AS (
    SELECT
        o.user_id,
        u.name,
        COUNT(*) as order_count,
        SUM(o.total) as total_spent,
        AVG(o.total) as avg_order_value
    FROM orders o
    JOIN users u ON o.user_id = u.user_id
    GROUP BY o.user_id, u.name
),
user_segments AS (
    SELECT
        user_id,
        name,
        order_count,
        total_spent,
        avg_order_value,
        CASE
            WHEN total_spent > 500 THEN 'High Value'
            WHEN total_spent >= 100 THEN 'Regular'
            ELSE 'Low Value'
        END as segment
    FROM order_stats
),
segment_summary AS (
    SELECT
        segment,
        COUNT(*) as user_count,
        AVG(total_spent) as avg_segment_spent,
        AVG(order_count) as avg_orders_per_user
    FROM user_segments
    GROUP BY segment
)
SELECT *
FROM segment_summary
ORDER BY
    CASE
        WHEN segment = 'High Value' THEN 1
        WHEN segment = 'Regular' THEN 2
        ELSE 3
    END;
```

**Explanation:**
1. `order_stats` - Aggregate spending per user
2. `user_segments` - Classify using CASE statement
3. `segment_summary` - Count users per segment with metrics
4. Main query sorts by segment priority

**Key Concepts:**
- Three CTEs working together
- CASE statement for classification
- Aggregation at multiple levels
- Multiple GROUP BY operations

**Expected Output Pattern:**
```
segment    | user_count | avg_segment_spent | avg_orders_per_user
-----------|------------|-------------------|---------------------
High Value | X          | $YYY.XX           | Z.ZZ
Regular    | A          | $BBB.BB           | C.CC
Low Value  | D          | $EEE.EE           | F.FF
```

---

### Challenge 2: Cohort Analysis

**Solution:**
```sql
WITH user_first_order AS (
    SELECT
        o.user_id,
        MIN(o.created_at::date) as first_order_date,
        DATE_TRUNC('month', MIN(o.created_at))::date as cohort_month
    FROM orders o
    GROUP BY o.user_id
),
order_timeline AS (
    SELECT
        ufo.user_id,
        ufo.cohort_month,
        o.created_at::date as order_date,
        o.total as order_amount,
        DATE_PART('month', AGE(o.created_at::timestamp, ufo.first_order_date::timestamp)) as months_since_first
    FROM orders o
    JOIN user_first_order ufo ON o.user_id = ufo.user_id
),
cohort_stats AS (
    SELECT
        cohort_month,
        COUNT(DISTINCT user_id) as cohort_size,
        SUM(order_amount) as total_cohort_revenue,
        AVG(order_amount) as avg_order_value,
        COUNT(DISTINCT user_id) as returning_customers
    FROM order_timeline
    WHERE months_since_first >= 0
    GROUP BY cohort_month
)
SELECT
    cohort_month,
    cohort_size,
    total_cohort_revenue,
    avg_order_value,
    ROUND(100.0 * returning_customers / cohort_size, 2) as retention_rate
FROM cohort_stats
ORDER BY cohort_month DESC;
```

**Explanation:**
1. `user_first_order` - Find each user's first order and cohort month
2. `order_timeline` - Track all orders with time since first order
3. `cohort_stats` - Aggregate metrics per cohort
4. Main query calculates retention rate

**Key Concepts:**
- Date/time functions (DATE_TRUNC, DATE_PART)
- Calculating time differences
- Multiple levels of aggregation
- DISTINCT in counting for retention

---

### Challenge 3: Time-Series Aggregation

**Solution:**
```sql
WITH daily_orders AS (
    SELECT
        o.created_at::date as order_date,
        COUNT(*) as daily_order_count,
        SUM(o.total) as daily_total,
        AVG(o.total) as daily_avg_order_value
    FROM orders o
    GROUP BY o.created_at::date
),
running_total AS (
    SELECT
        order_date,
        daily_order_count,
        daily_total,
        daily_avg_order_value,
        SUM(daily_total) OVER (
            ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as cumulative_total,
        AVG(daily_total) OVER (
            ORDER BY order_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) as moving_avg_7day
    FROM daily_orders
)
SELECT *
FROM running_total
ORDER BY order_date DESC;
```

**Explanation:**
1. `daily_orders` - Aggregate all metrics by date
2. `running_total` - Add window functions:
   - Cumulative sum from beginning to current row
   - 7-day moving average (current + 6 previous days)
3. Results show time-series with trends

**Key Concepts:**
- Window functions (SUM OVER, AVG OVER)
- ROWS BETWEEN for window definition
- UNBOUNDED PRECEDING for cumulative
- Moving averages for trends
- Casting date types (::date)

**Expected Output Pattern:**
```
order_date | daily_order_count | daily_total | cumulative_total | moving_avg_7day
-----------|-------------------|-------------|------------------|----------------
2024-12-10 | 5                 | $1,250.00   | $125,000.00     | $1,285.71
2024-12-09 | 4                 | $980.00     | $123,750.00     | $1,275.32
...
```

---

## 🎯 Tips for Learning

### Understanding CTEs Better

1. **Test incrementally**: Run each CTE independently first
   ```sql
   -- Test just the CTE
   WITH my_cte AS (SELECT ...)
   SELECT * FROM my_cte;
   ```

2. **Visualize the data**: Look at CTE output to understand transformations

3. **Compare approaches**: Rewrite solutions without CTEs using subqueries

4. **Practice variations**: Modify exercises (different thresholds, time periods, etc.)

### Common Mistakes to Avoid

1. **Missing comma between CTEs** - Use: `cte1 AS (...), cte2 AS (...)`
2. **Referencing undefined CTEs** - Define all CTEs before using them
3. **Forgetting WHERE termination** - Recursive CTEs need stopping conditions
4. **Performance issues** - Test with LIMIT first, then full queries

### Window Functions Refresher

```sql
-- Basic window function
SUM(amount) OVER (
    PARTITION BY category        -- Optional: split by groups
    ORDER BY date                 -- Required for running totals
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW  -- Window definition
) as running_sum
```

---

## ✅ Next Steps

1. **Run each solution** in Mode and verify results
2. **Try variations**: Change thresholds, date ranges, metrics
3. **Optimize**: See if you can rewrite more efficiently
4. **Combine**: Merge multiple exercises into one complex query
5. **Apply**: Use patterns on real data from your projects

Good luck! These solutions show industry-standard SQL patterns. 🚀
