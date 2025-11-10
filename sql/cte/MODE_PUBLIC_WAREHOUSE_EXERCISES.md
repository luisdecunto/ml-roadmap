# SQL CTE Exercises - Using Mode Public Warehouse

These exercises use Mode's built-in sample databases (no setup required!). All data is immediately available when you log in to Mode.

## 📊 Available Sample Tables

Mode's public warehouse includes several datasets. The most useful for CTE practice:

### **Tutorials Database** (Recommended)
- `users` - User information
- `orders` - Customer orders
- `orders_items` - Order line items
- `events` - User event tracking

## 🎯 How to Use This Guide

1. Log in to [app.mode.com](https://app.mode.com)
2. Create a new **SQL Query** or **Report**
3. Select **Data Source**: Choose "Mode Public Warehouse" or "Tutorials" database
4. Copy the exercise query into the editor
5. Run it (Ctrl+Enter)
6. Compare your results with expected output

---

## Easy Exercises

### Exercise 1: Users Above Average Age
**Task**: Find all users older than the average age using a CTE.

**Database**: `Tutorials`

**Your Solution:**
```sql
-- Write your CTE here
```

**Hint**: Use `users` table, calculate AVG(age)

**Expected Output**: Should show user_id, name, age for users above average age

---

### Exercise 2: High-Value Orders
**Task**: Find customers whose total order value exceeds $100 using a CTE.

**Database**: `Tutorials`

**Your Solution:**
```sql
-- Write your CTE here
```

**Tables**:
- `orders` (order_id, user_id, total)
- `users` (user_id, name)

**Hint**:
1. Create CTE: Sum orders by user_id
2. Filter where total > 100
3. Join with users to get names

**Expected Pattern**:
```
user_id | name      | total_spent
--------|-----------|-------------
    4   | [Name]    | $[Amount]
    ...
```

---

### Exercise 3: Order Count by User
**Task**: Show each user with their total number of orders and average order value.

**Your Solution:**
```sql
-- Write your CTE here
```

**Hint**: Group by user_id, use COUNT(*) and AVG(total)

---

## Medium Exercises

### Exercise 4: Top 3 Users by Total Spent
**Task**: Find the top 3 users who spent the most money, using two CTEs:
1. `user_spending` - Total spent per user
2. `ranked_users` - Users ranked by spending

**Your Solution:**
```sql
WITH user_spending AS (
    -- Your query here
),
ranked_users AS (
    -- Your query here
)
SELECT * FROM ranked_users;
```

**Hint**: Use window function RANK() OVER (ORDER BY total DESC)

---

### Exercise 5: Users with Multiple Orders
**Task**: Find users who have placed more than 2 orders, with their order details.

**Your Solution:**
```sql
WITH user_order_counts AS (
    -- Calculate orders per user
),
multi_order_users AS (
    -- Filter for users with >2 orders
)
SELECT u.name, u.user_id, COUNT(*) as total_orders
FROM multi_order_users m
JOIN users u ON m.user_id = u.user_id
GROUP BY u.user_id, u.name;
```

---

### Exercise 6: Order Items Analysis
**Task**: Find products that appear in more than 5 orders, with their average quantity per order.

**Tables**: `order_items` (order_id, product_id, quantity, price)

**Your Solution:**
```sql
WITH product_orders AS (
    -- Count distinct orders per product
),
popular_products AS (
    -- Filter for products in >5 orders
)
SELECT * FROM popular_products;
```

---

## Hard Exercises (Recursive CTEs)

### Exercise 7: Simple Sequence Generator
**Task**: Create a sequence of numbers from 1 to 10 using a RECURSIVE CTE.

**Your Solution:**
```sql
WITH RECURSIVE numbers AS (
    -- Base case: start at 1
    SELECT 1 as n

    UNION ALL

    -- Recursive case: increment
    SELECT n + 1 FROM numbers WHERE n < 10
)
SELECT * FROM numbers;
```

**Expected Output**: 1, 2, 3, ..., 10

**Hint**: RECURSIVE CTEs need:
- Base case (starting point)
- UNION ALL
- Recursive case with WHERE to stop

---

### Exercise 8: User Session Chain (if events table has session data)
**Task**: For each user, build a chain of events in order.

**Your Solution:**
```sql
WITH RECURSIVE event_chain AS (
    -- Base case: first event per user
    SELECT user_id, event_id, event_type, 1 as event_number
    FROM events
    WHERE [your base case condition]

    UNION ALL

    -- Recursive case: next events
    SELECT e.user_id, e.event_id, e.event_type, ec.event_number + 1
    FROM events e
    JOIN event_chain ec ON [your join condition]
    WHERE ec.event_number < 10
)
SELECT * FROM event_chain ORDER BY user_id, event_number;
```

---

## Challenge Exercises

### Challenge 1: Multi-CTE User Segmentation
**Task**: Create three CTEs to segment users:
1. `order_stats` - User order stats (count, total, average)
2. `user_segments` - Classify as 'High Value' (total > $500), 'Regular' ($100-$500), or 'Low Value' (<$100)
3. `segment_summary` - Count users in each segment

**Your Solution:**
```sql
WITH order_stats AS (
    -- Your query
),
user_segments AS (
    -- Your query
),
segment_summary AS (
    -- Your query
)
SELECT * FROM segment_summary;
```

**Expected Pattern**:
```
segment      | user_count
-------------|------------
High Value   | X
Regular      | Y
Low Value    | Z
```

---

### Challenge 2: Cohort Analysis
**Task**: Group users by their first order date (cohort), calculate retention metrics.

**Your Solution:**
```sql
WITH user_cohorts AS (
    -- User's first order date
),
order_timeline AS (
    -- All orders with cohort
),
cohort_stats AS (
    -- Count users per cohort
)
SELECT * FROM cohort_stats;
```

---

### Challenge 3: Time-Series Aggregation
**Task**: Show daily order totals and running total using CTEs.

**Your Solution:**
```sql
WITH daily_orders AS (
    -- Sum of orders by date
),
running_total AS (
    -- Running sum using window function
)
SELECT * FROM running_total ORDER BY order_date;
```

---

## 📊 Verification Queries

Before starting exercises, verify you can access the data:

```sql
-- Check users table
SELECT COUNT(*) as user_count FROM users;

-- Check orders table
SELECT COUNT(*) as order_count FROM orders;

-- Check first few users
SELECT * FROM users LIMIT 5;

-- Check first few orders
SELECT * FROM orders LIMIT 5;
```

---

## 🚀 Recommended Learning Path

**Session 1 (30 min):**
- Set up Mode and verify data access
- Complete Exercises 1-3 (basic CTEs)

**Session 2 (30 min):**
- Complete Exercises 4-6 (multiple CTEs and joins)

**Session 3 (30 min):**
- Complete Exercise 7 (recursive CTE concept)
- Attempt Exercises 8+ (advanced)

**Session 4 (30 min):**
- Complete Challenge exercises
- Experiment and optimize your solutions

---

## 💡 Tips for Mode

### Finding Table Schemas
In Mode, you can:
1. Click "Schema" on the left side
2. Browse available tables and columns
3. Click a table to see sample data

### Running Queries
- **Run**: Ctrl+Enter or click "Run Query"
- **Stop**: Click "Stop" while running
- **Format**: Click "Format" to auto-indent

### Viewing Results
- **Table**: Default view
- **Chart**: Click "Visualization" tab for charts
- **Download**: Results can be downloaded as CSV

### Keyboard Shortcuts
- `Ctrl+Enter` - Run query
- `Ctrl+/` - Toggle comment
- `Ctrl+Shift+F` - Format query

---

## 🐛 Common Issues

### "Table not found"
- Verify you selected the correct database
- Check table names (Mode tables are case-sensitive on some warehouses)
- Try: `SELECT * FROM information_schema.tables;` to list all tables

### Query too slow
- Use LIMIT to test before full run
- Mode has timeout limits
- Start with simple queries first

### CTE syntax errors
- Check comma between CTEs: `cte1 AS (...), cte2 AS (...)`
- Verify all CTEs are defined before main query
- Check parentheses are balanced

---

## 📚 References

- [Mode SQL Tutorial](https://mode.com/sql-tutorial/)
- [Mode CTE Guide](https://mode.com/sql-tutorial/introduction-to-sql/common-table-expressions/)
- [Mode Keyboard Shortcuts](https://mode.com/help/articles/sql-editor/)

---

## ✅ Progress Tracking

- [ ] Exercise 1: Users Above Average Age
- [ ] Exercise 2: High-Value Orders
- [ ] Exercise 3: Order Count by User
- [ ] Exercise 4: Top 3 Users by Spending
- [ ] Exercise 5: Users with Multiple Orders
- [ ] Exercise 6: Order Items Analysis
- [ ] Exercise 7: Number Sequence (Recursive)
- [ ] Exercise 8: Event Chain (Recursive)
- [ ] Challenge 1: User Segmentation
- [ ] Challenge 2: Cohort Analysis
- [ ] Challenge 3: Time-Series Aggregation

---

Good luck! Start with Exercise 1 and build up your CTE skills. 🎯
