# SQL Window Functions - Complete Practice Guide

**Time:** 3-4 hours
**Difficulty:** Intermediate to Advanced
**Prerequisites:** Basic SQL, CTEs

Master SQL window functions (ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, and more) through practical examples.

📝 **Solutions included inline** - Each exercise has a solution directly below it!

---

## 📋 Progress Tracker

Track your progress through the exercises:

### Part 1: Ranking Functions
- [ ] ROW_NUMBER() basics
- [ ] RANK() vs DENSE_RANK()
- [ ] Exercise 1: Employee rankings
- [ ] Exercise 2: Top N per group

### Part 2: Value Functions
- [ ] LAG() and LEAD()
- [ ] FIRST_VALUE() and LAST_VALUE()
- [ ] Exercise 3: Period-over-period comparison
- [ ] Exercise 4: Running differences

### Part 3: Aggregate Window Functions
- [ ] SUM() OVER
- [ ] AVG() OVER with moving windows
- [ ] Exercise 5: Running totals
- [ ] Exercise 6: Moving averages

### Part 4: Frame Clauses
- [ ] ROWS vs RANGE
- [ ] UNBOUNDED PRECEDING/FOLLOWING
- [ ] Exercise 7: Custom window frames
- [ ] Exercise 8: Centered moving average

### Challenge Problems
- [ ] Challenge 1: Gaps and Islands
- [ ] Challenge 2: Median calculation with windows

---

## 📚 What are Window Functions?

**Window functions** perform calculations across a set of table rows that are related to the current row. Unlike aggregate functions, window functions **do not collapse rows** — each row retains its identity.

### Key Concept

```sql
function() OVER (
    PARTITION BY column1   -- Optional: divide rows into groups
    ORDER BY column2       -- Optional: define order within partition
    ROWS/RANGE clause      -- Optional: define window frame
)
```

---

## 🎯 Part 1: Ranking Functions (45 min)

### ROW_NUMBER()

Assigns a unique sequential integer to each row within a partition.

```sql
SELECT
    employee_id,
    name,
    department,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS salary_rank_global,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS salary_rank_dept
FROM employees;
```

**Output:**
```
employee_id | name  | department | salary | salary_rank_global | salary_rank_dept
1           | Alice | Sales      | 95000  | 1                  | 1
2           | Bob   | Sales      | 85000  | 3                  | 2
3           | Carol | IT         | 90000  | 2                  | 1
4           | Dave  | IT         | 75000  | 4                  | 2
```

### RANK()

Assigns ranks with **gaps** when there are ties.

```sql
SELECT
    athlete_name,
    score,
    RANK() OVER (ORDER BY score DESC) AS rank
FROM competition_results;
```

**Example with ties:**
```
athlete_name | score | rank
Alice        | 95    | 1
Bob          | 90    | 2
Carol        | 90    | 2   -- Tie
Dave         | 85    | 4   -- Gap: skips 3
```

### DENSE_RANK()

Assigns ranks **without gaps** when there are ties.

```sql
SELECT
    athlete_name,
    score,
    DENSE_RANK() OVER (ORDER BY score DESC) AS dense_rank
FROM competition_results;
```

**Example with ties:**
```
athlete_name | score | dense_rank
Alice        | 95    | 1
Bob          | 90    | 2
Carol        | 90    | 2   -- Tie
Dave         | 85    | 3   -- No gap
```

### 💻 Practice Exercise 1.1

Given `sales` table:
```
salesperson | month   | amount
Alice       | 2024-01 | 5000
Alice       | 2024-02 | 6000
Bob         | 2024-01 | 4500
Bob         | 2024-02 | 7000
Carol       | 2024-01 | 5000
Carol       | 2024-02 | 5500
```

Write a query to rank salespeople by their February sales using all three ranking functions.

**Solution:**
```sql
SELECT
    salesperson,
    amount AS feb_sales,
    ROW_NUMBER() OVER (ORDER BY amount DESC) AS row_num,
    RANK() OVER (ORDER BY amount DESC) AS rank,
    DENSE_RANK() OVER (ORDER BY amount DESC) AS dense_rank
FROM sales
WHERE month = '2024-02'
ORDER BY amount DESC;
```

---

## 📊 Part 2: Aggregate Window Functions (45 min)

You can use aggregate functions as window functions!

### Running Totals

```sql
SELECT
    order_date,
    amount,
    SUM(amount) OVER (ORDER BY order_date) AS running_total
FROM orders
ORDER BY order_date;
```

**Output:**
```
order_date | amount | running_total
2024-01-01 | 100    | 100
2024-01-02 | 150    | 250
2024-01-03 | 200    | 450
```

### Moving Averages

```sql
SELECT
    sale_date,
    daily_sales,
    AVG(daily_sales) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7day
FROM daily_sales
ORDER BY sale_date;
```

This calculates a 7-day moving average (current row + 6 preceding).

### Percentage of Total

```sql
SELECT
    product_name,
    sales,
    ROUND(100.0 * sales / SUM(sales) OVER (), 2) AS pct_of_total
FROM product_sales;
```

**Output:**
```
product_name | sales | pct_of_total
Product A    | 5000  | 41.67
Product B    | 3000  | 25.00
Product C    | 4000  | 33.33
```

### 💻 Practice Exercise 2.1

Given `monthly_revenue` table:
```
month      | revenue
2024-01-01 | 10000
2024-02-01 | 12000
2024-03-01 | 11000
2024-04-01 | 13000
```

Calculate:
1. Running total revenue
2. 3-month moving average
3. Percentage change from previous month

**Solution:**
```sql
SELECT
    month,
    revenue,
    SUM(revenue) OVER (ORDER BY month) AS running_total,
    AVG(revenue) OVER (
        ORDER BY month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_3mo,
    ROUND(
        100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
        / LAG(revenue) OVER (ORDER BY month),
        2
    ) AS pct_change
FROM monthly_revenue
ORDER BY month;
```

---

## ⏭️ Part 3: LAG and LEAD Functions (45 min)

### LAG()

Access data from a **previous** row.

```sql
LAG(column, offset, default) OVER (ORDER BY ...)
```

**Example: Compare current month to previous month**
```sql
SELECT
    month,
    sales,
    LAG(sales, 1) OVER (ORDER BY month) AS prev_month_sales,
    sales - LAG(sales, 1) OVER (ORDER BY month) AS sales_diff
FROM monthly_sales;
```

**Output:**
```
month   | sales | prev_month_sales | sales_diff
2024-01 | 5000  | NULL             | NULL
2024-02 | 6000  | 5000             | 1000
2024-03 | 5500  | 6000             | -500
```

### LEAD()

Access data from a **following** row.

```sql
LEAD(column, offset, default) OVER (ORDER BY ...)
```

**Example: See next month's sales**
```sql
SELECT
    month,
    sales,
    LEAD(sales, 1) OVER (ORDER BY month) AS next_month_sales
FROM monthly_sales;
```

### Practical Use Case: Churn Detection

```sql
WITH customer_months AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', order_date) AS purchase_month
    FROM orders
    GROUP BY customer_id, DATE_TRUNC('month', order_date)
)
SELECT
    customer_id,
    purchase_month,
    LEAD(purchase_month) OVER (
        PARTITION BY customer_id
        ORDER BY purchase_month
    ) AS next_purchase_month,
    LEAD(purchase_month) OVER (
        PARTITION BY customer_id
        ORDER BY purchase_month
    ) - purchase_month AS months_until_next_purchase
FROM customer_months;
```

### 💻 Practice Exercise 3.1

Given `stock_prices` table:
```
date       | stock  | close_price
2024-01-01 | AAPL   | 150.00
2024-01-02 | AAPL   | 152.50
2024-01-03 | AAPL   | 149.00
2024-01-01 | GOOGL  | 2800.00
2024-01-02 | GOOGL  | 2850.00
```

Calculate daily price changes and identify the largest single-day drop for each stock.

**Solution:**
```sql
WITH price_changes AS (
    SELECT
        date,
        stock,
        close_price,
        LAG(close_price) OVER (PARTITION BY stock ORDER BY date) AS prev_close,
        close_price - LAG(close_price) OVER (PARTITION BY stock ORDER BY date) AS price_change,
        ROUND(
            100.0 * (close_price - LAG(close_price) OVER (PARTITION BY stock ORDER BY date))
            / LAG(close_price) OVER (PARTITION BY stock ORDER BY date),
            2
        ) AS pct_change
    FROM stock_prices
)
SELECT
    stock,
    date,
    price_change,
    pct_change
FROM price_changes
WHERE price_change = (
    SELECT MIN(price_change)
    FROM price_changes pc2
    WHERE pc2.stock = price_changes.stock
)
ORDER BY pct_change;
```

---

## 🪟 Part 4: Frame Clauses (ROWS vs RANGE) (30 min)

Frame clauses define the **window** of rows for aggregate functions.

### ROWS Clause

Based on **physical row positions**.

```sql
-- Last 3 rows (including current)
ROWS BETWEEN 2 PRECEDING AND CURRENT ROW

-- Next 3 rows (including current)
ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING

-- Current row only
ROWS BETWEEN CURRENT ROW AND CURRENT ROW

-- All preceding rows
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW

-- Centered window
ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
```

### RANGE Clause

Based on **logical value ranges** (useful for dates/numbers with duplicates).

```sql
-- All rows with same value
RANGE BETWEEN CURRENT ROW AND CURRENT ROW

-- All preceding values
RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
```

### Example: 7-Day Moving Average

```sql
SELECT
    date,
    revenue,
    AVG(revenue) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7day
FROM daily_revenue;
```

### Example: Year-to-Date Sum

```sql
SELECT
    month,
    revenue,
    SUM(revenue) OVER (
        ORDER BY month
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS ytd_revenue
FROM monthly_revenue
WHERE EXTRACT(YEAR FROM month) = 2024;
```

---

## 🔥 Part 5: Advanced Patterns (45 min)

### Pattern 1: Top N per Group

Find top 3 products per category:

```sql
WITH ranked_products AS (
    SELECT
        category,
        product_name,
        sales,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY sales DESC) AS rank
    FROM products
)
SELECT
    category,
    product_name,
    sales
FROM ranked_products
WHERE rank <= 3
ORDER BY category, rank;
```

### Pattern 2: First and Last Values

```sql
SELECT
    department,
    employee_name,
    hire_date,
    FIRST_VALUE(employee_name) OVER (
        PARTITION BY department
        ORDER BY hire_date
    ) AS first_hire,
    LAST_VALUE(employee_name) OVER (
        PARTITION BY department
        ORDER BY hire_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_hire
FROM employees;
```

**Note:** `LAST_VALUE` needs explicit frame clause to work as expected!

### Pattern 3: Gaps and Islands

Find consecutive sequences:

```sql
WITH numbered_rows AS (
    SELECT
        user_id,
        login_date,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn,
        login_date - (ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date))::INTEGER AS island_id
    FROM user_logins
)
SELECT
    user_id,
    MIN(login_date) AS streak_start,
    MAX(login_date) AS streak_end,
    COUNT(*) AS consecutive_days
FROM numbered_rows
GROUP BY user_id, island_id
HAVING COUNT(*) >= 7  -- Find streaks of 7+ consecutive days
ORDER BY user_id, streak_start;
```

### Pattern 4: Percentiles and NTILE

Divide data into quartiles:

```sql
SELECT
    employee_name,
    salary,
    NTILE(4) OVER (ORDER BY salary) AS salary_quartile,
    PERCENT_RANK() OVER (ORDER BY salary) AS percentile
FROM employees;
```

---

## 💪 Comprehensive Exercises

### Exercise 1: Customer Lifetime Value Ranking

Given `orders` table: `customer_id`, `order_date`, `amount`

Find:
1. Total spending per customer
2. Rank customers by lifetime value
3. Show top 10% of customers (use NTILE or PERCENT_RANK)

<details>
<summary>Click for solution</summary>

```sql
WITH customer_ltv AS (
    SELECT
        customer_id,
        SUM(amount) AS lifetime_value,
        COUNT(*) AS order_count,
        MIN(order_date) AS first_order,
        MAX(order_date) AS last_order
    FROM orders
    GROUP BY customer_id
),
ranked_customers AS (
    SELECT
        customer_id,
        lifetime_value,
        order_count,
        NTILE(10) OVER (ORDER BY lifetime_value DESC) AS decile,
        PERCENT_RANK() OVER (ORDER BY lifetime_value DESC) AS percentile
    FROM customer_ltv
)
SELECT
    customer_id,
    lifetime_value,
    order_count,
    decile,
    ROUND(percentile * 100, 2) AS percentile
FROM ranked_customers
WHERE decile = 1  -- Top 10%
ORDER BY lifetime_value DESC;
```
</details>

### Exercise 2: Month-over-Month Growth

Given `monthly_sales`: `month`, `product_id`, `sales`

Calculate month-over-month growth rate and identify products with 3+ consecutive months of growth.

<details>
<summary>Click for solution</summary>

```sql
WITH sales_with_growth AS (
    SELECT
        month,
        product_id,
        sales,
        LAG(sales) OVER (PARTITION BY product_id ORDER BY month) AS prev_month_sales,
        CASE
            WHEN LAG(sales) OVER (PARTITION BY product_id ORDER BY month) IS NOT NULL
            THEN ROUND(
                100.0 * (sales - LAG(sales) OVER (PARTITION BY product_id ORDER BY month))
                / LAG(sales) OVER (PARTITION BY product_id ORDER BY month),
                2
            )
        END AS mom_growth_pct,
        CASE
            WHEN sales > LAG(sales) OVER (PARTITION BY product_id ORDER BY month)
            THEN 1
            ELSE 0
        END AS is_growing
    FROM monthly_sales
),
growth_streaks AS (
    SELECT
        *,
        SUM(CASE WHEN is_growing = 0 THEN 1 ELSE 0 END)
            OVER (PARTITION BY product_id ORDER BY month) AS streak_group
    FROM sales_with_growth
),
consecutive_growth AS (
    SELECT
        product_id,
        MIN(month) AS streak_start,
        MAX(month) AS streak_end,
        COUNT(*) AS consecutive_growth_months
    FROM growth_streaks
    WHERE is_growing = 1
    GROUP BY product_id, streak_group
    HAVING COUNT(*) >= 3
)
SELECT
    product_id,
    streak_start,
    streak_end,
    consecutive_growth_months
FROM consecutive_growth
ORDER BY consecutive_growth_months DESC, product_id;
```
</details>

### Exercise 3: Cohort Retention Matrix

Given `user_activity`: `user_id`, `activity_date`, `cohort_month` (month user signed up)

Create a retention matrix showing % of each cohort that was active in months 0, 1, 2, 3.

<details>
<summary>Click for solution</summary>

```sql
WITH cohort_data AS (
    SELECT
        cohort_month,
        DATE_TRUNC('month', activity_date) AS activity_month,
        user_id,
        EXTRACT(MONTH FROM AGE(activity_date, cohort_month)) AS months_since_signup
    FROM user_activity
),
cohort_sizes AS (
    SELECT
        cohort_month,
        COUNT(DISTINCT user_id) AS cohort_size
    FROM user_activity
    WHERE DATE_TRUNC('month', activity_date) = cohort_month
    GROUP BY cohort_month
),
retention AS (
    SELECT
        cd.cohort_month,
        cd.months_since_signup,
        COUNT(DISTINCT cd.user_id) AS active_users,
        cs.cohort_size,
        ROUND(100.0 * COUNT(DISTINCT cd.user_id) / cs.cohort_size, 2) AS retention_pct
    FROM cohort_data cd
    JOIN cohort_sizes cs ON cd.cohort_month = cs.cohort_month
    WHERE cd.months_since_signup BETWEEN 0 AND 3
    GROUP BY cd.cohort_month, cd.months_since_signup, cs.cohort_size
)
SELECT
    cohort_month,
    MAX(CASE WHEN months_since_signup = 0 THEN retention_pct END) AS month_0,
    MAX(CASE WHEN months_since_signup = 1 THEN retention_pct END) AS month_1,
    MAX(CASE WHEN months_since_signup = 2 THEN retention_pct END) AS month_2,
    MAX(CASE WHEN months_since_signup = 3 THEN retention_pct END) AS month_3
FROM retention
GROUP BY cohort_month
ORDER BY cohort_month;
```
</details>

---

## 📖 Resources

1. **MODE Analytics**: [SQL Window Functions Tutorial](https://mode.com/sql-tutorial/sql-window-functions/)
2. **PostgreSQL Docs**: [Window Functions](https://www.postgresql.org/docs/current/tutorial-window.html)
3. **Use The Index, Luke**: [Window Functions](https://use-the-index-luke.com/sql/partial-results/window-functions)

---

## ✅ Mastery Checklist

After completing this guide, you should be able to:

- [ ] Use ROW_NUMBER, RANK, and DENSE_RANK appropriately
- [ ] Calculate running totals and moving averages
- [ ] Use LAG/LEAD for time-series analysis
- [ ] Understand ROWS vs RANGE frame clauses
- [ ] Find top N per group with window functions
- [ ] Solve gaps-and-islands problems
- [ ] Create cohort retention analyses
- [ ] Optimize window function performance

---

## 🚀 Next Steps

1. Complete Week 2 practice tasks (write 5 queries using CTEs and window functions)
2. Solve 20 LeetCode SQL Medium problems focused on window functions
3. Practice StrataScratch interview questions
4. Learn query optimization and EXPLAIN ANALYZE

Great job! Window functions are one of the most powerful SQL features. 🎉
