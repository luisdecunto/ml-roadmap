# SQL Common Table Expressions (CTEs) - Practice Guide

**Time:** 2-3 hours
**Difficulty:** Intermediate
**Prerequisites:** Basic SQL (SELECT, JOINs, aggregations)

This guide will help you master CTEs (Common Table Expressions) with the WITH clause through practical examples and exercises.

📝 **[Quick Solutions Reference](../practice/SQL_CTE_SOLUTIONS.md)** - Check your answers here!

---

## 📋 Progress Tracker

Track your progress through the exercises:

### Part 1: Basic CTEs
- [ ] Example 1: Simple CTE
- [ ] Example 2: Multiple CTEs
- [ ] Exercise 1.1: Customer Totals

### Part 2: Multi-Step Calculations
- [ ] Example 3: Sales Ranking
- [ ] Example 4: Customer Cohort Analysis
- [ ] Exercise 2.1: Revenue by Category

### Part 3: Recursive CTEs
- [ ] Example 5: Employee Hierarchy
- [ ] Example 6: Graph Traversal
- [ ] Exercise 3.1: Region Hierarchy

### Challenge Problems
- [ ] Challenge 1: Running Totals with CTEs
- [ ] Challenge 2: Complex Recursive Query

---

## 📚 What are CTEs?

**Common Table Expressions (CTEs)** are temporary named result sets that exist only within the execution scope of a single SQL statement. They make complex queries more readable and maintainable.

### Basic Syntax

```sql
WITH cte_name AS (
    SELECT column1, column2
    FROM table_name
    WHERE condition
)
SELECT *
FROM cte_name;
```

### Why Use CTEs?

1. **Readability** - Break complex queries into logical steps
2. **Maintainability** - Easier to debug and modify
3. **Reusability** - Reference the same subquery multiple times
4. **Recursive queries** - Traverse hierarchical data

---

## 🎯 Part 1: Basic CTEs (30 min)

### Example 1: Simple CTE

Let's say we have an `employees` table and want to find employees who earn more than the average salary.

**Without CTE (using subquery):**
```sql
SELECT employee_id, name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```

**With CTE (more readable):**
```sql
WITH avg_salary AS (
    SELECT AVG(salary) AS average
    FROM employees
)
SELECT e.employee_id, e.name, e.salary
FROM employees e
CROSS JOIN avg_salary
WHERE e.salary > avg_salary.average;
```

### Example 2: Multiple CTEs

You can define multiple CTEs separated by commas:

```sql
WITH
    high_earners AS (
        SELECT employee_id, name, salary, department_id
        FROM employees
        WHERE salary > 75000
    ),
    dept_summary AS (
        SELECT department_id, COUNT(*) AS dept_count
        FROM high_earners
        GROUP BY department_id
    )
SELECT
    he.name,
    he.salary,
    ds.dept_count
FROM high_earners he
JOIN dept_summary ds ON he.department_id = ds.department_id
WHERE ds.dept_count > 5;
```

### 💻 Practice Exercise 1.1

Given this `orders` table:

| order_id | customer_id | order_date | amount |
|----------|-------------|------------|--------|
| 1        | 101         | 2024-01-05 | 250.00 |
| 2        | 102         | 2024-01-06 | 180.00 |
| 3        | 101         | 2024-01-08 | 320.00 |

Write a CTE to:
1. Calculate total sales per customer
2. Filter customers with total sales > $400

**Solution:**
```sql
WITH customer_totals AS (
    SELECT
        customer_id,
        SUM(amount) AS total_sales,
        COUNT(*) AS order_count
    FROM orders
    GROUP BY customer_id
)
SELECT
    customer_id,
    total_sales,
    order_count
FROM customer_totals
WHERE total_sales > 400;
```

---

## 🔥 Part 2: Multi-Step Calculations with CTEs (45 min)

CTEs shine when you need to perform calculations in stages.

### Example 3: Sales Ranking

```sql
WITH
    -- Step 1: Calculate monthly sales per salesperson
    monthly_sales AS (
        SELECT
            salesperson_id,
            DATE_TRUNC('month', sale_date) AS month,
            SUM(amount) AS total_sales
        FROM sales
        GROUP BY salesperson_id, DATE_TRUNC('month', sale_date)
    ),
    -- Step 2: Rank salespeople within each month
    ranked_sales AS (
        SELECT
            salesperson_id,
            month,
            total_sales,
            RANK() OVER (PARTITION BY month ORDER BY total_sales DESC) AS sales_rank
        FROM monthly_sales
    )
-- Step 3: Get top 3 performers each month
SELECT
    month,
    salesperson_id,
    total_sales,
    sales_rank
FROM ranked_sales
WHERE sales_rank <= 3
ORDER BY month, sales_rank;
```

### Example 4: Customer Cohort Analysis

```sql
WITH
    -- First purchase date for each customer
    first_purchase AS (
        SELECT
            customer_id,
            MIN(order_date) AS first_order_date
        FROM orders
        GROUP BY customer_id
    ),
    -- Assign cohort based on first purchase month
    customer_cohorts AS (
        SELECT
            customer_id,
            DATE_TRUNC('month', first_order_date) AS cohort_month
        FROM first_purchase
    ),
    -- Calculate metrics per cohort
    cohort_metrics AS (
        SELECT
            cc.cohort_month,
            COUNT(DISTINCT o.customer_id) AS customers,
            SUM(o.amount) AS total_revenue,
            AVG(o.amount) AS avg_order_value
        FROM customer_cohorts cc
        JOIN orders o ON cc.customer_id = o.customer_id
        GROUP BY cc.cohort_month
    )
SELECT
    cohort_month,
    customers,
    total_revenue,
    avg_order_value,
    total_revenue / customers AS revenue_per_customer
FROM cohort_metrics
ORDER BY cohort_month;
```

### 💻 Practice Exercise 2.1

Given `products` and `sales` tables:

**products:**
| product_id | category | price |
|------------|----------|-------|
| 1          | Electronics | 299 |
| 2          | Books    | 15  |
| 3          | Electronics | 799 |

**sales:**
| sale_id | product_id | quantity | sale_date |
|---------|------------|----------|-----------|
| 1       | 1          | 2        | 2024-01-10 |
| 2       | 2          | 5        | 2024-01-11 |
| 3       | 3          | 1        | 2024-01-12 |

Write CTEs to calculate:
1. Total revenue per category
2. Percentage of total revenue for each category
3. Categories contributing > 30% of revenue

**Solution:**
```sql
WITH
    -- Calculate revenue per category
    category_revenue AS (
        SELECT
            p.category,
            SUM(p.price * s.quantity) AS revenue
        FROM products p
        JOIN sales s ON p.product_id = s.product_id
        GROUP BY p.category
    ),
    -- Calculate total revenue
    total_revenue AS (
        SELECT SUM(revenue) AS total
        FROM category_revenue
    )
-- Calculate percentages
SELECT
    cr.category,
    cr.revenue,
    ROUND(100.0 * cr.revenue / tr.total, 2) AS revenue_percentage
FROM category_revenue cr
CROSS JOIN total_revenue tr
WHERE cr.revenue / tr.total > 0.30
ORDER BY revenue_percentage DESC;
```

---

## 🌳 Part 3: Recursive CTEs (45 min)

Recursive CTEs are used for hierarchical or tree-structured data (org charts, category trees, etc.).

### Syntax

```sql
WITH RECURSIVE cte_name AS (
    -- Base case (anchor member)
    SELECT ...

    UNION ALL

    -- Recursive case (recursive member)
    SELECT ...
    FROM cte_name
    WHERE termination_condition
)
SELECT * FROM cte_name;
```

### Example 5: Employee Hierarchy

```sql
-- employees table:
-- employee_id | name       | manager_id
-- 1           | Alice CEO  | NULL
-- 2           | Bob VP     | 1
-- 3           | Carol Mgr  | 2
-- 4           | Dave Staff | 3

WITH RECURSIVE employee_hierarchy AS (
    -- Base case: Start with CEO (no manager)
    SELECT
        employee_id,
        name,
        manager_id,
        1 AS level,
        name AS path
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case: Find direct reports
    SELECT
        e.employee_id,
        e.name,
        e.manager_id,
        eh.level + 1,
        eh.path || ' > ' || e.name AS path
    FROM employees e
    JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
)
SELECT
    employee_id,
    name,
    level,
    path AS reporting_chain
FROM employee_hierarchy
ORDER BY level, name;
```

**Output:**
```
employee_id | name       | level | reporting_chain
1           | Alice CEO  | 1     | Alice CEO
2           | Bob VP     | 2     | Alice CEO > Bob VP
3           | Carol Mgr  | 3     | Alice CEO > Bob VP > Carol Mgr
4           | Dave Staff | 4     | Alice CEO > Bob VP > Carol Mgr > Dave Staff
```

### Example 6: Category Tree

```sql
-- categories table:
-- category_id | name        | parent_id
-- 1           | Electronics | NULL
-- 2           | Computers   | 1
-- 3           | Laptops     | 2
-- 4           | Desktops    | 2

WITH RECURSIVE category_tree AS (
    -- Base case: Top-level categories
    SELECT
        category_id,
        name,
        parent_id,
        1 AS depth,
        ARRAY[category_id] AS path_ids,
        name AS full_path
    FROM categories
    WHERE parent_id IS NULL

    UNION ALL

    -- Recursive case: Child categories
    SELECT
        c.category_id,
        c.name,
        c.parent_id,
        ct.depth + 1,
        ct.path_ids || c.category_id,
        ct.full_path || ' / ' || c.name
    FROM categories c
    JOIN category_tree ct ON c.parent_id = ct.category_id
    WHERE ct.depth < 10  -- Prevent infinite loops
)
SELECT
    category_id,
    name,
    depth,
    full_path
FROM category_tree
ORDER BY full_path;
```

### 💻 Practice Exercise 3.1

Given a `regions` table representing geographical hierarchy:

| region_id | name          | parent_id |
|-----------|---------------|-----------|
| 1         | World         | NULL      |
| 2         | North America | 1         |
| 3         | USA           | 2         |
| 4         | California    | 3         |
| 5         | San Francisco | 4         |

Write a recursive CTE to find all descendants of "North America" (region_id = 2).

**Solution:**
```sql
WITH RECURSIVE region_descendants AS (
    -- Base case: Start with North America
    SELECT
        region_id,
        name,
        parent_id,
        1 AS level
    FROM regions
    WHERE region_id = 2

    UNION ALL

    -- Recursive case: Find children
    SELECT
        r.region_id,
        r.name,
        r.parent_id,
        rd.level + 1
    FROM regions r
    JOIN region_descendants rd ON r.parent_id = rd.region_id
)
SELECT
    region_id,
    name,
    level
FROM region_descendants
ORDER BY level, name;
```

---

## 🎓 Part 4: Advanced CTE Patterns (30 min)

### Pattern 1: Running Totals

```sql
WITH daily_sales AS (
    SELECT
        DATE(order_date) AS sale_date,
        SUM(amount) AS daily_total
    FROM orders
    GROUP BY DATE(order_date)
)
SELECT
    sale_date,
    daily_total,
    SUM(daily_total) OVER (ORDER BY sale_date) AS running_total
FROM daily_sales
ORDER BY sale_date;
```

### Pattern 2: Gap Detection

Find missing IDs in a sequence:

```sql
WITH expected_ids AS (
    SELECT generate_series(1, (SELECT MAX(order_id) FROM orders)) AS id
),
actual_ids AS (
    SELECT order_id AS id FROM orders
)
SELECT e.id AS missing_id
FROM expected_ids e
LEFT JOIN actual_ids a ON e.id = a.id
WHERE a.id IS NULL;
```

### Pattern 3: Pivoting Data

```sql
WITH monthly_metrics AS (
    SELECT
        salesperson_id,
        DATE_TRUNC('month', sale_date) AS month,
        SUM(amount) AS total_sales
    FROM sales
    GROUP BY salesperson_id, DATE_TRUNC('month', sale_date)
)
SELECT
    salesperson_id,
    SUM(CASE WHEN month = '2024-01-01' THEN total_sales ELSE 0 END) AS jan_sales,
    SUM(CASE WHEN month = '2024-02-01' THEN total_sales ELSE 0 END) AS feb_sales,
    SUM(CASE WHEN month = '2024-03-01' THEN total_sales ELSE 0 END) AS mar_sales
FROM monthly_metrics
GROUP BY salesperson_id;
```

### Pattern 4: Self-Join Elimination

Instead of self-joining the same table multiple times, use CTEs:

```sql
-- Find customers who made purchases in consecutive months
WITH
    monthly_customers AS (
        SELECT DISTINCT
            customer_id,
            DATE_TRUNC('month', order_date) AS purchase_month
        FROM orders
    ),
    next_month AS (
        SELECT
            customer_id,
            purchase_month,
            LEAD(purchase_month) OVER (PARTITION BY customer_id ORDER BY purchase_month) AS next_purchase_month
        FROM monthly_customers
    )
SELECT
    customer_id,
    purchase_month,
    next_purchase_month
FROM next_month
WHERE next_purchase_month = purchase_month + INTERVAL '1 month';
```

---

## 💪 Practice Exercises

### Exercise 1: Sales Performance Analysis

Given `sales` table with columns: `sale_id`, `salesperson_id`, `product_id`, `amount`, `sale_date`

Write a query using CTEs to:
1. Calculate each salesperson's total sales
2. Calculate the average sales across all salespeople
3. Identify salespeople performing above average
4. Show their total sales and percentage above average

<details>
<summary>Click for solution</summary>

```sql
WITH
    salesperson_totals AS (
        SELECT
            salesperson_id,
            SUM(amount) AS total_sales
        FROM sales
        GROUP BY salesperson_id
    ),
    avg_sales AS (
        SELECT AVG(total_sales) AS average
        FROM salesperson_totals
    )
SELECT
    st.salesperson_id,
    st.total_sales,
    a.average AS avg_sales,
    st.total_sales - a.average AS above_average,
    ROUND(100.0 * (st.total_sales - a.average) / a.average, 2) AS pct_above_avg
FROM salesperson_totals st
CROSS JOIN avg_sales a
WHERE st.total_sales > a.average
ORDER BY st.total_sales DESC;
```
</details>

### Exercise 2: Customer Retention

Given `orders` table: `order_id`, `customer_id`, `order_date`, `amount`

Calculate customer retention:
1. Group customers by their first purchase month (cohort)
2. For each cohort, calculate how many returned in month 1, 2, 3
3. Calculate retention rate as percentage

<details>
<summary>Click for solution</summary>

```sql
WITH
    first_purchase AS (
        SELECT
            customer_id,
            DATE_TRUNC('month', MIN(order_date)) AS cohort_month
        FROM orders
        GROUP BY customer_id
    ),
    customer_months AS (
        SELECT
            fp.customer_id,
            fp.cohort_month,
            DATE_TRUNC('month', o.order_date) AS purchase_month,
            EXTRACT(MONTH FROM AGE(DATE_TRUNC('month', o.order_date), fp.cohort_month)) AS months_since_first
        FROM first_purchase fp
        JOIN orders o ON fp.customer_id = o.customer_id
    ),
    cohort_size AS (
        SELECT
            cohort_month,
            COUNT(DISTINCT customer_id) AS cohort_customers
        FROM first_purchase
        GROUP BY cohort_month
    ),
    retention_data AS (
        SELECT
            cm.cohort_month,
            cm.months_since_first,
            COUNT(DISTINCT cm.customer_id) AS retained_customers
        FROM customer_months cm
        GROUP BY cm.cohort_month, cm.months_since_first
    )
SELECT
    rd.cohort_month,
    rd.months_since_first,
    rd.retained_customers,
    cs.cohort_customers,
    ROUND(100.0 * rd.retained_customers / cs.cohort_customers, 2) AS retention_rate
FROM retention_data rd
JOIN cohort_size cs ON rd.cohort_month = cs.cohort_month
WHERE rd.months_since_first <= 3
ORDER BY rd.cohort_month, rd.months_since_first;
```
</details>

### Exercise 3: Product Recommendation

Find products frequently bought together (market basket analysis):

Given `order_items`: `order_id`, `product_id`

Find pairs of products that appear together in at least 5 orders.

<details>
<summary>Click for solution</summary>

```sql
WITH
    product_pairs AS (
        SELECT
            oi1.product_id AS product_a,
            oi2.product_id AS product_b,
            oi1.order_id
        FROM order_items oi1
        JOIN order_items oi2
            ON oi1.order_id = oi2.order_id
            AND oi1.product_id < oi2.product_id  -- Avoid duplicates
    ),
    pair_counts AS (
        SELECT
            product_a,
            product_b,
            COUNT(*) AS times_bought_together
        FROM product_pairs
        GROUP BY product_a, product_b
    )
SELECT
    product_a,
    product_b,
    times_bought_together
FROM pair_counts
WHERE times_bought_together >= 5
ORDER BY times_bought_together DESC;
```
</details>

---

## 📖 Additional Resources

1. **MODE Analytics Blog**: [Use Common Table Expressions to Keep Your SQL Clean](https://mode.com/blog/use-common-table-expressions-to-keep-your-sql-clean/)
2. **PostgreSQL Documentation**: [WITH Queries (CTEs)](https://www.postgresql.org/docs/current/queries-with.html)
3. **DataCamp Tutorial**: [CTE in SQL](https://www.datacamp.com/tutorial/cte-sql)

---

## ✅ Checklist

After completing this guide, you should be able to:

- [ ] Write basic CTEs to improve query readability
- [ ] Chain multiple CTEs for multi-step calculations
- [ ] Use recursive CTEs for hierarchical data
- [ ] Apply CTEs for common patterns (running totals, gap detection, pivoting)
- [ ] Debug CTE queries effectively
- [ ] Decide when CTEs are better than subqueries or temporary tables

---

## 🚀 Next Steps

1. Practice the 5 coding tasks in Week 2 of the ML Roadmap
2. Learn Window Functions (companion guide coming soon)
3. Combine CTEs with Window Functions for powerful analytics
4. Study query performance and indexing

Happy querying! 🎉
