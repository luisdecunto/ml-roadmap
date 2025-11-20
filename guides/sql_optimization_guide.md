# SQL Query Optimization & Performance Guide

**Time:** 2-3 hours
**Difficulty:** Intermediate to Advanced
**Prerequisites:** SQL basics, CTEs, Window Functions

Learn to analyze, optimize, and tune SQL queries for better performance using EXPLAIN ANALYZE, indexes, and best practices.

📝 **Solutions and examples included inline** - Practical optimization techniques with before/after comparisons!

---

## 📋 Progress Tracker

Track your progress through optimization techniques:

### Part 1: EXPLAIN and Analysis
- [ ] Understanding EXPLAIN output
- [ ] Using EXPLAIN ANALYZE
- [ ] Reading query plans
- [ ] Exercise 1: Analyze slow query

### Part 2: Indexes
- [ ] B-tree indexes
- [ ] Composite indexes
- [ ] Partial indexes
- [ ] Exercise 2: Add appropriate indexes
- [ ] Exercise 3: Index selectivity

### Part 3: Query Rewriting
- [ ] Avoiding SELECT *
- [ ] Efficient JOINs
- [ ] Subquery vs JOIN
- [ ] Exercise 4: Rewrite inefficient query

### Part 4: Advanced Techniques
- [ ] Materialized views
- [ ] Query caching
- [ ] Partition pruning
- [ ] Exercise 5: Optimize complex analytics query

### Challenge Problems
- [ ] Challenge 1: Debug slow production query
- [ ] Challenge 2: Design optimal index strategy

---

## 📚 Understanding Query Performance

### The Query Execution Pipeline

1. **Parser** - Checks syntax
2. **Planner** - Creates execution plan
3. **Optimizer** - Chooses best plan
4. **Executor** - Runs the query

Our job: Help the optimizer make good decisions!

---

## 🔍 Part 1: EXPLAIN and EXPLAIN ANALYZE (45 min)

### EXPLAIN

Shows the **query plan** without executing:

```sql
EXPLAIN
SELECT * FROM orders WHERE customer_id = 100;
```

**Output example:**
```
Seq Scan on orders  (cost=0.00..180.00 rows=5 width=32)
  Filter: (customer_id = 100)
```

**Key metrics:**
- **Cost**: Estimated processing cost (startup..total)
- **Rows**: Estimated number of rows
- **Width**: Average row size in bytes

### EXPLAIN ANALYZE

**Executes** the query and shows actual performance:

```sql
EXPLAIN ANALYZE
SELECT * FROM orders WHERE customer_id = 100;
```

**Output example:**
```
Seq Scan on orders  (cost=0.00..180.00 rows=5 width=32)
                    (actual time=0.023..4.567 rows=8 loops=1)
  Filter: (customer_id = 100)
  Rows Removed by Filter: 9992
Planning Time: 0.123 ms
Execution Time: 4.892 ms
```

**Key metrics:**
- **actual time**: Real execution time (startup..total) in milliseconds
- **rows**: Actual rows returned
- **loops**: Number of times node was executed
- **Planning Time**: Time to create execution plan
- **Execution Time**: Time to execute query

### Common Scan Types

| Scan Type | Description | Performance |
|-----------|-------------|-------------|
| **Seq Scan** | Scans entire table | Slow for large tables |
| **Index Scan** | Uses index + table | Fast for small result sets |
| **Index Only Scan** | Uses index only | Fastest (no table access) |
| **Bitmap Scan** | Uses index + bitmap | Good for moderate result sets |

### 💻 Practice Exercise 1.1

Run EXPLAIN ANALYZE on a slow query and identify the bottleneck:

```sql
-- Slow query
EXPLAIN ANALYZE
SELECT o.order_id, c.name, o.amount
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_date > '2024-01-01'
ORDER BY o.amount DESC
LIMIT 10;
```

Look for:
1. Sequential scans on large tables
2. High "Rows Removed by Filter"
3. Expensive sorts
4. Large differences between estimated and actual rows

---

## 📑 Part 2: Indexes (60 min)

### What is an Index?

Like a book index - helps find data without scanning everything!

### Creating Indexes

```sql
-- Simple index
CREATE INDEX idx_customers_email ON customers(email);

-- Multi-column index (order matters!)
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);

-- Partial index (only some rows)
CREATE INDEX idx_active_users ON users(user_id) WHERE is_active = true;

-- Unique index
CREATE UNIQUE INDEX idx_users_email ON users(email);
```

### When to Use Indexes

✅ **Use indexes for:**
- Columns in WHERE clauses
- Columns in JOIN conditions
- Columns in ORDER BY
- Columns used for grouping
- Foreign keys

❌ **Don't index:**
- Small tables (< 1000 rows)
- Columns with low cardinality (few unique values)
- Columns frequently updated
- Every column (indexes slow down writes!)

### Index Column Order Matters!

For multi-column indexes, order by:
1. Equality conditions (=)
2. Range conditions (>, <, BETWEEN)
3. Sort order (ORDER BY)

```sql
-- Query: WHERE customer_id = 100 AND order_date > '2024-01-01' ORDER BY amount
-- Good index:
CREATE INDEX idx_orders_opt ON orders(customer_id, order_date, amount);

-- customer_id first (equality)
-- order_date second (range)
-- amount third (sort)
```

### Checking Index Usage

```sql
-- List all indexes on a table
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'orders';

-- Check if index is being used
EXPLAIN ANALYZE
SELECT * FROM orders WHERE customer_id = 100;
-- Look for "Index Scan using idx_..."
```

### 💻 Practice Exercise 2.1

Given this slow query:

```sql
SELECT *
FROM orders
WHERE status = 'completed'
  AND order_date BETWEEN '2024-01-01' AND '2024-03-31'
ORDER BY order_date DESC;
```

1. Run EXPLAIN ANALYZE to see current performance
2. Create an appropriate index
3. Run EXPLAIN ANALYZE again to verify improvement

**Solution:**
```sql
-- Before: Seq Scan
EXPLAIN ANALYZE
SELECT * FROM orders
WHERE status = 'completed'
  AND order_date BETWEEN '2024-01-01' AND '2024-03-31'
ORDER BY order_date DESC;

-- Create index
CREATE INDEX idx_orders_status_date
ON orders(status, order_date DESC);

-- After: Index Scan (much faster!)
EXPLAIN ANALYZE
SELECT * FROM orders
WHERE status = 'completed'
  AND order_date BETWEEN '2024-01-01' AND '2024-03-31'
ORDER BY order_date DESC;
```

---

## ⚡ Part 3: Query Optimization Techniques (60 min)

### Technique 1: Avoid SELECT *

❌ **Bad:**
```sql
SELECT * FROM users WHERE user_id = 100;
```

✅ **Good:**
```sql
SELECT user_id, name, email FROM users WHERE user_id = 100;
```

**Why:** Fetches only needed columns, enables Index Only Scans.

### Technique 2: Use LIMIT for Pagination

❌ **Bad (slow for page 1000):**
```sql
SELECT * FROM orders ORDER BY order_date OFFSET 10000 LIMIT 10;
```

✅ **Good (keyset pagination):**
```sql
SELECT * FROM orders
WHERE order_date < '2024-01-15'  -- Last date from previous page
ORDER BY order_date DESC
LIMIT 10;
```

### Technique 3: Avoid Functions in WHERE

❌ **Bad (can't use index):**
```sql
SELECT * FROM orders
WHERE EXTRACT(YEAR FROM order_date) = 2024;
```

✅ **Good (can use index):**
```sql
SELECT * FROM orders
WHERE order_date >= '2024-01-01'
  AND order_date < '2025-01-01';
```

### Technique 4: Use EXISTS instead of IN for Large Sets

❌ **Bad (slow for large subquery):**
```sql
SELECT * FROM customers
WHERE customer_id IN (
    SELECT customer_id FROM orders WHERE amount > 1000
);
```

✅ **Good:**
```sql
SELECT * FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.customer_id = c.customer_id AND o.amount > 1000
);
```

### Technique 5: JOIN Order Matters

Start with smallest table:

```sql
-- Assume: customers = 1M rows, orders = 10M rows, premium_users = 1000 rows

-- Bad: Joins large tables first
SELECT c.name, o.amount
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN premium_users p ON c.customer_id = p.user_id;

-- Good: Filter early with smallest table
SELECT c.name, o.amount
FROM premium_users p
JOIN customers c ON p.user_id = c.customer_id
JOIN orders o ON c.customer_id = o.customer_id;
```

### Technique 6: Optimize Aggregations

❌ **Bad (counts all rows then filters):**
```sql
SELECT customer_id, COUNT(*)
FROM orders
GROUP BY customer_id
HAVING COUNT(*) > 10;
```

✅ **Good (with index on customer_id):**
```sql
WITH customer_counts AS (
    SELECT customer_id, COUNT(*) as order_count
    FROM orders
    GROUP BY customer_id
)
SELECT customer_id, order_count
FROM customer_counts
WHERE order_count > 10;
```

### 💻 Practice Exercise 3.1

Optimize this slow query:

```sql
-- Slow query
SELECT
    u.name,
    COUNT(*) as post_count
FROM users u
LEFT JOIN posts p ON LOWER(u.email) = LOWER(p.author_email)
WHERE EXTRACT(YEAR FROM p.created_at) = 2024
GROUP BY u.user_id, u.name
HAVING COUNT(*) > 5;
```

**Issues:**
1. Functions in JOIN condition (LOWER)
2. Function in WHERE (EXTRACT)
3. LEFT JOIN should be INNER JOIN
4. Missing indexes

**Optimized Solution:**
```sql
-- Add indexes
CREATE INDEX idx_users_email_lower ON users(LOWER(email));
CREATE INDEX idx_posts_author_created ON posts(LOWER(author_email), created_at);

-- Optimized query
SELECT
    u.name,
    COUNT(*) as post_count
FROM users u
INNER JOIN posts p ON LOWER(u.email) = LOWER(p.author_email)
WHERE p.created_at >= '2024-01-01'
  AND p.created_at < '2025-01-01'
GROUP BY u.user_id, u.name
HAVING COUNT(*) > 5;
```

---

## 📊 Part 4: Performance Monitoring (30 min)

### Slow Query Log

Enable slow query logging:

```sql
-- PostgreSQL
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log queries > 1 second
SELECT pg_reload_conf();
```

### Find Slow Queries

```sql
-- PostgreSQL: Top 10 slowest queries
SELECT
    query,
    calls,
    total_time,
    mean_time,
    max_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

### Analyze Table Statistics

Update table statistics for better query planning:

```sql
ANALYZE orders;  -- Single table
ANALYZE;         -- All tables
```

### Check Table/Index Sizes

```sql
-- Table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Index sizes
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(schemaname||'.'||indexname)) AS size
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY pg_relation_size(schemaname||'.'||indexname) DESC;
```

### Unused Indexes

```sql
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan AS index_scans
FROM pg_stat_user_indexes
WHERE idx_scan = 0
  AND indexrelname NOT LIKE 'pg_toast%'
ORDER BY pg_relation_size(schemaname||'.'||indexrelname) DESC;
```

---

## 🎯 Optimization Workflow

### Step 1: Identify Slow Query
- Check application logs
- Monitor slow query log
- Use pg_stat_statements

### Step 2: Analyze with EXPLAIN ANALYZE
```sql
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
[Your slow query here];
```

### Step 3: Look for Red Flags
- ❌ Seq Scan on large tables
- ❌ High "Rows Removed by Filter"
- ❌ Expensive sorts (high cost)
- ❌ Nested Loop with large outer relation
- ❌ Large discrepancy between estimated and actual rows

### Step 4: Apply Fixes
1. **Add indexes** for WHERE, JOIN, ORDER BY columns
2. **Rewrite query** to avoid functions on indexed columns
3. **Add/update statistics**: `ANALYZE table_name`
4. **Consider partitioning** for very large tables
5. **Use materialized views** for complex aggregations

### Step 5: Verify Improvement
```sql
EXPLAIN ANALYZE [Optimized query];
```

Compare execution time before and after!

---

## 💪 Comprehensive Exercise

### Scenario: E-commerce Database

Tables:
- `customers` (1M rows)
- `orders` (10M rows)
- `order_items` (50M rows)
- `products` (100K rows)

### Slow Query:

```sql
SELECT
    c.name,
    COUNT(DISTINCT o.order_id) as total_orders,
    SUM(oi.quantity * p.price) as total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
WHERE o.order_date >= '2024-01-01'
GROUP BY c.customer_id, c.name
HAVING SUM(oi.quantity * p.price) > 10000
ORDER BY total_spent DESC
LIMIT 100;
```

**Your Task:**
1. Run EXPLAIN ANALYZE
2. Identify performance issues
3. Create appropriate indexes
4. Rewrite query if needed
5. Document before/after performance

<details>
<summary>Click for solution</summary>

**Issues Found:**
1. LEFT JOINs should be INNER JOINs (we filter on o.order_date)
2. No indexes on join columns
3. Expensive aggregation on 50M rows
4. HAVING clause forces full aggregation before filtering

**Optimizations:**

```sql
-- Create indexes
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);
CREATE INDEX idx_products_id_price ON products(product_id, price);

-- Optimized query
WITH high_value_orders AS (
    SELECT
        o.customer_id,
        o.order_id
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.order_date >= '2024-01-01'
    GROUP BY o.customer_id, o.order_id
    HAVING SUM(oi.quantity * p.price) > 10000
)
SELECT
    c.name,
    COUNT(DISTINCT hvo.order_id) as total_orders,
    SUM(oi.quantity * p.price) as total_spent
FROM high_value_orders hvo
JOIN customers c ON hvo.customer_id = c.customer_id
JOIN order_items oi ON hvo.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
GROUP BY c.customer_id, c.name
ORDER BY total_spent DESC
LIMIT 100;
```

**Results:**
- Before: 45 seconds
- After: 1.2 seconds
- Improvement: 97% faster!

</details>

---

## 📖 Resources

1. **Use The Index, Luke**: https://use-the-index-luke.com/
2. **PostgreSQL Performance Optimization**: https://www.postgresql.org/docs/current/performance-tips.html
3. **EXPLAIN Visualizer**: https://explain.dalibo.com/
4. **pgMustard** (EXPLAIN analyzer): https://www.pgmustard.com/

---

## ✅ Mastery Checklist

After completing this guide, you should be able to:

- [ ] Read and interpret EXPLAIN ANALYZE output
- [ ] Identify performance bottlenecks (seq scans, sorts, filters)
- [ ] Create appropriate indexes for different query patterns
- [ ] Rewrite queries to use indexes effectively
- [ ] Avoid common anti-patterns (functions in WHERE, SELECT *)
- [ ] Monitor query performance over time
- [ ] Document before/after optimization improvements

---

## 🚀 Next Steps

1. Practice optimizing 5 slow queries with EXPLAIN ANALYZE
2. Learn about advanced indexing (GiST, GIN, BRIN)
3. Study PostgreSQL query planner internals
4. Explore partitioning for very large tables
5. Learn about connection pooling and caching

Now go make those queries fly! ⚡
