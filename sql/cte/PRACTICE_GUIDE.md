# SQL CTE Practice Guide

Complete guide with setup instructions and practice exercises.

## 📋 Quick Start

### 1. Setup the Database

**Copy the entire contents of `setup.sql`** and paste it into your SQL client (PostgreSQL, MySQL, SQLite, etc.)

The script will create:
- `employees` - Employee data with salary and department info
- `departments` - Department information
- `orders` - Customer orders
- `products` - Product catalog
- `projects` - Hierarchical project structure
- `project_assignments` - Project-to-employee assignments
- `sales_records` - Sales data by employee and region

### 2. Verify Setup

Run this query to check all tables were created:

```sql
SELECT 'EMPLOYEES' as table_name, COUNT(*) as row_count FROM employees
UNION ALL
SELECT 'DEPARTMENTS', COUNT(*) FROM departments
UNION ALL
SELECT 'ORDERS', COUNT(*) FROM orders
UNION ALL
SELECT 'PRODUCTS', COUNT(*) FROM products
UNION ALL
SELECT 'PROJECTS', COUNT(*) FROM projects
UNION ALL
SELECT 'PROJECT_ASSIGNMENTS', COUNT(*) FROM project_assignments
UNION ALL
SELECT 'SALES_RECORDS', COUNT(*) FROM sales_records;
```

---

## 🎯 Best Way to Practice CTEs

### Phase 1: Learn (30 min)
1. Read the CTE practice guide: `guides/sql_cte_practice_guide.md`
2. Understand the basic syntax and why CTEs are useful
3. Look at the examples provided

### Phase 2: Simple CTEs (30 min)
Start with basic CTEs. Try to:
1. Write the query without CTE first (using subqueries)
2. Then rewrite using CTE to see the difference
3. Run both and compare results

**Example Exercise:**
```sql
-- Without CTE (subquery)
SELECT e.name, e.salary
FROM employees e
WHERE e.salary > (SELECT AVG(salary) FROM employees);

-- With CTE (cleaner!)
WITH avg_salary AS (
    SELECT AVG(salary) AS average FROM employees
)
SELECT e.name, e.salary
FROM employees e
CROSS JOIN avg_salary
WHERE e.salary > avg_salary.average;
```

### Phase 3: Multiple CTEs (30 min)
Practice combining multiple CTEs:

```sql
WITH
high_earners AS (
    SELECT employee_id, name, salary, department_id
    FROM employees
    WHERE salary > 75000
),
dept_summary AS (
    SELECT department_id, COUNT(*) as emp_count
    FROM high_earners
    GROUP BY department_id
)
SELECT h.name, h.salary, d.emp_count
FROM high_earners h
JOIN dept_summary d ON h.department_id = d.department_id;
```

### Phase 4: Recursive CTEs (Advanced, 30 min)
This is where CTEs really shine - for hierarchical data:

```sql
-- Find all projects and their sub-projects
WITH RECURSIVE project_hierarchy AS (
    -- Base case: top-level projects
    SELECT project_id, project_name, parent_project_id, 1 as level
    FROM projects
    WHERE parent_project_id IS NULL

    UNION ALL

    -- Recursive case: child projects
    SELECT p.project_id, p.project_name, p.parent_project_id, ph.level + 1
    FROM projects p
    INNER JOIN project_hierarchy ph ON p.parent_project_id = ph.project_id
)
SELECT REPEAT('  ', level - 1) || project_name as project_hierarchy
FROM project_hierarchy
ORDER BY level, project_id;
```

---

## 📝 Practice Exercises (Difficulty: Easy → Hard)

### Easy: Single CTE

**Exercise 1: High Earners**
Find all employees earning more than the average salary using CTE.
```sql
-- Write your CTE here
```

**Expected Result:**
```
Alice Johnson - $95,000
Carol Williams - $85,000
Grace Lee - $95,000
Jack Anderson - $78,000
```

---

**Exercise 2: Customer Total Orders**
Calculate total order amount per customer, then find customers with >$500 in orders.
```sql
WITH customer_totals AS (
    -- Your query here
)
-- Your final query here
```

---

### Medium: Multiple CTEs

**Exercise 3: Department Salary Analysis**
Create two CTEs:
1. `dept_stats` - average salary per department
2. `above_avg_emps` - employees earning above their dept average

Then join them to show which employees are above their department average.

```sql
WITH
dept_stats AS (
    -- Calculate avg salary per department
),
above_avg_emps AS (
    -- Find employees above dept average
)
-- Join and display
```

---

**Exercise 4: Sales by Region**
Find top 3 salespeople (employees) by region using two CTEs:
1. `regional_sales` - total sales per employee per region
2. `regional_ranking` - rank employees within each region

---

### Hard: Recursive CTEs

**Exercise 5: Project Hierarchy**
Display the complete project hierarchy showing parent and child projects with indentation.

Use RECURSIVE CTE with UNION ALL to traverse the project tree.

---

**Exercise 6: Employee Management Chain**
Find the complete management chain from each employee up to the CEO.

Example:
- David Brown → Carol Williams → Alice Johnson (CEO)
- Frank Miller → Eve Davis → Alice Johnson (CEO)

---

## 🚀 Tips for Success

### 1. **Start Simple**
- Don't jump to recursive CTEs
- Practice with 1, then 2, then 3 CTEs
- Build complexity gradually

### 2. **Compare Approaches**
- Write a query with subqueries first
- Then convert to CTE
- See how CTEs improve readability

### 3. **Test Incrementally**
```sql
-- Test the first CTE alone
WITH high_earners AS (
    SELECT * FROM employees WHERE salary > 75000
)
SELECT * FROM high_earners;

-- Then build on it
WITH high_earners AS (
    SELECT * FROM employees WHERE salary > 75000
),
departments_with_high_earners AS (
    SELECT DISTINCT d.department_name
    FROM high_earners h
    JOIN departments d ON h.department_id = d.department_id
)
SELECT * FROM departments_with_high_earners;
```

### 4. **Use Meaningful Names**
Good CTE names are self-documenting:
- ✅ `customer_order_totals`
- ✅ `employees_above_avg_salary`
- ❌ `temp`
- ❌ `data`

### 5. **Understand Scope**
- CTEs exist only for the current query
- Each CTE can reference previous CTEs
- The main query comes after all CTEs

### 6. **When to Use CTEs**
- ✅ Breaking complex queries into logical steps
- ✅ Reusing the same subquery multiple times
- ✅ Recursive hierarchical queries
- ❌ Very simple queries (use subqueries instead)
- ❌ Performance-critical code (test both approaches)

---

## 📊 Practice Schedule

**Recommended 2-3 hour practice session:**
- 0:00-0:30 - Read guide + setup database
- 0:30-1:00 - Easy exercises (1-2)
- 1:00-1:30 - Medium exercises (3-4)
- 1:30-2:00 - Hard exercises (5-6)
- 2:00-2:30 - Advanced: Experiment + optimize

---

## 💡 Common Mistakes to Avoid

### 1. **Missing Comma Between CTEs**
```sql
-- ❌ Wrong
WITH cte1 AS (SELECT ...)
WITH cte2 AS (SELECT ...)  -- Error! Missing comma

-- ✅ Correct
WITH
cte1 AS (SELECT ...),
cte2 AS (SELECT ...)
```

### 2. **Referencing Non-existent CTE**
```sql
-- ❌ Wrong
WITH high_earners AS (SELECT ...)
SELECT * FROM high_earners, low_earners  -- low_earners doesn't exist!

-- ✅ Correct
WITH
high_earners AS (SELECT ...),
low_earners AS (SELECT ...)
SELECT * FROM high_earners, low_earners
```

### 3. **Forgetting Main Query**
```sql
-- ❌ Wrong - CTE defined but not used
WITH high_earners AS (
    SELECT * FROM employees WHERE salary > 75000
)
-- Missing: SELECT statement

-- ✅ Correct
WITH high_earners AS (
    SELECT * FROM employees WHERE salary > 75000
)
SELECT * FROM high_earners;
```

### 4. **Infinite Recursion**
```sql
-- ❌ Wrong - Will loop forever
WITH RECURSIVE infinite AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM infinite  -- Never stops!
)
SELECT * FROM infinite;

-- ✅ Correct - Has termination condition
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 10  -- Stops at 10
)
SELECT * FROM numbers;
```

---

## 🔗 Resources

- Full CTE Guide: `guides/sql_cte_practice_guide.md`
- Setup SQL: `sql/cte/setup.sql`
- PostgreSQL CTE Docs: https://www.postgresql.org/docs/current/queries-with.html

---

## ✅ Practice Checklist

- [ ] Setup database with `setup.sql`
- [ ] Verified all tables created successfully
- [ ] Completed Exercise 1 (High Earners)
- [ ] Completed Exercise 2 (Customer Orders)
- [ ] Completed Exercise 3 (Department Analysis)
- [ ] Completed Exercise 4 (Regional Sales)
- [ ] Completed Exercise 5 (Project Hierarchy)
- [ ] Completed Exercise 6 (Management Chain)
- [ ] Wrote at least 3 CTEs without looking at solutions
- [ ] Understand when to use CTEs vs subqueries

Good luck! 🚀
