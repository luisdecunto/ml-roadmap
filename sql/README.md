# SQL Practice Exercises

Organized collection of SQL exercises for mastering different SQL concepts.

## 📁 Folder Structure

```
sql/
├── README.md (this file)
└── cte/
    ├── setup.sql           # Create all sample tables
    └── PRACTICE_GUIDE.md   # Complete CTE practice guide
```

## 🎯 Topics Covered

### CTE (Common Table Expressions)
- **Location**: `cte/`
- **Time**: 2-3 hours
- **Difficulty**: Intermediate
- **What you'll learn**:
  - Basic CTEs with WITH clause
  - Multiple CTEs in one query
  - Recursive CTEs for hierarchical data
  - When to use CTEs vs subqueries

**Getting Started**:
1. Read `cte/PRACTICE_GUIDE.md`
2. Run `cte/setup.sql` to create sample tables
3. Complete the exercises

---

## 🚀 How to Use

### 1. Choose a Topic
Pick a SQL concept you want to learn (e.g., CTEs)

### 2. Read the Practice Guide
Each topic has a `PRACTICE_GUIDE.md` with:
- Setup instructions
- Theory and examples
- Best practices
- Progressive exercises (Easy → Hard)

### 3. Setup Sample Data
Run the `setup.sql` script in your SQL client:
```bash
# PostgreSQL
psql -U your_user -d your_database -f cte/setup.sql

# Or copy-paste the entire SQL file into your client
```

### 4. Practice Exercises
Start with Easy exercises, then progress to Medium and Hard.

### 5. Check Results
Compare your queries with expected results shown in the guide.

---

## 📊 Sample Tables

The `setup.sql` creates realistic business data:

**EMPLOYEES** (10 rows)
- Salary, department, manager relationships

**DEPARTMENTS** (3 rows)
- Location, budget information

**ORDERS** (15 rows)
- Customer orders with employees

**PRODUCTS** (4 rows)
- Product catalog with inventory

**PROJECTS** (6 rows)
- Hierarchical project structure (perfect for recursive CTEs!)

**PROJECT_ASSIGNMENTS** (9 rows)
- Links employees to projects

**SALES_RECORDS** (10 rows)
- Sales data by region and employee

---

## 💡 Learning Strategy

### Progressive Difficulty
- **Easy**: Single CTEs, simple WHERE clauses
- **Medium**: Multiple CTEs, JOINs, aggregations
- **Hard**: Recursive CTEs, complex hierarchies

### Active Learning
1. Read the example
2. Write the query yourself
3. Run it and verify results
4. Optimize and refactor

### Comparison Approach
- Write with subqueries first
- Then convert to CTEs
- Understand the readability difference

---

## 📝 Best Practices

### SQL Style
- Use MEANINGFUL CTE names: `customer_order_totals` not `temp`
- Format: Each CTE on new lines
- Comments: Explain complex logic
- Indentation: Use consistent spacing

### Testing Strategy
- Test each CTE independently first
- Then build up the full query
- Use LIMIT for large datasets
- Always verify row counts

### Performance
- CTEs are often as efficient as subqueries
- For massive datasets, test both approaches
- Use EXPLAIN to analyze query plans

---

## 🆘 Common Issues

### "Syntax Error" with Multiple CTEs
**Problem**: Missing comma between CTEs
```sql
-- ❌ Wrong
WITH cte1 AS (...)
WITH cte2 AS (...)  -- Error!

-- ✅ Correct
WITH cte1 AS (...),
     cte2 AS (...)
```

### "CTE does not exist"
**Problem**: CTE referenced in wrong scope
```sql
-- Only use CTEs in the query immediately after them
WITH my_cte AS (...)
SELECT * FROM my_cte;  -- ✅ Works

SELECT * FROM my_cte;  -- ❌ Fails - CTE is gone
```

### Recursive CTE Runs Forever
**Problem**: Missing termination condition
```sql
-- ❌ Infinite
WITH RECURSIVE nums AS (
    SELECT 1 n
    UNION ALL
    SELECT n + 1 FROM nums
)

-- ✅ Terminates
WITH RECURSIVE nums AS (
    SELECT 1 n
    UNION ALL
    SELECT n + 1 FROM nums WHERE n < 100
)
```

---

## 📚 Resources

- [PostgreSQL CTE Documentation](https://www.postgresql.org/docs/current/queries-with.html)
- [MySQL CTE Documentation](https://dev.mysql.com/doc/refman/8.0/en/with.html)
- [SQL Server CTE Documentation](https://learn.microsoft.com/en-us/sql/t-sql/queries/with-common-table-expression-transact-sql)

---

## ✅ Checklist for Mastery

- [ ] Understand CTE syntax and structure
- [ ] Write multiple CTEs in one query
- [ ] Convert subqueries to CTEs
- [ ] Use recursive CTEs for hierarchies
- [ ] Know when to use CTEs vs subqueries
- [ ] Optimize CTE performance
- [ ] Handle edge cases (NULL values, empty sets)

---

**Next Topic**: More SQL topics coming soon! (Window Functions, Optimization, etc.)
