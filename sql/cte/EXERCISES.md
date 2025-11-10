# CTE Exercises - Solutions

Complete your solutions here! Write your SQL queries for each exercise.

---

## Easy Exercises

### Exercise 1: High Earners
**Task**: Find all employees earning more than the average salary using CTE.

**Your Solution:**
```sql
-- Write your CTE here
```

**Expected Output:**
- 4 employees should appear
- Names: Alice Johnson, Carol Williams, Grace Lee, Jack Anderson
- All earning more than average (~73,500)

---

### Exercise 2: Customer Total Orders
**Task**: Calculate total order amount per customer, then find customers with >$500 in orders.

**Your Solution:**
```sql
-- Write your CTE here
```

**Expected Output:**
- Customer 101: $910.00
- Customer 103: $1,110.00
- Customer 104: $810.00

---

## Medium Exercises

### Exercise 3: Department Salary Analysis
**Task**: Create two CTEs:
1. `dept_stats` - average salary per department
2. `above_avg_emps` - employees earning above their dept average

Then show all employees with their salary and their department average.

**Your Solution:**
```sql
-- Write your CTEs here
```

**Expected Output:**
- Show which employees earn above their department's average
- Include: employee name, salary, department, department average

**Example:**
```
Alice Johnson - $95,000 - Engineering (avg: $85,500)
Carol Williams - $85,000 - Sales (avg: $71,666)
...
```

---

### Exercise 4: Sales by Region - Top 3 Per Region
**Task**: Find top 3 salespeople (by total sales) in each region using:
1. `regional_sales` - total sales per employee per region
2. `regional_ranking` - rank employees within each region

**Your Solution:**
```sql
-- Write your CTEs here
```

**Expected Output:**
```
Region    Employee          Total Sales  Rank
North     Bob Smith         $15,700      1
North     Frank Miller      $4,800       2
South     Carol Williams    $10,500      1
East      David Brown       $14,500      1
West      Eve Davis         $12,700      1
```

---

## Hard Exercises

### Exercise 5: Project Hierarchy
**Task**: Display the complete project hierarchy with indentation showing parent-child relationships.

Use RECURSIVE CTE with:
- Base case: top-level projects (parent_project_id IS NULL)
- Recursive case: child projects with indentation

**Your Solution:**
```sql
-- Write your RECURSIVE CTE here
```

**Expected Output:**
```
Company Transformation
  Digital Platform
    Mobile App
    Web Portal
  Data Analytics
    Cloud Migration
```

---

### Exercise 6: Employee Management Chain
**Task**: Find the complete management chain from each employee up to the CEO.

Use RECURSIVE CTE to:
- Start with leaf employees
- Move up through managers
- Stop at CEO (manager_id IS NULL)

**Your Solution:**
```sql
-- Write your RECURSIVE CTE here
```

**Expected Output:**
```
Employee               Management Chain
David Brown           → Carol Williams → Alice Johnson (CEO)
Frank Miller          → Eve Davis → Alice Johnson (CEO)
Henry Wilson          → Carol Williams → Alice Johnson (CEO)
...
```

---

## 🎯 Challenge Exercises

### Challenge 1: Multi-Level Project Budget Rollup
**Task**: Calculate total budget for each project including all child projects (recursive sum).

**Your Solution:**
```sql
-- Write your query here
```

---

### Challenge 2: Employee Salary Percentile
**Task**: Show each employee's salary percentile within their department.

Example: If Alice earns more than 80% of engineers, her percentile is 80.

**Hint**: Use multiple CTEs with aggregations.

**Your Solution:**
```sql
-- Write your query here
```

---

### Challenge 3: Department Comparison
**Task**: For each department, show:
- Average salary
- Highest paid employee
- Lowest paid employee
- Salary range (max - min)

**Your Solution:**
```sql
-- Write your query here
```

---

## 📊 Verification Queries

Use these to verify your table setup:

```sql
-- Check employees
SELECT COUNT(*) as emp_count FROM employees;  -- Should be 10

-- Check orders
SELECT COUNT(*) as order_count FROM orders;   -- Should be 15

-- Check total customers
SELECT COUNT(DISTINCT customer_id) as customers FROM orders;  -- Should be 6

-- Check projects
SELECT COUNT(*) as project_count FROM projects;  -- Should be 6

-- Check average salary
SELECT AVG(salary) as avg_salary FROM employees;  -- Should be ~73,500
```

---

## 💡 Tips for Solving

### For Easy Exercises:
- Start with a simple SELECT
- Wrap it in WITH ... AS
- Then use the CTE in main query

### For Medium Exercises:
- Build first CTE
- Test it independently
- Build second CTE
- Build third CTE
- Join them all together

### For Hard/Recursive Exercises:
- Test the base case first
- Make sure it returns correct rows
- Add the UNION ALL
- Add the recursive part slowly
- Add the WHERE clause for termination
- Test with LIMIT first

### General Tips:
1. Comment your code
2. Use meaningful CTE names
3. Format for readability
4. Test each CTE independently
5. Start simple, add complexity

---

## 📈 Progress Tracking

- [ ] Exercise 1: High Earners
- [ ] Exercise 2: Customer Orders
- [ ] Exercise 3: Department Analysis
- [ ] Exercise 4: Regional Sales
- [ ] Exercise 5: Project Hierarchy
- [ ] Exercise 6: Management Chain
- [ ] Challenge 1: Budget Rollup
- [ ] Challenge 2: Salary Percentile
- [ ] Challenge 3: Department Comparison

---

## 🚀 Next Steps After Completing Exercises

1. **Optimize**: Rewrite your queries for performance
2. **Add Conditions**: Modify exercises to filter by date range, threshold, etc.
3. **Combine**: Merge multiple exercises into one complex query
4. **Real Data**: Try these patterns on actual work data
5. **Teach**: Explain your solutions to someone else

Good luck! 🎯
