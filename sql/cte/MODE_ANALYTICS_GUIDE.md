# Using Mode Analytics for CTE Exercises

**Mode Analytics** is a great platform for practicing SQL with an interactive editor, built-in visualizations, and query sharing features. This guide shows you how to set up and use Mode for your CTE exercises.

## ✅ Why Mode Analytics?

- **Easy Setup**: No installation needed - works entirely in browser
- **MySQL Support**: Supports MySQL databases with sample data
- **Query Editor**: Clean, feature-rich SQL editor with syntax highlighting
- **Results Visualization**: Automatically shows results in table or chart format
- **Sharing**: Can share queries and results with others
- **Account**: You already have an account!

## 📋 Setup Steps

### Step 1: Log In to Mode

1. Go to [app.mode.com](https://app.mode.com)
2. Sign in with your account
3. You'll see your workspace/dashboard

### Step 2: Create a New Report

1. Click **"Create"** button (usually top-right or in sidebar)
2. Select **"Report"** or **"SQL Query"**
3. Name it something like: `CTE Practice - Exercises`
4. Choose or create a workspace

### Step 3: Connect to a Database

Mode provides several sample databases. You have two options:

#### Option A: Use Mode's Sample Database (Recommended for quick start)
- Mode has pre-loaded sample databases
- Skip to Step 4 and paste your setup script

#### Option B: Connect Your Own MySQL Server
1. In your report, look for **"Data Source"** or **"Database"** selector
2. Select **"New Connection"**
3. Choose **"MySQL"**
4. Enter your local MySQL credentials:
   - Host: `localhost` (or your server IP)
   - Port: `3306` (default)
   - Username: `root` (or your username)
   - Password: (your password)
   - Database: `cte_practice`
5. Test connection
6. Save

### Step 4: Load Your Sample Data

#### If using Mode's sample database:

1. In the SQL editor, copy and paste the entire contents of `setup_mysql.sql`
2. Run the script (usually Ctrl+Enter or click Run)
3. The tables will be created in Mode's database

#### If using your own MySQL server:

1. Your database is already set up from earlier (or run setup_mysql.sql if not)
2. Skip to Step 5

### Step 5: Verify Setup

1. In a new query, paste this verification script:

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

2. Run it (Ctrl+Enter)
3. You should see 7 rows with table names and counts:
   - EMPLOYEES: 10
   - DEPARTMENTS: 3
   - ORDERS: 15
   - PRODUCTS: 4
   - PROJECTS: 6
   - PROJECT_ASSIGNMENTS: 9
   - SALES_RECORDS: 10

## 🎯 How to Practice Exercises

### For Each Exercise:

1. **Create a new query** in Mode (or add to same report with a new cell)
2. **Write your SQL** in the editor
3. **Run it** (Ctrl+Enter or Run button)
4. **Review results** in the table below
5. **Compare with expected output** from EXERCISES.md
6. **Refine and iterate**

### Organizing Your Work

**Best Practice**: Create one report with multiple queries (cells)

```
Report: "CTE Practice - Exercises"
├── Cell 1: Exercise 1 - High Earners
├── Cell 2: Exercise 2 - Customer Orders
├── Cell 3: Exercise 3 - Department Analysis
├── Cell 4: Exercise 4 - Regional Sales
├── Cell 5: Exercise 5 - Project Hierarchy
├── Cell 6: Exercise 6 - Management Chain
└── Cells 7-9: Challenge exercises
```

### Helpful Mode Features

**Syntax Highlighting**: Mode auto-highlights SQL keywords, strings, comments

**Auto-complete**: Start typing and Mode suggests table names, columns

**Query History**: Mode saves your query history - easy to go back

**Visualizations**: After running a query, click the chart icon to see different views (table, bar chart, scatter, etc.)

**Sharing**: Click "Share" to get a link to your query (read-only by default)

## 📝 Example: Exercise 1 (High Earners)

### In Mode:

1. Create new cell in your report
2. Write your CTE:

```sql
WITH avg_salary AS (
    SELECT AVG(salary) AS average_salary FROM employees
)
SELECT
    e.name,
    e.salary
FROM employees e
CROSS JOIN avg_salary
WHERE e.salary > avg_salary.average_salary
ORDER BY e.salary DESC;
```

3. Click Run (Ctrl+Enter)
4. Results appear in table below

### Expected Output:
- Alice Johnson - $95,000
- Carol Williams - $85,000
- Grace Lee - $95,000
- Jack Anderson - $78,000

### If your results don't match:
1. Check the expected output in EXERCISES.md
2. Review your CTE logic
3. Test parts of your query independently
4. Use Mode's syntax highlighting to check for typos

## 🐛 Troubleshooting

### "Table not found" error
- Verify you ran the setup script successfully
- Run the verification query from Step 5
- Check that you're in the correct database

### Query runs but returns no results
- Check your WHERE conditions
- Verify you're using correct column names
- Test a simpler SELECT first: `SELECT * FROM employees LIMIT 5;`

### Syntax errors
- Mode highlights syntax errors in red
- Check for missing commas between CTEs
- Verify CTE references exist
- Check parentheses are balanced

### Slow queries (rare)
- For recursive CTEs, start with LIMIT to test:
  ```sql
  WITH RECURSIVE ...
  SELECT * FROM ... LIMIT 10;
  ```

## 🚀 Next Steps After Exercises

1. **Optimize**: Rewrite queries for performance
2. **Add Filters**: Modify exercises with date ranges or thresholds
3. **Combine**: Merge multiple exercises into complex queries
4. **Save Results**: Mode can save query results to datasets
5. **Share**: Share your solutions with others via Mode links

## 💡 Tips

- **Comment your code**: Use `-- comment` in Mode
- **Format for readability**: Mode has a format button
- **Test incrementally**: Build complex CTEs step by step
- **Name CTEs meaningfully**: `customer_totals` not `temp`
- **Save frequently**: Mode auto-saves, but you can manually save versions

## 📚 Resources

- [Mode SQL Editor Guide](https://mode.com/help/articles/sql-tutorial/)
- [CTE Examples in Mode](https://mode.com/sql-tutorial/introduction-to-sql/common-table-expressions/)
- Your local guides: `PRACTICE_GUIDE.md` and `EXERCISES.md`

---

**Ready to start?**

1. Log in to Mode
2. Create a report
3. Paste setup_mysql.sql
4. Start with Exercise 1: High Earners
5. Reference EXERCISES.md for expected outputs

Good luck! 🎯
