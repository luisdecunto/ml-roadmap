# PostgreSQL Setup Guide

**Time:** 30-45 minutes
**Difficulty:** Beginner

## Prerequisites
- Windows 10/11, macOS, or Linux
- Admin/sudo access
- 500MB free disk space

---

## Step 1: Download PostgreSQL

1. Visit [postgresql.org/download](https://www.postgresql.org/download/)
2. Select your operating system
3. Download **PostgreSQL 15** or later (recommended: latest stable version)

---

## Step 2: Install PostgreSQL

### Windows:
1. Run the downloaded `.exe` installer
2. Click "Next" through the wizard
3. **Important:** When asked for a password for the `postgres` user:
   - Choose a password you'll remember (e.g., `postgres123`)
   - **Write it down!** You'll need this constantly
4. Default port: `5432` (keep default unless it conflicts)
5. Locale: Default (or your preferred)
6. Uncheck "Launch Stack Builder" at the end
7. Click "Finish"

### macOS:
```bash
# Using Homebrew (recommended)
brew install postgresql@15

# Start PostgreSQL service
brew services start postgresql@15

# Set password for postgres user
psql postgres
ALTER USER postgres PASSWORD 'your_password';
\q
```

### Linux (Ubuntu/Debian):
```bash
# Update package list
sudo apt update

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Start service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Set password
sudo -u postgres psql
ALTER USER postgres PASSWORD 'your_password';
\q
```

---

## Step 3: Install pgAdmin 4 (GUI Tool)

### Windows:
- Already installed with PostgreSQL installer

### macOS:
```bash
brew install --cask pgadmin4
```

### Linux:
```bash
# Add repository
curl -fsS https://www.pgadmin.org/static/packages_pgadmin_org.pub | sudo gpg --dearmor -o /usr/share/keyrings/packages-pgadmin-org.gpg
sudo sh -c 'echo "deb [signed-by=/usr/share/keyrings/packages-pgadmin-org.gpg] https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" > /etc/apt/sources.list.d/pgadmin4.list'

# Install
sudo apt update
sudo apt install pgadmin4
```

---

## Step 4: Test Connection with pgAdmin

1. Open pgAdmin 4
2. Set a master password (for pgAdmin itself)
3. Expand "Servers" in left panel
4. Right-click "Servers" → "Register" → "Server"
5. **General tab:**
   - Name: `Local PostgreSQL`
6. **Connection tab:**
   - Host: `localhost`
   - Port: `5432`
   - Username: `postgres`
   - Password: (the password you set earlier)
   - Save password: ✓
7. Click "Save"
8. You should now see your server connected!

---

## Step 5: Setup Command Line (psql)

### Windows:
1. Add PostgreSQL to PATH:
   - Search "Environment Variables" in Start Menu
   - Click "Environment Variables"
   - Under "System variables", find `Path`
   - Click "Edit" → "New"
   - Add: `C:\Program Files\PostgreSQL\15\bin`
   - Click OK on all dialogs
2. **Restart your terminal/command prompt**
3. Test: `psql --version`

### macOS/Linux:
- Should already be in PATH
- Test: `psql --version`

---

## Step 6: Test psql Command Line

### Windows:
```cmd
psql -U postgres
# Enter your password when prompted
```

### macOS/Linux:
```bash
psql -U postgres
# Enter your password when prompted
```

**You should see:**
```
psql (15.x)
Type "help" for help.

postgres=#
```

---

## Step 7: Create Your First Database

In psql:
```sql
-- Create a practice database
CREATE DATABASE practice_db;

-- List all databases
\l

-- Connect to your new database
\c practice_db

-- Quit psql
\q
```

---

## Step 8: Create a Test Table

```sql
-- Connect to practice_db
psql -U postgres -d practice_db

-- Create a simple table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert test data
INSERT INTO users (name, email) VALUES
    ('Alice', 'alice@example.com'),
    ('Bob', 'bob@example.com'),
    ('Charlie', 'charlie@example.com');

-- Query your data
SELECT * FROM users;

-- You should see your 3 users!
```

---

## Verification Checklist

- [ ] PostgreSQL installed and running
- [ ] pgAdmin 4 installed and can connect to localhost
- [ ] psql command works in terminal
- [ ] Created `practice_db` database
- [ ] Created and queried `users` table successfully

---

## Common Issues

### Issue: "psql: command not found" (Windows)
- **Solution:** PostgreSQL not in PATH. Redo Step 5 and restart terminal.

### Issue: "password authentication failed"
- **Solution:** You're using the wrong password. Reset it:
  ```bash
  # Windows (run as Administrator)
  psql -U postgres
  ALTER USER postgres PASSWORD 'new_password';
  ```

### Issue: Port 5432 already in use
- **Solution:** Another PostgreSQL instance is running.
  - Windows: Check Services, stop PostgreSQL service
  - Mac/Linux: `sudo lsof -i :5432` to find what's using it

### Issue: pgAdmin won't connect
- **Solution:**
  1. Check PostgreSQL service is running
  2. Verify username is `postgres`
  3. Try connecting via psql first to verify password

---

## Next Steps

Once setup is complete:
1. Download sample database: [PostgreSQL Tutorial Sample Database](https://www.postgresqltutorial.com/postgresql-getting-started/postgresql-sample-database/)
2. Practice basic queries from SQLBolt or Mode Analytics
3. Start building your SQL portfolio project!

---

## Useful Commands

```sql
-- List all databases
\l

-- Connect to database
\c database_name

-- List all tables in current database
\dt

-- Describe table structure
\d table_name

-- Show current database
SELECT current_database();

-- Quit psql
\q
```

---

## Resources

- [PostgreSQL Official Docs](https://www.postgresql.org/docs/)
- [PostgreSQL Tutorial](https://www.postgresqltutorial.com/)
- [pgAdmin 4 Documentation](https://www.pgadmin.org/docs/)
