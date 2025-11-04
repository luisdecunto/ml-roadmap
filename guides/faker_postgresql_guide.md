# Generate 1M+ Rows with Python Faker & Load to PostgreSQL

**Time:** 1-2 hours
**Difficulty:** Intermediate
**Prerequisites:** Python basics, PostgreSQL installed

Learn to generate realistic test data at scale using Python's Faker library and efficiently load it into PostgreSQL.

---

## 📚 Why Generate Test Data?

- **Performance testing** - Test query performance on large datasets
- **Development** - Realistic data for local testing
- **Learning SQL optimization** - Practice indexing and query tuning
- **ETL practice** - Learn batch loading and data pipeline concepts

---

## 🎯 Part 1: Setup (15 min)

### Install Required Libraries

```bash
pip install faker psycopg2-binary pandas tqdm
```

**Libraries:**
- `faker` - Generate fake data
- `psycopg2` - PostgreSQL adapter for Python
- `pandas` - Data manipulation (optional, for CSV export)
- `tqdm` - Progress bars

### Verify PostgreSQL Connection

```python
import psycopg2

# Test connection
try:
    conn = psycopg2.connect(
        host="localhost",
        database="ml_roadmap",  # Your database name
        user="postgres",         # Your username
        password="your_password" # Your password
    )
    print("✓ Connection successful!")
    conn.close()
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

---

## 💻 Part 2: Create Database Schema (10 min)

### Design a Realistic Schema

We'll create an e-commerce database with customers, orders, and products.

```python
import psycopg2

def create_schema(conn):
    """Create tables for our fake data"""
    cursor = conn.cursor()

    # Drop tables if they exist (careful in production!)
    cursor.execute("""
        DROP TABLE IF EXISTS order_items CASCADE;
        DROP TABLE IF EXISTS orders CASCADE;
        DROP TABLE IF EXISTS customers CASCADE;
        DROP TABLE IF EXISTS products CASCADE;
    """)

    # Customers table
    cursor.execute("""
        CREATE TABLE customers (
            customer_id SERIAL PRIMARY KEY,
            first_name VARCHAR(100),
            last_name VARCHAR(100),
            email VARCHAR(255) UNIQUE,
            phone VARCHAR(20),
            address TEXT,
            city VARCHAR(100),
            state VARCHAR(50),
            zip_code VARCHAR(10),
            country VARCHAR(100),
            registration_date DATE,
            date_of_birth DATE,
            is_active BOOLEAN DEFAULT TRUE
        )
    """)

    # Products table
    cursor.execute("""
        CREATE TABLE products (
            product_id SERIAL PRIMARY KEY,
            product_name VARCHAR(255),
            category VARCHAR(100),
            brand VARCHAR(100),
            price DECIMAL(10, 2),
            cost DECIMAL(10, 2),
            stock_quantity INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Orders table
    cursor.execute("""
        CREATE TABLE orders (
            order_id SERIAL PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(customer_id),
            order_date TIMESTAMP,
            total_amount DECIMAL(10, 2),
            status VARCHAR(50),
            payment_method VARCHAR(50),
            shipping_address TEXT
        )
    """)

    # Order items table
    cursor.execute("""
        CREATE TABLE order_items (
            order_item_id SERIAL PRIMARY KEY,
            order_id INTEGER REFERENCES orders(order_id),
            product_id INTEGER REFERENCES products(product_id),
            quantity INTEGER,
            unit_price DECIMAL(10, 2),
            discount DECIMAL(5, 2) DEFAULT 0.00
        )
    """)

    conn.commit()
    print("✓ Schema created successfully!")

# Usage
conn = psycopg2.connect(...)
create_schema(conn)
conn.close()
```

---

## 🏭 Part 3: Generate Fake Data (30 min)

### Basic Faker Usage

```python
from faker import Faker
import random

fake = Faker()

# Generate sample data
print(fake.name())              # "John Smith"
print(fake.email())             # "john.smith@example.com"
print(fake.address())           # "123 Main St, New York, NY 10001"
print(fake.date_between(start_date='-2y', end_date='today'))  # Random date
print(fake.random_int(min=1, max=100))  # Random integer
```

### Generate 1M Customers

```python
from faker import Faker
from tqdm import tqdm
import psycopg2
from psycopg2.extras import execute_batch

def generate_customers(n=1_000_000, batch_size=10_000):
    """Generate n customers and insert in batches"""
    fake = Faker()

    conn = psycopg2.connect(
        host="localhost",
        database="ml_roadmap",
        user="postgres",
        password="your_password"
    )
    cursor = conn.cursor()

    print(f"Generating {n:,} customers...")

    customers = []
    for i in tqdm(range(n), desc="Customers"):
        customer = (
            fake.first_name(),
            fake.last_name(),
            fake.email(),
            fake.phone_number(),
            fake.street_address(),
            fake.city(),
            fake.state(),
            fake.zipcode(),
            'USA',
            fake.date_between(start_date='-5y', end_date='today'),
            fake.date_of_birth(minimum_age=18, maximum_age=90),
            random.choice([True, True, True, False])  # 75% active
        )
        customers.append(customer)

        # Insert in batches
        if len(customers) >= batch_size:
            execute_batch(cursor, """
                INSERT INTO customers (
                    first_name, last_name, email, phone, address,
                    city, state, zip_code, country, registration_date,
                    date_of_birth, is_active
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, customers)
            conn.commit()
            customers = []

    # Insert remaining
    if customers:
        execute_batch(cursor, """
            INSERT INTO customers (
                first_name, last_name, email, phone, address,
                city, state, zip_code, country, registration_date,
                date_of_birth, is_active
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, customers)
        conn.commit()

    cursor.close()
    conn.close()

    print(f"✓ Inserted {n:,} customers!")

# Run it!
generate_customers(n=1_000_000, batch_size=10_000)
```

**Performance Tips:**
- Use `execute_batch()` instead of individual inserts (100x faster!)
- Commit in batches, not every row
- Drop indexes before bulk insert, recreate after
- Use `COPY` command for maximum speed (covered next)

---

## ⚡ Part 4: Ultra-Fast Loading with COPY (20 min)

### Method 1: Generate CSV then COPY

This is the **fastest** way to load data!

```python
import csv
from faker import Faker
from tqdm import tqdm

def generate_customers_csv(n=1_000_000, filename='customers.csv'):
    """Generate customers to CSV file"""
    fake = Faker()

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'first_name', 'last_name', 'email', 'phone', 'address',
            'city', 'state', 'zip_code', 'country', 'registration_date',
            'date_of_birth', 'is_active'
        ])

        # Write rows
        for _ in tqdm(range(n), desc="Generating CSV"):
            writer.writerow([
                fake.first_name(),
                fake.last_name(),
                fake.email(),
                fake.phone_number(),
                fake.street_address().replace('\n', ', '),  # Remove newlines
                fake.city(),
                fake.state(),
                fake.zipcode(),
                'USA',
                fake.date_between(start_date='-5y', end_date='today'),
                fake.date_of_birth(minimum_age=18, maximum_age=90),
                random.choice([True, True, True, False])
            ])

    print(f"✓ Generated {filename} with {n:,} rows")

def load_csv_to_postgres(filename='customers.csv'):
    """Load CSV into PostgreSQL using COPY"""
    conn = psycopg2.connect(...)
    cursor = conn.cursor()

    with open(filename, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)

        # COPY command - ultra fast!
        cursor.copy_expert("""
            COPY customers (
                first_name, last_name, email, phone, address,
                city, state, zip_code, country, registration_date,
                date_of_birth, is_active
            ) FROM STDIN WITH CSV
        """, f)

    conn.commit()
    cursor.close()
    conn.close()

    print(f"✓ Loaded {filename} to PostgreSQL!")

# Usage
generate_customers_csv(n=1_000_000)
load_csv_to_postgres()
```

**Performance:** COPY can load 1M rows in ~30 seconds!

### Method 2: In-Memory CSV with StringIO

Even faster - no disk I/O!

```python
from io import StringIO
import psycopg2
from faker import Faker
from tqdm import tqdm

def generate_and_load_customers(n=1_000_000, batch_size=100_000):
    """Generate and load customers in memory"""
    fake = Faker()
    conn = psycopg2.connect(...)

    total_loaded = 0

    for batch_start in tqdm(range(0, n, batch_size), desc="Batches"):
        batch_end = min(batch_start + batch_size, n)
        batch_n = batch_end - batch_start

        # Create in-memory CSV
        csv_buffer = StringIO()

        for _ in range(batch_n):
            row = [
                fake.first_name(),
                fake.last_name(),
                fake.email(),
                fake.phone_number(),
                fake.street_address().replace('\n', ', '),
                fake.city(),
                fake.state(),
                fake.zipcode(),
                'USA',
                str(fake.date_between(start_date='-5y', end_date='today')),
                str(fake.date_of_birth(minimum_age=18, maximum_age=90)),
                'TRUE' if random.random() < 0.75 else 'FALSE'
            ]
            csv_buffer.write(','.join(row) + '\n')

        # Reset buffer position
        csv_buffer.seek(0)

        # Load to PostgreSQL
        cursor = conn.cursor()
        cursor.copy_expert("""
            COPY customers (
                first_name, last_name, email, phone, address,
                city, state, zip_code, country, registration_date,
                date_of_birth, is_active
            ) FROM STDIN WITH (FORMAT CSV)
        """, csv_buffer)
        conn.commit()
        cursor.close()

        total_loaded += batch_n

    conn.close()
    print(f"✓ Loaded {total_loaded:,} customers!")

# Run it
generate_and_load_customers(n=1_000_000)
```

---

## 📦 Part 5: Generate Related Data (30 min)

### Generate Products

```python
def generate_products(n=10_000):
    """Generate product catalog"""
    fake = Faker()

    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports',
                  'Books', 'Toys', 'Food', 'Beauty', 'Automotive']
    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']

    csv_buffer = StringIO()

    for _ in tqdm(range(n), desc="Products"):
        price = round(random.uniform(5.0, 500.0), 2)
        cost = round(price * random.uniform(0.4, 0.7), 2)

        row = [
            fake.catch_phrase(),  # Product name
            random.choice(categories),
            random.choice(brands),
            str(price),
            str(cost),
            str(random.randint(0, 1000))  # Stock
        ]
        csv_buffer.write(','.join(row) + '\n')

    csv_buffer.seek(0)

    conn = psycopg2.connect(...)
    cursor = conn.cursor()
    cursor.copy_expert("""
        COPY products (product_name, category, brand, price, cost, stock_quantity)
        FROM STDIN WITH (FORMAT CSV)
    """, csv_buffer)
    conn.commit()
    cursor.close()
    conn.close()

    print(f"✓ Generated {n:,} products!")

generate_products(n=10_000)
```

### Generate Orders and Order Items

```python
def generate_orders(n_orders=2_000_000):
    """Generate orders with items"""
    fake = Faker()
    conn = psycopg2.connect(...)
    cursor = conn.cursor()

    # Get customer and product IDs
    cursor.execute("SELECT customer_id FROM customers")
    customer_ids = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT product_id, price FROM products")
    products = cursor.fetchall()

    print(f"Generating {n_orders:,} orders...")

    orders_buffer = StringIO()
    items_buffer = StringIO()

    statuses = ['Pending', 'Processing', 'Shipped', 'Delivered', 'Cancelled']
    payment_methods = ['Credit Card', 'PayPal', 'Debit Card', 'Bank Transfer']

    for order_id in tqdm(range(1, n_orders + 1), desc="Orders"):
        customer_id = random.choice(customer_ids)
        order_date = fake.date_time_between(start_date='-2y', end_date='now')
        status = random.choice(statuses)
        payment = random.choice(payment_methods)

        # Generate 1-5 items per order
        n_items = random.randint(1, 5)
        order_items = random.sample(products, min(n_items, len(products)))

        total_amount = 0
        for product_id, price in order_items:
            quantity = random.randint(1, 3)
            discount = random.choice([0, 0, 0, 5, 10, 15])  # Most no discount
            unit_price = float(price)
            item_total = unit_price * quantity * (1 - discount/100)
            total_amount += item_total

            # Add to items buffer
            items_buffer.write(f"{order_id},{product_id},{quantity},{unit_price},{discount}\n")

        # Add to orders buffer
        orders_buffer.write(
            f"{customer_id},{order_date},{total_amount:.2f},{status},{payment},{fake.address().replace(chr(10), ' ')}\n"
        )

        # Commit in batches
        if order_id % 10000 == 0:
            orders_buffer.seek(0)
            items_buffer.seek(0)

            cursor.copy_expert("""
                COPY orders (customer_id, order_date, total_amount, status, payment_method, shipping_address)
                FROM STDIN WITH (FORMAT CSV)
            """, orders_buffer)

            cursor.copy_expert("""
                COPY order_items (order_id, product_id, quantity, unit_price, discount)
                FROM STDIN WITH (FORMAT CSV)
            """, items_buffer)

            conn.commit()

            orders_buffer = StringIO()
            items_buffer = StringIO()

    # Insert remaining
    if orders_buffer.tell() > 0:
        orders_buffer.seek(0)
        items_buffer.seek(0)

        cursor.copy_expert("""
            COPY orders (customer_id, order_date, total_amount, status, payment_method, shipping_address)
            FROM STDIN WITH (FORMAT CSV)
        """, orders_buffer)

        cursor.copy_expert("""
            COPY order_items (order_id, product_id, quantity, unit_price, discount)
            FROM STDIN WITH (FORMAT CSV)
        """, items_buffer)

        conn.commit()

    cursor.close()
    conn.close()

    print(f"✓ Generated {n_orders:,} orders!")

generate_orders(n_orders=2_000_000)
```

---

## 🔍 Part 6: Add Indexes for Performance (10 min)

After loading data, add indexes:

```python
def create_indexes(conn):
    """Create indexes for query performance"""
    cursor = conn.cursor()

    print("Creating indexes...")

    # Customers indexes
    cursor.execute("CREATE INDEX idx_customers_email ON customers(email)")
    cursor.execute("CREATE INDEX idx_customers_registration ON customers(registration_date)")
    cursor.execute("CREATE INDEX idx_customers_city_state ON customers(city, state)")

    # Products indexes
    cursor.execute("CREATE INDEX idx_products_category ON products(category)")
    cursor.execute("CREATE INDEX idx_products_price ON products(price)")

    # Orders indexes
    cursor.execute("CREATE INDEX idx_orders_customer ON orders(customer_id)")
    cursor.execute("CREATE INDEX idx_orders_date ON orders(order_date)")
    cursor.execute("CREATE INDEX idx_orders_status ON orders(status)")

    # Order items indexes
    cursor.execute("CREATE INDEX idx_order_items_order ON order_items(order_id)")
    cursor.execute("CREATE INDEX idx_order_items_product ON order_items(product_id)")

    conn.commit()
    print("✓ Indexes created!")

conn = psycopg2.connect(...)
create_indexes(conn)
conn.close()
```

---

## 📊 Part 7: Verify and Query (10 min)

### Check Row Counts

```python
def verify_data(conn):
    """Verify data was loaded correctly"""
    cursor = conn.cursor()

    tables = ['customers', 'products', 'orders', 'order_items']

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table}: {count:,} rows")

    cursor.close()

conn = psycopg2.connect(...)
verify_data(conn)
conn.close()
```

### Sample Queries

```sql
-- Top 10 customers by total spending
SELECT
    c.customer_id,
    c.first_name || ' ' || c.last_name AS name,
    COUNT(o.order_id) AS order_count,
    SUM(o.total_amount) AS total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, name
ORDER BY total_spent DESC
LIMIT 10;

-- Monthly sales trends
SELECT
    DATE_TRUNC('month', order_date) AS month,
    COUNT(*) AS orders,
    SUM(total_amount) AS revenue
FROM orders
GROUP BY month
ORDER BY month;

-- Best-selling products
SELECT
    p.product_name,
    p.category,
    SUM(oi.quantity) AS units_sold,
    SUM(oi.quantity * oi.unit_price) AS revenue
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name, p.category
ORDER BY units_sold DESC
LIMIT 20;
```

---

## 🎯 Complete Script

```python
#!/usr/bin/env python3
"""
Generate 1M+ rows of fake data and load to PostgreSQL
"""

import psycopg2
from faker import Faker
from tqdm import tqdm
from io import StringIO
import random
import time

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'ml_roadmap',
    'user': 'postgres',
    'password': 'your_password'
}

def main():
    print("="*60)
    print("FAKE DATA GENERATOR FOR POSTGRESQL")
    print("="*60)

    start_time = time.time()

    # Connect
    conn = psycopg2.connect(**DB_CONFIG)

    # Create schema
    print("\n1. Creating schema...")
    create_schema(conn)

    # Generate data
    print("\n2. Generating customers...")
    generate_and_load_customers(n=1_000_000)

    print("\n3. Generating products...")
    generate_products(n=10_000)

    print("\n4. Generating orders...")
    generate_orders(n_orders=2_000_000)

    # Create indexes
    print("\n5. Creating indexes...")
    create_indexes(conn)

    # Verify
    print("\n6. Verifying data...")
    verify_data(conn)

    conn.close()

    elapsed = time.time() - start_time
    print(f"\n✓ Complete! Total time: {elapsed/60:.1f} minutes")

if __name__ == '__main__':
    main()
```

---

## ✅ Summary

You've learned to:
- ✓ Generate realistic fake data with Faker
- ✓ Use COPY for ultra-fast loading (30 seconds for 1M rows!)
- ✓ Create relational data (customers → orders → items)
- ✓ Add indexes for query performance
- ✓ Verify and query large datasets

**Performance Benchmark:**
- 1M customers: ~30 seconds
- 10K products: ~2 seconds
- 2M orders + 5M items: ~2 minutes
- Total: ~3 minutes for 8M+ rows!

---

## 🚀 Next Steps

1. Practice query optimization on your test data
2. Learn about partitioning for even larger datasets (10M+ rows)
3. Explore PostgreSQL EXPLAIN ANALYZE for performance tuning
4. Try generating time-series data for analytics practice

Now you can practice SQL performance tuning on realistic, large-scale data! 📊
