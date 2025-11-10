-- SQL CTE Practice - Complete Setup Script
-- This script creates sample tables for practicing Common Table Expressions (CTEs)
-- Copy and paste this entire script into your PostgreSQL/SQL client

-- ============================================================================
-- 1. EMPLOYEES TABLE
-- ============================================================================
DROP TABLE IF EXISTS employees CASCADE;

CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    salary DECIMAL(10, 2) NOT NULL,
    department_id INT,
    manager_id INT,
    hire_date DATE,
    email VARCHAR(100)
);

INSERT INTO employees (name, salary, department_id, manager_id, hire_date, email) VALUES
('Alice Johnson', 95000, 1, NULL, '2018-03-15', 'alice.johnson@company.com'),
('Bob Smith', 75000, 1, 1, '2019-05-20', 'bob.smith@company.com'),
('Carol Williams', 85000, 2, 1, '2019-07-10', 'carol.williams@company.com'),
('David Brown', 65000, 2, 3, '2020-01-12', 'david.brown@company.com'),
('Eve Davis', 72000, 3, 1, '2020-06-05', 'eve.davis@company.com'),
('Frank Miller', 68000, 3, 5, '2021-02-18', 'frank.miller@company.com'),
('Grace Lee', 95000, 1, 1, '2019-09-22', 'grace.lee@company.com'),
('Henry Wilson', 58000, 2, 3, '2022-01-10', 'henry.wilson@company.com'),
('Iris Taylor', 62000, 3, 5, '2022-03-15', 'iris.taylor@company.com'),
('Jack Anderson', 78000, 1, 1, '2021-04-20', 'jack.anderson@company.com');

-- ============================================================================
-- 2. DEPARTMENTS TABLE
-- ============================================================================
DROP TABLE IF EXISTS departments CASCADE;

CREATE TABLE departments (
    department_id SERIAL PRIMARY KEY,
    department_name VARCHAR(100) NOT NULL,
    location VARCHAR(100),
    budget DECIMAL(12, 2)
);

INSERT INTO departments (department_id, department_name, location, budget) VALUES
(1, 'Engineering', 'San Francisco', 500000),
(2, 'Sales', 'New York', 300000),
(3, 'Marketing', 'Los Angeles', 200000);

-- ============================================================================
-- 3. ORDERS TABLE
-- ============================================================================
DROP TABLE IF EXISTS orders CASCADE;

CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INT NOT NULL,
    employee_id INT,
    order_date DATE NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    product_id INT,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

INSERT INTO orders (customer_id, employee_id, order_date, amount, product_id) VALUES
(101, 1, '2024-01-05', 250.00, 1),
(102, 2, '2024-01-06', 180.00, 2),
(101, 1, '2024-01-08', 320.00, 3),
(103, 3, '2024-01-10', 450.00, 1),
(102, 2, '2024-01-12', 220.00, 2),
(101, 1, '2024-01-15', 150.00, 4),
(104, 4, '2024-01-18', 500.00, 3),
(105, 5, '2024-01-20', 175.00, 1),
(103, 3, '2024-01-22', 280.00, 2),
(102, 2, '2024-01-25', 350.00, 4),
(106, 6, '2024-02-01', 420.00, 3),
(101, 1, '2024-02-05', 190.00, 2),
(104, 4, '2024-02-10', 310.00, 1),
(105, 5, '2024-02-15', 265.00, 4),
(103, 3, '2024-02-18', 380.00, 2);

-- ============================================================================
-- 4. PRODUCTS TABLE
-- ============================================================================
DROP TABLE IF EXISTS products CASCADE;

CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    category VARCHAR(50),
    price DECIMAL(10, 2),
    stock_quantity INT
);

INSERT INTO products (product_id, product_name, category, price, stock_quantity) VALUES
(1, 'Laptop Pro', 'Electronics', 1200.00, 45),
(2, 'Mouse Wireless', 'Electronics', 25.00, 150),
(3, 'USB Cable', 'Electronics', 12.50, 200),
(4, 'Monitor 4K', 'Electronics', 450.00, 30);

-- ============================================================================
-- 5. PROJECT TABLE (For hierarchical queries)
-- ============================================================================
DROP TABLE IF EXISTS projects CASCADE;

CREATE TABLE projects (
    project_id SERIAL PRIMARY KEY,
    project_name VARCHAR(100) NOT NULL,
    parent_project_id INT,
    budget DECIMAL(12, 2),
    start_date DATE,
    end_date DATE,
    FOREIGN KEY (parent_project_id) REFERENCES projects(project_id)
);

INSERT INTO projects (project_id, project_name, parent_project_id, budget, start_date, end_date) VALUES
(1, 'Company Transformation', NULL, 1000000.00, '2023-01-01', '2025-12-31'),
(2, 'Digital Platform', 1, 500000.00, '2023-06-01', '2024-12-31'),
(3, 'Mobile App', 2, 200000.00, '2023-09-01', '2024-06-30'),
(4, 'Web Portal', 2, 150000.00, '2023-08-01', '2024-08-31'),
(5, 'Data Analytics', 1, 300000.00, '2024-01-01', '2025-06-30'),
(6, 'Cloud Migration', 5, 150000.00, '2024-03-01', '2025-03-31');

-- ============================================================================
-- 6. PROJECT_ASSIGNMENTS TABLE
-- ============================================================================
DROP TABLE IF EXISTS project_assignments CASCADE;

CREATE TABLE project_assignments (
    assignment_id SERIAL PRIMARY KEY,
    project_id INT NOT NULL,
    employee_id INT NOT NULL,
    hours_allocated DECIMAL(5, 2),
    assignment_date DATE,
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

INSERT INTO project_assignments (project_id, employee_id, hours_allocated, assignment_date) VALUES
(2, 1, 40.00, '2023-06-01'),
(2, 2, 30.00, '2023-06-01'),
(3, 2, 35.00, '2023-09-01'),
(3, 7, 40.00, '2023-09-01'),
(4, 3, 25.00, '2023-08-01'),
(4, 4, 20.00, '2023-08-01'),
(5, 5, 30.00, '2024-01-01'),
(6, 6, 40.00, '2024-03-01'),
(1, 1, 10.00, '2023-01-01');

-- ============================================================================
-- 7. SALES_RECORDS TABLE (For advanced aggregations)
-- ============================================================================
DROP TABLE IF EXISTS sales_records CASCADE;

CREATE TABLE sales_records (
    record_id SERIAL PRIMARY KEY,
    employee_id INT NOT NULL,
    sale_date DATE NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    region VARCHAR(50),
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

INSERT INTO sales_records (employee_id, sale_date, amount, region) VALUES
(2, '2024-01-05', 5000.00, 'North'),
(2, '2024-01-12', 7500.00, 'North'),
(2, '2024-01-20', 3200.00, 'North'),
(3, '2024-01-08', 6000.00, 'South'),
(3, '2024-01-15', 4500.00, 'South'),
(4, '2024-01-10', 8000.00, 'East'),
(4, '2024-01-18', 6500.00, 'East'),
(5, '2024-01-12', 5500.00, 'West'),
(5, '2024-01-25', 7200.00, 'West'),
(6, '2024-02-05', 4800.00, 'North');

-- ============================================================================
-- VERIFY DATA
-- ============================================================================
-- Run these SELECT statements to verify all tables were created correctly

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
