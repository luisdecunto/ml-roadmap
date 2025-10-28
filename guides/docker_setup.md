# Docker Setup Guide

**Time:** 45-60 minutes
**Difficulty:** Beginner-Intermediate

## Prerequisites
- Windows 10/11 (64-bit, Pro/Enterprise/Education), macOS, or Linux
- 4GB RAM minimum (8GB recommended)
- Admin/sudo access
- Virtualization enabled in BIOS (Windows)

---

## Step 1: Check System Requirements

### Windows:
- Windows 10/11 64-bit: Pro, Enterprise, or Education (Build 19041 or higher)
- **Home edition:** Can use Docker Desktop with WSL 2
- Enable virtualization in BIOS (usually enabled by default)
- Check: Run `systeminfo` in CMD and look for "Hyper-V Requirements"

### macOS:
- macOS 11 or newer
- Apple silicon (M1/M2) or Intel processor

### Linux:
- 64-bit kernel and CPU
- KVM virtualization support
- GNOME or KDE desktop environment

---

## Step 2: Install Docker Desktop

### Windows:

1. **Enable WSL 2 (Required for Home edition, recommended for all):**
   ```powershell
   # Run PowerShell as Administrator
   wsl --install
   # Restart computer
   ```

2. **Download Docker Desktop:**
   - Visit [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
   - Click "Download for Windows"
   - Run `Docker Desktop Installer.exe`

3. **Installation:**
   - Check "Use WSL 2 instead of Hyper-V" (recommended)
   - Click "OK" and let it install
   - Restart computer when prompted

4. **First Launch:**
   - Accept terms
   - Skip sign-in (optional but recommended to create free account)
   - Complete tutorial (optional)

### macOS:

```bash
# Option 1: Download from website
# Visit docker.com/products/docker-desktop and download

# Option 2: Using Homebrew
brew install --cask docker

# Launch Docker Desktop from Applications
```

### Linux (Ubuntu/Debian):

```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Update apt
sudo apt-get update

# Install dependencies
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add your user to docker group (avoid sudo)
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

---

## Step 3: Verify Installation

```bash
# Check Docker version
docker --version
# Should output: Docker version 24.x.x, build xxxxx

# Check if Docker daemon is running
docker info

# Run hello-world container (this pulls and runs a test image)
docker run hello-world

# You should see:
# "Hello from Docker!"
# "This message shows that your installation appears to be working correctly."
```

---

## Step 4: Complete Docker Getting Started Tutorial

### Option 1: In-App Tutorial
1. Open Docker Desktop
2. Click "Start" under "Getting Started" tutorial
3. Follow the interactive guide

### Option 2: Command Line Tutorial
```bash
# Run the getting started container
docker run -d -p 80:80 docker/getting-started

# Open browser to http://localhost
# Follow the tutorial
```

**Key Concepts to Learn:**
- What is a container?
- What is an image?
- Dockerfile basics
- docker build, run, stop, rm commands

---

## Step 5: Docker Basics - Essential Commands

```bash
# Images
docker images                    # List all images
docker pull nginx                # Download an image
docker rmi image_name            # Remove an image

# Containers
docker ps                        # List running containers
docker ps -a                     # List all containers (including stopped)
docker run nginx                 # Run a container
docker run -d nginx              # Run in detached mode (background)
docker run -p 8080:80 nginx      # Map port 8080 (host) to 80 (container)
docker stop container_id         # Stop a container
docker start container_id        # Start a stopped container
docker rm container_id           # Remove a container

# Logs & Debugging
docker logs container_id         # View logs
docker exec -it container_id bash # Open terminal inside container

# Clean up
docker system prune              # Remove unused containers, images, networks
docker system prune -a           # Remove ALL unused images
```

---

## Step 6: Your First Dockerfile

Create a directory for testing:
```bash
mkdir docker-test
cd docker-test
```

Create `Dockerfile`:
```dockerfile
# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir flask

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
```

Create `app.py`:
```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello from Docker!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Build and run:
```bash
# Build image
docker build -t my-first-app .

# Run container
docker run -p 5000:5000 my-first-app

# Visit http://localhost:5000 in browser
# You should see "Hello from Docker!"
```

---

## Step 7: Docker Compose Basics

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: myapp
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  pgadmin:
    image: dpage/pgadmin4
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@admin.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"
    depends_on:
      - postgres

volumes:
  postgres_data:
```

Run with Docker Compose:
```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs

# Stop all services
docker-compose down

# Stop and remove volumes (delete data!)
docker-compose down -v
```

Access services:
- **PostgreSQL:** localhost:5432
- **pgAdmin:** http://localhost:5050 (login: admin@admin.com / admin)

---

## Step 8: Containerize Your PostgreSQL Database

Create `Dockerfile.postgres`:
```dockerfile
FROM postgres:15

# Copy initialization scripts
COPY init.sql /docker-entrypoint-initdb.d/

# These scripts run automatically when container starts first time
```

Create `init.sql`:
```sql
-- Create tables
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);

-- Insert sample data
INSERT INTO users (name, email) VALUES
    ('Alice', 'alice@example.com'),
    ('Bob', 'bob@example.com');
```

Build and run:
```bash
docker build -f Dockerfile.postgres -t my-postgres .
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres my-postgres

# Connect with psql
docker exec -it <container_id> psql -U postgres

# Query your data
SELECT * FROM users;
```

---

## Verification Checklist

- [ ] Docker Desktop installed and running
- [ ] `docker --version` works
- [ ] Successfully ran `hello-world` container
- [ ] Completed Getting Started tutorial
- [ ] Created and ran a custom Dockerfile
- [ ] Created and ran docker-compose.yml with multiple services
- [ ] Can connect to containerized PostgreSQL

---

## Common Issues

### Issue: "Docker daemon is not running"
- **Windows/Mac:** Open Docker Desktop application
- **Linux:** `sudo systemctl start docker`

### Issue: "permission denied" (Linux)
- **Solution:** Add user to docker group:
  ```bash
  sudo usermod -aG docker $USER
  # Log out and back in
  ```

### Issue: WSL 2 installation failed (Windows)
- **Solution:**
  1. Enable virtualization in BIOS
  2. Run: `dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart`
  3. Run: `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart`
  4. Restart computer
  5. Run: `wsl --set-default-version 2`

### Issue: Port already in use
- **Solution:** Check what's using the port:
  ```bash
  # Windows
  netstat -ano | findstr :5432

  # Mac/Linux
  lsof -i :5432
  ```

### Issue: "Cannot connect to Docker daemon"
- **Solution:** Restart Docker Desktop or Docker service

---

## Best Practices

1. **Use .dockerignore:**
   ```
   node_modules
   .git
   .env
   __pycache__
   *.pyc
   .DS_Store
   ```

2. **Use specific image versions:**
   ```dockerfile
   FROM python:3.10-slim  # Good
   FROM python:latest     # Bad (unpredictable)
   ```

3. **Minimize layers:**
   ```dockerfile
   # Bad
   RUN apt-get update
   RUN apt-get install -y python3

   # Good
   RUN apt-get update && apt-get install -y python3
   ```

4. **Clean up in same layer:**
   ```dockerfile
   RUN apt-get update && \
       apt-get install -y python3 && \
       rm -rf /var/lib/apt/lists/*
   ```

---

## Next Steps

1. Containerize your ML model API
2. Create multi-container applications with Docker Compose
3. Learn about Docker volumes for data persistence
4. Explore Docker Hub for pre-built images
5. Learn Docker networking basics

---

## Resources

- [Docker Official Docs](https://docs.docker.com/)
- [Docker Getting Started Guide](https://docs.docker.com/get-started/)
- [Docker Compose Docs](https://docs.docker.com/compose/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker Hub](https://hub.docker.com/) - Find pre-built images
