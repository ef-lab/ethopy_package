# Database Setup Guide

Ethopy uses MySQL as its database backend, with [DataJoint](https://datajoint.com/) as the data management framework. This guide explains how to set up and manage the database for Ethopy.

## Docker Setup (Recommended)

The recommended way to run the database is using Docker with the official DataJoint MySQL image.

### Prerequisites

1. [Docker](https://docs.docker.com/get-docker/) installed and running
2. Docker Compose (usually included with Docker Desktop)

### Using the Built-in Setup Command

The easiest way to set up the database is using the provided command:

```bash
ethopy-setup-djdocker
```

This command will:
1. Check if Docker is running
2. Create a MySQL container named `ethopy_sql_db`
3. Set up the necessary volumes and configurations
4. Prompt for a root password
5. Start the container

The Docker container uses:
- Image: `datajoint/mysql:5.7` (https://github.com/datajoint/mysql-docker)
- Port: 3306 (standard MySQL port)
- Volume: `./data_ethopy_sql_db:/var/lib/mysql` for persistent data storage

### Manual Docker Setup

If you prefer to set up the container manually:

1. Create a `docker-compose.yaml` file:
```yaml
version: '2.4'
services:
  ethopy_sql_db:
    image: datajoint/mysql:5.7
    environment:
      - MYSQL_ROOT_PASSWORD=your_password
    ports:
      - '3306:3306'
    volumes:
      - ./data_ethopy_sql_db:/var/lib/mysql
```

2. Start the container:
```bash
docker compose up -d
```

### Remote Access

To access the database from another computer:

1. Update the Docker port mapping to allow external access:
```yaml
ports:
  - '0.0.0.0:3306:3306'
```

2. Configure the remote machine's `local_conf.json`:
```json
{
    "dj_local_conf": {
        "database.host": "remote_server_ip",
        "database.user": "root",
        "database.password": "your_password",
        "database.port": 3306
    }
}
```

3. Ensure the Docker host's firewall allows connections on port 3306

## Standalone MySQL Setup

If you prefer not to use Docker, you can install MySQL directly:

### Ubuntu/Debian
```bash
# Install MySQL
sudo apt update
sudo apt install mysql-server

# Start MySQL service
sudo systemctl start mysql
sudo systemctl enable mysql

# Secure the installation
sudo mysql_secure_installation

# Create user and grant privileges
sudo mysql
CREATE USER 'ethopy'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON *.* TO 'ethopy'@'localhost';
FLUSH PRIVILEGES;
```

### macOS (using Homebrew)
```bash
# Install MySQL
brew install mysql

# Start MySQL service
brew services start mysql

# Secure the installation
mysql_secure_installation

# Create user and grant privileges
mysql -u root -p
CREATE USER 'ethopy'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON *.* TO 'ethopy'@'localhost';
FLUSH PRIVILEGES;
```

### Windows
1. Download and install [MySQL Community Server](https://dev.mysql.com/downloads/mysql/)
2. Follow the installation wizard
3. Use MySQL Workbench or command line to create user and grant privileges

## Database Schema Setup

After setting up the MySQL server (either via Docker or standalone), initialize the schemas:

```bash
# Verify database connection
ethopy-db-connection

# Create schemas
ethopy-setup-schema
```

This will create the following schemas:
- `lab_experiments`
- `lab_behavior`
- `lab_stimuli`

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check if MySQL service is running
   - Verify port 3306 is not blocked by firewall
   - Ensure correct host/IP in configuration

2. **Authentication Failed**
   - Verify username and password in `local_conf.json`
   - Check user privileges in MySQL

3. **Docker Container Issues**
   - Check Docker logs: `docker logs ethopy_sql_db`
   - Verify Docker daemon is running
   - Check available disk space for volume

### Useful Commands

```bash
# Check Docker container status
docker ps -a | grep ethopy_sql_db

# View Docker logs
docker logs ethopy_sql_db

# Restart Docker container
docker restart ethopy_sql_db

# Check MySQL status (standalone installation)
sudo systemctl status mysql

# Test MySQL connection
mysql -u root -p
```

## Additional Resources

- [DataJoint Documentation](https://docs.datajoint.org/)
- [MySQL Documentation](https://dev.mysql.com/doc/)
- [Docker MySQL Guide](https://hub.docker.com/_/mysql)
- [DataJoint MySQL Image](https://hub.docker.com/r/datajoint/mysql)