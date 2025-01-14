

# Database Setup Guide

## Quick Start
If you just want to get the database running:

1. Install Docker on your system
2. Run the setup command:
```bash
ethopy-setup-djdocker
```
3. Enter a password when prompted
4. Verify the setup:
```bash
ethopy-db-connection
```

That's it! The database is ready to use.

## Detailed Guide

### Prerequisites
- Docker installed and running on your system
- Python 3.8 or higher
- EthoPy package installed

### Understanding the Components
The database setup consists of:
- A MySQL database running in Docker
- DataJoint configuration for EthoPy

### Setup Process

#### 1. Docker Container Setup
```bash
ethopy-setup-djdocker [--mysql-path PATH] [--container-name NAME]
```

Options:
- `--mysql-path`: Custom path to store MySQL data (default: ~/.ethopy/mysql-docker)
- `--container-name`: Custom name for the container (default: ethopy_sql_db)

This command:
- Creates a Docker container with MySQL
- Sets up initial configuration
- Ensures the database is accessible

#### 2. Database Connection Check
```bash
ethopy-db-connection
```
Verifies that EthoPy can connect to the database.

#### 3. Schema Creation
```bash
ethopy-setup-schema
```
Creates all required database tables and structures.

### Troubleshooting

#### Container Won't Start
1. Check Docker status:
```bash
docker ps -a
```
2. Check Docker logs:
```bash
docker logs ethopy_sql_db
```

#### Connection Issues
1. Verify port 3306 is available:
```bash
netstat -an | grep 3306
```
2. Check your local configuration file (~/.ethopy/local_conf.json)

### Advanced Configuration

#### Custom Database Host
Edit your local_conf.json:
```json
{
    "dj_local_conf": {
        "database.host": "YOUR_HOST",
        "database.user": "root",
        "database.password": "YOUR_PASSWORD",
        "database.port": 3306
    }
}
```

#### Multiple Environments
You can run multiple database instances by using different container names:
```bash
ethopy-setup-djdocker --container-name ethopy_dev
ethopy-setup-djdocker --container-name ethopy_prod
```

### Security Considerations
- Use strong passwords
- Don't expose the database port publicly
- Regular backups of ~/.ethopy/mysql-docker/data_*

## Technical Reference

### Command Details

#### setup_dj_docker
Sets up the Docker container with MySQL configured for EthoPy.
```python
setup_dj_docker(mysql_path: Optional[str], container_name: str)
```

#### check_db_connection
Verifies database connectivity.
```python
check_db_connection()
```

### Architecture
The setup follows a three-tier approach:
1. Container Management (Docker)
2. Database Configuration (MySQL)
3. Schema Management (DataJoint)

Each tier is independent and can be modified without affecting the others.