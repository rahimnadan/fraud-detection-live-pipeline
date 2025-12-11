from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import psycopg2


DATABASE_URL = 'postgresql://postgres:root@localhost/transactions'
# DATABASE_URL= f'postgresql://{setting.database_username}:{setting.database_password}@{setting.database_hostname}:{setting.database_port}/{setting.database_name}'

DB_NAME = "transactions"
DB_USER = "postgres"
DB_PASSWORD = "root"
DB_HOST = "localhost"
DB_PORT = "5432"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Step 1: Connect to PostgreSQL to check if the database exists
def create_database():
    connection = psycopg2.connect(
        dbname="postgres", user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
    )
    connection.autocommit = True  # Enable auto-commit mode
    cursor = connection.cursor()

    # Check if database exists
    cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}';")
    exists = cursor.fetchone()

    if not exists:
        cursor.execute(f"CREATE DATABASE {DB_NAME};")  # Create DB if not exists
        print(f"Database '{DB_NAME}' created successfully.")

    cursor.close()
    connection.close()

# Step 2: Call function before creating engine
create_database()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()