import os
from urllib.parse import quote
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def get_engine(database: str | None = None):
    """
    Create PostgreSQL connection engine for local free version (SQLAlchemy).
    """
    # Agar direct DATABASE_URL mil jaye (sabse clean aur best tarika)
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        # Agar url me galti se require reh gaya ho toh disable kar dete hain
        if "sslmode=require" in db_url:
            db_url = db_url.replace("sslmode=require", "sslmode=disable")
        return create_engine(db_url)

    # Fallback to individual variables if DATABASE_URL is missing
    if database is None:
        database = os.getenv("POSTGRES_DATABASE") or os.getenv("DB_NAME", "investor_intelligence")
    
    host = os.getenv("POSTGRES_HOST") or os.getenv("DB_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT") or os.getenv("DB_PORT", "5432")
    user = os.getenv("POSTGRES_USER") or os.getenv("DB_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD") or os.getenv("DB_PASSWORD", "")

    # URL-encode credentials to handle special characters (e.g., @ in password)
    encoded_user = quote(user, safe="")
    encoded_password = quote(password, safe="")

    # ❌ "?sslmode=require" ko hatakar "?sslmode=disable" kar diya local machine ke liye
    connection_string = (
        f"postgresql+psycopg2://"
        f"{encoded_user}:{encoded_password}@{host}:{port}/{database}"
        "?sslmode=disable"
    )

    return create_engine(connection_string)


def create_database() -> None:
    """
    Create the target database if it does not exist using native psycopg2.
    """
    target_db = os.getenv("POSTGRES_DATABASE") or os.getenv("DB_NAME", "investor_intelligence")
    db_url = os.getenv("DATABASE_URL")

    try:
        # Connect to default 'postgres' system database to check or create target database
        if db_url:
            # Agar URL pure main database ka hai, toh bootstrap ke liye default 'postgres' db string nikalte hain
            # Kyunki chalte hue target database ko hum drop/create nahi kar sakte uske andar rehte hue.
            base_url, _ = db_url.split('?')[0].rsplit('/', 1)
            bootstrap_url = f"{base_url}/postgres?sslmode=disable"
            conn = psycopg2.connect(dsn=bootstrap_url)
        else:
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                database="postgres",  # Default system DB to perform creation
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", ""),
                port=os.getenv("DB_PORT", "5432"),
                sslmode="disable"
            )
            
        conn.autocommit = True
        cursor = conn.cursor()
        
        try:
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (target_db,)
            )
            
            if not cursor.fetchone():
                print(f"Database '{target_db}' does not exist. Creating locally...")
                cursor.execute(f"CREATE DATABASE {target_db}")
                print(f"Database '{target_db}' created successfully.")
            else:
                print(f"Database '{target_db}' already exists and is ready.")
        finally:
            cursor.close()
            conn.close()
            
    except Exception as exc:
        print(f"Failed to create database: {exc}")
        raise