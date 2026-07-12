import os
from urllib.parse import quote
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

def get_engine(database: str | None = None):
    """
    Create PostgreSQL connection engine. Optimised for Supabase & Local DBs.
    """
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        # 🔥 Supabase standard / pooled connection configurations
        if "localhost" not in db_url and "127.0.0.1" not in db_url:
            # Agar URL me sslmode nahi hai, toh use configure karein
            if "sslmode=disable" in db_url:
                db_url = db_url.replace("sslmode=disable", "sslmode=require")
            elif "sslmode" not in db_url:
                separator = "&" if "?" in db_url else "?"
                db_url = f"{db_url}{separator}sslmode=require"
            
            # 💡 Supabase supports standard postgres connection pooling parameters
            return create_engine(
                db_url, 
                connect_args={"sslmode": "require"},
                pool_pre_ping=True,  # Supabase stale connections ko drop karne ke liye
                pool_recycle=300     # Connections refresh hote rahenge
            )
        return create_engine(db_url)

    # Fallback to individual variables if DATABASE_URL is missing
    if database is None:
        database = os.getenv("POSTGRES_DATABASE") or os.getenv("DB_NAME", "investor_intelligence")
    
    host = os.getenv("POSTGRES_HOST") or os.getenv("DB_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT") or os.getenv("DB_PORT", "5432")
    user = os.getenv("POSTGRES_USER") or os.getenv("DB_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD") or os.getenv("DB_PASSWORD", "")

    encoded_user = quote(user, safe="")
    encoded_password = quote(password, safe="")

    # Local machine ke liye disable, Supabase ke liye require
    ssl_mode = "disable" if host in ["localhost", "127.0.0.1"] else "require"

    connection_string = (
        f"postgresql+psycopg2://"
        f"{encoded_user}:{encoded_password}@{host}:{port}/{database}"
        f"?sslmode={ssl_mode}"
    )

    if ssl_mode == "require":
        return create_engine(
            connection_string, 
            connect_args={"sslmode": "require"},
            pool_pre_ping=True,
            pool_recycle=300
        )
    return create_engine(connection_string)


def create_database() -> None:
    """
    Create the target database if it does not exist. Gracefully bypasses on Supabase cloud.
    """
    target_db = os.getenv("POSTGRES_DATABASE") or os.getenv("DB_NAME", "investor_intelligence")
    db_url = os.getenv("DATABASE_URL")
    host = os.getenv("POSTGRES_HOST") or os.getenv("DB_HOST", "localhost")

    # 🔥 Supabase Security Guard: Supabase me defaults database 'postgres' hi hota hai,
    # aur aap custom databases programmatic way me create nahi kar sakte (Permission Denied error aayega).
    is_supabase = (db_url and "supabase" in db_url) or (host and "supabase" in host)
    is_cloud = db_url and "localhost" not in db_url and "127.0.0.1" not in db_url

    if is_supabase or is_cloud:
        print("Supabase/Cloud Database detected. Skipping native database creation to avoid permission crash.")
        return

    try:
        if db_url:
            base_url, _ = db_url.split('?')[0].rsplit('/', 1)
            bootstrap_url = f"{base_url}/postgres?sslmode=disable"
            conn = psycopg2.connect(dsn=bootstrap_url)
        else:
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                database="postgres",
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", ""),
                port=os.getenv("DB_PORT", "5432"),
                sslmode="disable"
            )
            
        conn.autocommit = True
        cursor = conn.cursor()
        
        try:
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
        print(f"Failed to create database locally: {exc}")
        if not db_url or "localhost" in db_url:
            raise