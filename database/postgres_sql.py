import os
from urllib.parse import quote
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

def get_engine(database: str | None = None):
    """
    Create PostgreSQL connection engine. Supports both Local and Cloud AWS DBs automatically.
    """
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        # 🔥 AWS / क्लाउड के लिए SSL मोड को सुरक्षित तरीके से ऑन रखना अनिवार्य है
        if "localhost" not in db_url and "127.0.0.1" not in db_url:
            # अगर URL में sslmode नहीं है, तो जोड़ें, और अगर 'disable' है तो उसे 'require' करें
            if "sslmode=disable" in db_url:
                db_url = db_url.replace("sslmode=disable", "sslmode=require")
            elif "sslmode" not in db_url:
                separator = "&" if "?" in db_url else "?"
                db_url = f"{db_url}{separator}sslmode=require"
            
            return create_engine(db_url, connect_args={"sslmode": "require"})
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

    # लोकल मशीन के लिए disable रहेगा, लेकिन क्लाउड के लिए ऑटो-डिटेक्ट होगा
    ssl_mode = "disable" if host in ["localhost", "127.0.0.1"] else "require"

    connection_string = (
        f"postgresql+psycopg2://"
        f"{encoded_user}:{encoded_password}@{host}:{port}/{database}"
        f"?sslmode={ssl_mode}"
    )

    if ssl_mode == "require":
        return create_engine(connection_string, connect_args={"sslmode": "require"})
    return create_engine(connection_string)


def create_database() -> None:
    """
    Create the target database if it does not exist. Gracefully bypasses on Cloud environments.
    """
    target_db = os.getenv("POSTGRES_DATABASE") or os.getenv("DB_NAME", "investor_intelligence")
    db_url = os.getenv("DATABASE_URL")

    # 🔥 सुरक्षा जांच: अगर क्लाउड डेटाबेस है, तो हम 'CREATE DATABASE' स्क्रिप्ट को स्किप कर देंगे 
    # ताकि परमिशन एरर की वजह से सर्वर क्रैश न हो।
    if db_url and "localhost" not in db_url and "127.0.0.1" not in db_url:
        print("Cloud Database detected. Skipping native database creation to avoid permission crash.")
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
        # इसे केवल लोकल पर क्रैश होने देंगे, क्लाउड पर नहीं
        if not db_url or "localhost" in db_url:
            raise
