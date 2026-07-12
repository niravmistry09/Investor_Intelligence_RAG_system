from sqlalchemy import text

from database.postgres_sql import get_engine, create_database


def create_tables() -> None:
    engine = get_engine()

    
    setup_vector_extension = "CREATE EXTENSION IF NOT EXISTS vector;"

    query = """
    CREATE TABLE IF NOT EXISTS financial_metrics (
        id SERIAL PRIMARY KEY,
        company VARCHAR(100),
        year VARCHAR(10),
        revenue TEXT,
        net_income TEXT,
        operating_income TEXT,
        cash_flow TEXT,
        total_assets TEXT,
        total_liabilities TEXT,
        risk_factors TEXT,
        growth_drivers TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    with engine.begin() as connection:
       
        connection.execute(text(setup_vector_extension))
        connection.execute(text(query))

    print("Vector extension checked/enabled & financial_metrics table created.")


if __name__ == "__main__":
    create_database()
    create_tables()