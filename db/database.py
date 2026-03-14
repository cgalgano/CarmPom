"""
database.py
-----------
Manages the SQLAlchemy engine and session factory.

Everything that needs to talk to the database imports `get_session` from here.
The database file lives at data/carmpom.db (created automatically on first run).
"""

from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# PROJECT_ROOT is the CarmPom/ directory regardless of where the script is run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "carmpom.db"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def get_engine():
    """
    Create (or reuse) the SQLAlchemy engine.

    - Creates the data/ directory if it doesn't exist yet.
    - Uses SQLite for now. In Phase 7 this will swap to PostgreSQL by
      changing the connection string here — nothing else needs to change.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection_string = f"sqlite:///{DB_PATH}"
    engine = create_engine(
        connection_string,
        connect_args={"check_same_thread": False},  # Needed for SQLite + multi-threading
        echo=False,  # Set to True to see raw SQL in the terminal (useful for debugging)
    )
    return engine


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

# A single engine is created once when this module is imported.
engine = get_engine()

# SessionLocal is a factory: call SessionLocal() to get a new database session.
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_session():
    """
    Context manager that yields a database session and handles cleanup.

    Usage:
        with get_session() as session:
            results = session.execute(text("SELECT 1")).fetchall()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Base class for ORM models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """
    All ORM table classes in models.py inherit from this Base.
    Calling Base.metadata.create_all(engine) creates every table at once.
    """
    pass


# ---------------------------------------------------------------------------
# Quick connectivity test (run this file directly to verify the DB is reachable)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with engine.connect() as conn:
        result = conn.execute(text("SELECT sqlite_version()"))
        version = result.fetchone()[0]
        print(f"Connected to SQLite {version} at: {DB_PATH}")
