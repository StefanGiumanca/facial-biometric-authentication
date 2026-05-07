import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, sessionmaker


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/visionauth",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def init_db():
    from backend.models import database_models  # noqa: F401

    try:
        Base.metadata.create_all(bind=engine)
        ensure_kyc_session_schema()
        print("[DB] Tables are ready")
    except SQLAlchemyError as error:
        print(f"[DB] Could not initialize database: {error}")


def ensure_kyc_session_schema():
    """Add recovered MVP columns to an existing local database table."""
    inspector = inspect(engine)
    if not inspector.has_table("kyc_sessions"):
        return

    existing_columns = {column["name"] for column in inspector.get_columns("kyc_sessions")}
    columns_to_add = {
        "document_path": "TEXT",
        "id_face_path": "TEXT",
        "selfie_path": "TEXT",
        "liveness_video_path": "TEXT",
        "first_name": "VARCHAR(128)",
        "last_name": "VARCHAR(128)",
        "cnp": "VARCHAR(32)",
        "series_number": "VARCHAR(32)",
        "sex": "VARCHAR(8)",
        "nationality": "VARCHAR(64)",
        "address": "TEXT",
        "valid_from": "VARCHAR(32)",
        "valid_until": "VARCHAR(32)",
        "raw_ocr_text": "TEXT",
        "liveness_passed": "BOOLEAN",
        "face_match_distance": "FLOAT",
        "face_match_decision": "VARCHAR(64)",
        "final_decision": "VARCHAR(64)",
        "selfie_gate_distance": "FLOAT",
        "selfie_gate_decision": "VARCHAR(64)",
        "final_face_match_distance": "FLOAT",
        "final_face_match_decision": "VARCHAR(64)",
        "security_fail_count": "INTEGER NOT NULL DEFAULT 0",
        "reject_reason": "VARCHAR(128)",
        "locked_at": "TIMESTAMP WITH TIME ZONE",
    }

    with engine.begin() as connection:
        for column_name, column_type in columns_to_add.items():
            if column_name not in existing_columns:
                connection.execute(
                    text(f"ALTER TABLE kyc_sessions ADD COLUMN {column_name} {column_type}")
                )
