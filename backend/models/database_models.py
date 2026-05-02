from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.database import Base


class KycSession(Base):
    __tablename__ = "kyc_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(64), default="STARTED", nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    document_path: Mapped[str | None] = mapped_column(Text)
    id_face_path: Mapped[str | None] = mapped_column(Text)
    selfie_path: Mapped[str | None] = mapped_column(Text)
    liveness_video_path: Mapped[str | None] = mapped_column(Text)

    first_name: Mapped[str | None] = mapped_column(String(128))
    last_name: Mapped[str | None] = mapped_column(String(128))
    cnp: Mapped[str | None] = mapped_column(String(32))
    series_number: Mapped[str | None] = mapped_column(String(32))
    sex: Mapped[str | None] = mapped_column(String(8))
    nationality: Mapped[str | None] = mapped_column(String(64))
    address: Mapped[str | None] = mapped_column(Text)
    valid_from: Mapped[str | None] = mapped_column(String(32))
    valid_until: Mapped[str | None] = mapped_column(String(32))
    raw_ocr_text: Mapped[str | None] = mapped_column(Text)

    liveness_passed: Mapped[bool | None] = mapped_column(Boolean)
    face_match_distance: Mapped[float | None] = mapped_column(Float)
    face_match_decision: Mapped[str | None] = mapped_column(String(64))
    final_decision: Mapped[str | None] = mapped_column(String(64))
    
    # Audit trail for face matching (stores both gate and final thresholds used)
    selfie_gate_distance: Mapped[float | None] = mapped_column(Float)
    selfie_gate_decision: Mapped[str | None] = mapped_column(String(64))
    final_face_match_distance: Mapped[float | None] = mapped_column(Float)
    final_face_match_decision: Mapped[str | None] = mapped_column(String(64))

    # Security hardening fields
    security_fail_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reject_reason: Mapped[str | None] = mapped_column(String(128))
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("kyc_sessions.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class EmbeddingRecord(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("kyc_sessions.id"), index=True, nullable=False)
    embedding_type: Mapped[str] = mapped_column(String(32), nullable=False)
    embedding_vector: Mapped[dict | list | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
