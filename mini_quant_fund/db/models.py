from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class PriceHistory(Base):
    __tablename__ = "price_history"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), index=True)
    date = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    adjusted_close = Column(Float)
    data_source = Column(String(50))
    ingestion_ts = Column(DateTime, default=datetime.utcnow)

class Decision(Base):
    __tablename__ = "decisions"
    id = Column(String(50), primary_key=True)
    run_id = Column(String(50), nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp_utc = Column(DateTime, nullable=False)
    decision = Column(String(20), nullable=False) # EXECUTE|HOLD|REJECT|ERROR
    weight = Column(Float)
    price_at_decision = Column(Float)
    mu = Column(Float)
    sigma = Column(Float)
    mu_adjusted = Column(Float)
    data_quality = Column(Float)
    reason_codes = Column(JSON)
    model_versions = Column(JSON)
    config_sha256 = Column(String(64))
    execution_id = Column(String(50), nullable=True)

class DataQuality(Base):
    __tablename__ = "data_quality"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), index=True)
    data_quality_score = Column(Float)
    validation_errors = Column(JSON)
    provider_used = Column(String(50))
    ingestion_ts = Column(DateTime, default=datetime.utcnow)
