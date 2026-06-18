from sqlalchemy import Column, Integer, String, Float, DateTime, func, ForeignKey
from backend.database import Base

class GlobalModel(Base):
    __tablename__ = "global_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    version = Column(String(32), nullable=False)
    accuracy = Column(Float)
    round_number = Column(Integer, ForeignKey("rounds.round_number"))
    hf_path = Column(String(256))
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

