from sqlalchemy import Column, Integer, String, Float, DateTime, func, JSON, UniqueConstraint, ForeignKey, CheckConstraint
from backend.database import Base

class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(String(64), ForeignKey("devices.client_id"), nullable=False)
    round_number = Column(Integer, ForeignKey("rounds.round_number"), nullable=False)
    epoch = Column(Integer)
    avg_loss = Column(Float)
    accuracy = Column(Float)
    epsilon_spent = Column(Float)
    delta = Column(Float)
    data_size = Column(Integer)
    partition_info = Column(JSON)
    training_duration_s = Column(Float)
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    reported_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('client_id', 'round_number', 'epoch', name='uix_client_round_epoch'),
        CheckConstraint('epsilon_spent >= 0', name='check_epsilon_spent_positive')
    )
