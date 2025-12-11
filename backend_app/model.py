# Database Model
from sqlalchemy import  Column, Integer, Float, Boolean, String, Date
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_type = Column(String)
    amount = Column(Float)
    merchant_category = Column(String)
    # lon = Column(Float)
    # hour_of_day = Column(Integer)
    # transaction_ratio = Column(Float)
    is_fraudulent = Column(String)
    prediction_prob = Column(Float)
    created_at = Column(Date)
    transaction_id = Column(String, nullable=True)
    user_id = Column(String, nullable=True)
    
