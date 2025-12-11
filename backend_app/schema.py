from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional
from datetime import date
from enum import Enum

class TransactionType(str, Enum):
    REFUND = "refund"
    TRANSFER = "transfer"
    PURCHASE = "purchase"
    WITHDRAWAL = "withdrawal"

class MerchantCategory(str, Enum):
    GROCERIES = "groceries"
    BANK = "bank"
    ELECTRONICS = "electronics"
    ATM = "atm"
    RESTAURANT = "restaurant"
    LUXURY_GOODS = "luxury goods"

class TransactionDetails(BaseModel):
    transaction_type: str = Field(..., description="Type of transaction")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_category: str = Field(..., description="Merchant category")
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="User identifier")

    @validator('transaction_type')
    def validate_transaction_type(cls, v):
        if not v or v.lower() not in [t.value for t in TransactionType]:
            raise ValueError(f"Invalid transaction type. Must be one of: {[t.value for t in TransactionType]}")
        return v.lower()

    @validator('merchant_category')
    def validate_merchant_category(cls, v):
        if not v or v.lower() not in [c.value for c in MerchantCategory]:
            raise ValueError(f"Invalid merchant category. Must be one of: {[c.value for c in MerchantCategory]}")
        return v.lower()

# class TransactionResponse(TransactionDetails):
#     id: int
#     is_fraudulent: str
#     prediction_prob: float
#     created_at: date

#     class Config:
#         orm_mode = True  # Enables ORM compatibility
 
class TransactionResponse(BaseModel):
    transaction_type: str
    amount: float
    merchant_category: str
    is_fraudulent: str
    prediction_prob: float
    created_at: date
    transaction_id: str
    user_id: str

    class Config:
        from_attributes = True

class ModelMonitoring(BaseModel):
    date_from: date
    date_to: date

    @validator('date_to')
    def validate_dates(cls, v, values):
        if 'date_from' in values and v < values['date_from']:
            raise ValueError("date_to must be greater than or equal to date_from")
        return v

class Graph(BaseModel):
    graph_data: Dict[str, Any]

    @validator('graph_data')
    def validate_graph_data(cls, v):
        required_keys = ["shap_values", "base_values", "data_values"]
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required key: {key}")
        return v