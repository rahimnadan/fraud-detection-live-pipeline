from typing import List
from fastapi import FastAPI, HTTPException, Depends, Security, Request, status
from inference import inference_code
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import uvicorn
import numpy as np
from db_connection import engine, get_db
from sqlalchemy.orm import sessionmaker, Session
from traceback import format_exc
from model import Transaction, Base
from fastapi.security import APIKeyHeader
import os 
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from training_pipeline import run_training_pipeline
import shap 
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse, JSONResponse
import io
from schema import *
import base64
from enum import Enum

plt.switch_backend('Agg')

# Define valid transaction types and merchant categories
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

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["500/minute"],
    storage_uri="memory://",
    enabled=True
)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

load_dotenv()

import joblib

# Save the trained model
dt_model = joblib.load('trained_model_weights/decision_tree_model.pkl')

API_KEY = os.getenv("API_KEY")  # Default value for testing

# Create the database tables
Base.metadata.create_all(bind=engine)

model = load_model("trained_model_weights/latest_fraud_model.keras")
scaler = MinMaxScaler()

# Define API Key Header
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

# API Key Authentication Dependency
async def check_api_key(x_api_key: str = Security(api_key_header)):
    """Validate the API Key"""
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API Key not configured on server"
        )
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key is required"
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return True

@app.get("/")
def read_root():
    return {"Hello": "World"}

def validate_transaction(data: TransactionDetails):
    """Validate transaction data"""
    if not data.transaction_type or not data.amount or not data.merchant_category:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required fields"
        )
    
    # Validate transaction type
    try:
        # Convert to lowercase for validation
        transaction_type = data.transaction_type.lower()
        if transaction_type not in [t.value for t in TransactionType]:
            raise ValueError()
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid transaction type. Must be one of: {[t.value for t in TransactionType]}"
        )
    
    # Validate merchant category
    try:
        # Convert to lowercase for validation
        merchant_category = data.merchant_category.lower()
        if merchant_category not in [c.value for c in MerchantCategory]:
            raise ValueError()
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid merchant category. Must be one of: {[c.value for c in MerchantCategory]}"
        )
    
    # Validate amount
    if not isinstance(data.amount, (int, float)) or data.amount < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Amount must be a non-negative number"
        )

@app.post('/fraud_detection')
@limiter.limit("500/minute")
async def fraud_detect(request: Request, data: TransactionDetails, db: Session = Depends(get_db)):
    """
    The endpoint is getting the given features in body and classify them and save response and these features in db.
    """
    try:
        # Validate transaction data
        validate_transaction(data)
        
        users = db.query(Transaction).filter(Transaction.user_id == data.user_id).order_by(Transaction.created_at.desc()).limit(6).all()

        new_transaction = {
            "transaction_type": data.transaction_type.lower(),
            "amount": data.amount,
            "merchant_category": data.merchant_category.lower(),
        }

        try:
            fraud, risk_score, features_importance = inference_code(new_transaction.copy(), users, model, dt_model, scaler)
        except Exception as e:
            print(f'Error in inference_code: {format_exc()}')
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference error: {str(e)}"
            )
        
        # Handle NaN values in features_importance
        processed_features = {}
        if features_importance is not None:
            for key, value in features_importance.items():
                if isinstance(value, (list, np.ndarray)):
                    # Convert to list if it's a numpy array
                    value_list = value.tolist() if isinstance(value, np.ndarray) else value
                    # Process each element in the list
                    processed_features[key] = [
                        float(x) if isinstance(x, (int, float, np.number)) and not np.isnan(x) else 0.0 
                        for x in value_list
                    ]
                elif isinstance(value, (int, float, np.number)):
                    processed_features[key] = float(value) if not np.isnan(value) else 0.0
                else:
                    processed_features[key] = 0.0
        else:
            processed_features = {"default": 0.0}

        new_transaction.update({
            'is_fraudulent': fraud,
            'prediction_prob': float(risk_score),
            "created_at": date.today(),
            "transaction_id": data.transaction_id,
            "user_id": data.user_id
        })

        if new_transaction:
            new_transaction = Transaction(**new_transaction)
            db.add(new_transaction)
            db.commit()
            db.refresh(new_transaction)

        return {
            "verdict": "Fraud" if fraud else "Legitimate",
            "fraud_probability_score": float(risk_score),
            "features_importance": processed_features
        }

    except HTTPException as he:
        raise he
    except RateLimitExceeded:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests"
        )
    except Exception as e:
        print(f'Error in api: {format_exc()}')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

@app.get("/model_logs/", response_model=List[TransactionResponse])
@limiter.limit("500/minute")
async def get_transactions(
    request: Request,
    db: Session = Depends(get_db),
    authenticated: bool = Depends(check_api_key)
):
    """This endpoint is fetching all predictions of model stored in db."""
    try:
        transactions = db.query(Transaction).all()
        # Ensure all required fields are present and valid
        validated_transactions = []
        for transaction in transactions:
            if transaction.transaction_id is None:
                transaction.transaction_id = "unknown"
            if transaction.user_id is None:
                transaction.user_id = "unknown"
            validated_transactions.append(transaction)
        return validated_transactions
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

@app.post('/model_monitoring')
@limiter.limit('500/minute')
async def model_monitoring(
    request: Request,
    data: ModelMonitoring,
    authenticated: bool = Depends(check_api_key)
):
    """
    This method will get date range as an input and then will get logs from db in that range
    will find sk_test and p_value and based on those calculation then it will retrain model again.
    """
    try:
        print("welcome to model_monitoring api.")
        run_training_pipeline(data.date_from, data.date_to)
        return {"message": "Model retrained."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

@app.post('/draw_graph')
async def draw_graph(
    data: Graph,
    authenticated: bool = Depends(check_api_key)
):
    """
    This endpoint accept shap values as input in dict and return encoded graph of features contribution in fraud transaction.
    """
    try:
        # Convert to NumPy arrays and handle NaN values
        shap_values = np.nan_to_num(np.array(data.graph_data.get("shap_values", [[]])))
        base_values = np.nan_to_num(np.array(data.graph_data.get("base_values", [[]])))
        data_values = np.nan_to_num(np.array(data.graph_data.get("data_values", [[]])))

        # Create the SHAP waterfall plot
        plt.figure()
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[0],
            base_values=base_values[0] if base_values.size > 0 else 0,
            data=data_values[0]
        ))

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        # Convert the image to a base64-encoded string
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return {"graph_image": image_base64}
    except Exception as e:
        print(f"Error in graph endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Add rate limit middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except RateLimitExceeded:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests"
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

