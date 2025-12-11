import pytest
from fastapi.testclient import TestClient
from datetime import date, datetime, timedelta
import sys
import os
from pathlib import Path
import json
import time
import logging
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to import app
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Change working directory to backend_app directory for model loading
os.chdir(str(parent_dir))

from app import app
from model import Transaction

# Create a test client
client = TestClient(app)

# Test data
valid_transaction = {
    "transaction_type": "refund",
    "amount": 45.75,
    "merchant_category": "groceries",
    "transaction_id": "test_123",
    "user_id": "user_123"
}

# Get API key from environment or use default for testing
API_KEY = os.getenv("API_KEY", "test_key")
if not os.getenv("API_KEY"):
    os.environ["API_KEY"] = "test_key"  # Set default API key for testing

def retry_on_failure(max_retries=3, delay=2):
    """Decorator to retry tests on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except AssertionError as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Test failed, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Test database setup
@pytest.fixture(scope="session")
def test_db():
    """Setup test database and cleanup after tests"""
    # Here we could set up a test database
    # For now, we'll use the main database
    yield
    # Cleanup could be added here

@pytest.fixture
def auth_headers():
    """Provide authentication headers for tests"""
    return {"X-API-KEY": os.environ["API_KEY"]}

@pytest.fixture
def test_transaction():
    """Provide a test transaction with timestamp"""
    return {
        **valid_transaction,
        "transaction_id": f"test_{datetime.now().timestamp()}"
    }

# Basic Health Check Tests
def test_root_endpoint():
    """Test the root endpoint for health check"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}

# Fraud Detection API Tests
class TestFraudDetectionAPI:
    """Test suite for fraud detection endpoint"""
    
    def test_valid_transaction(self):
        """Test fraud detection with valid transaction data"""
        response = client.post("/fraud_detection", json=valid_transaction)
        assert response.status_code == 200
        
        data = response.json()
        assert "verdict" in data
        assert "fraud_probability_score" in data
        assert "features_importance" in data
        
        # Type checks
        assert isinstance(data["verdict"], str)
        assert isinstance(data["fraud_probability_score"], (int, float))
        assert isinstance(data["features_importance"], dict)
        
        # Value range checks
        assert data["verdict"] in ["Fraud", "Legitimate"]
        assert 0 <= data["fraud_probability_score"] <= 1
    
    def test_all_transaction_types(self):
        """Test all valid transaction types"""
        transaction_types = ["refund", "transfer", "purchase", "withdrawal"]
        for t_type in transaction_types:
            test_data = valid_transaction.copy()
            test_data["transaction_type"] = t_type.lower()  # Ensure lowercase
            response = client.post("/fraud_detection", json=test_data)
            assert response.status_code == 200, f"Failed for transaction type: {t_type}"
    
    def test_all_merchant_categories(self):
        """Test all valid merchant categories"""
        categories = ["groceries", "bank", "electronics", "atm", "restaurant", "luxury goods"]
        for category in categories:
            test_data = valid_transaction.copy()
            test_data["merchant_category"] = category.lower()  # Ensure lowercase
            response = client.post("/fraud_detection", json=test_data)
            assert response.status_code == 200, f"Failed for category: {category}"
    
    def test_invalid_transaction_data(self):
        """Test various invalid transaction scenarios"""
        invalid_cases = [
            {},  # Empty data
            {"transaction_type": "refund"},  # Missing required fields
            {"transaction_type": "invalid", "amount": 100, "merchant_category": "groceries"},  # Invalid type
            {"transaction_type": "refund", "amount": "invalid", "merchant_category": "groceries"},  # Invalid amount
            {"transaction_type": "refund", "amount": 100, "merchant_category": "invalid"}  # Invalid category
        ]
        
        for test_case in invalid_cases:
            response = client.post("/fraud_detection", json=test_case)
            assert response.status_code in [400, 422]  # Either bad request or validation error
    
    def test_extreme_amounts(self):
        """Test transaction amounts at extremes"""
        amounts = [0, 0.01, 999999.99, -100]
        for amount in amounts:
            test_data = valid_transaction.copy()
            test_data["amount"] = amount
            response = client.post("/fraud_detection", json=test_data)
            assert response.status_code == 200, f"Failed for amount: {amount}"
    
    @retry_on_failure(max_retries=3, delay=1)
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        logger.info("Starting rate limiting test")
        responses = []
        # Send 550 requests (50 more than limit) to ensure we hit the rate limit
        for i in range(550):
            response = client.post("/fraud_detection", json=valid_transaction)
            responses.append(response.status_code)
            if response.status_code == 429:
                logger.info(f"Rate limit hit after {i+1} requests")
                break
            # Small delay to prevent overwhelming the server
            time.sleep(0.001)
        
        # Check if we got any 429 (Too Many Requests) responses
        has_rate_limit = 429 in responses
        if not has_rate_limit:
            logger.error(f"Rate limit not triggered. Status codes received: {set(responses)}")
        assert has_rate_limit, "Rate limiting did not trigger after 550 requests"
        
        # Verify the number of successful requests before rate limit
        successful_requests = responses.count(200)
        logger.info(f"Successful requests before rate limit: {successful_requests}")
        assert successful_requests <= 500, f"Got {successful_requests} successful requests, expected maximum 500"
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import concurrent.futures
        
        def make_request():
            return client.post("/fraud_detection", json=valid_transaction)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            responses = [f.result() for f in futures]
        
        # Check all responses are valid
        for response in responses:
            assert response.status_code in [200, 429]

# Model Logs API Tests
class TestModelLogsAPI:
    """Test suite for model logs endpoint"""
    
    def test_without_api_key(self):
        """Test access without API key"""
        response = client.get("/model_logs/")
        assert response.status_code == 401, "Should require authentication"
    
    def test_with_invalid_api_key(self):
        """Test access with invalid API key"""
        response = client.get("/model_logs/", headers={"X-API-KEY": "invalid_key"})
        assert response.status_code == 401, "Should reject invalid API key"
    
    @retry_on_failure(max_retries=3, delay=1)
    def test_with_valid_api_key(self, auth_headers):
        """Test access with valid API key"""
        response = client.get("/model_logs/", headers=auth_headers)
        assert response.status_code == 200, "Should accept valid API key"
        data = response.json()
        assert isinstance(data, list), "Should return a list of transactions"
        
        if len(data) > 0:
            first_record = data[0]
            required_fields = [
                "transaction_type", "amount", "merchant_category",
                "is_fraudulent", "prediction_prob", "created_at"
            ]
            for field in required_fields:
                assert field in first_record, f"Missing required field: {field}"
    
    @retry_on_failure(max_retries=3, delay=1)
    def test_model_logs_response_time(self, auth_headers):
        """Test response time for model logs"""
        start_time = time.time()
        response = client.get("/model_logs/", headers=auth_headers)
        end_time = time.time()
        
        response_time = end_time - start_time
        logger.info(f"Model logs response time: {response_time:.2f} seconds")
        
        assert response.status_code == 200
        assert response_time < 3, f"Response too slow: {response_time:.2f} seconds"

# Performance Tests
class TestPerformance:
    """Performance test suite"""
    
    @retry_on_failure(max_retries=2, delay=1)
    def test_response_time(self):
        """Test response time for fraud detection"""
        # Warm up the model
        logger.info("Warming up the model...")
        client.post("/fraud_detection", json=valid_transaction)
        time.sleep(1)  # Allow warm-up to complete
        
        # Actual test
        start_time = time.time()
        response = client.post("/fraud_detection", json=valid_transaction)
        end_time = time.time()
        
        response_time = end_time - start_time
        logger.info(f"Response time: {response_time:.2f} seconds")
        
        assert response.status_code == 200
        assert response_time < 5, f"Response too slow: {response_time:.2f} seconds"
    
    def test_model_logs_response_time(self, auth_headers):
        """Test response time for model logs"""
        import time
        
        start_time = time.time()
        response = client.get("/model_logs/", headers=auth_headers)
        end_time = time.time()
        
        assert response.status_code == 200
        assert end_time - start_time < 3, "Response too slow"

if __name__ == "__main__":
    # Run tests with coverage report
    pytest.main([__file__, "-v", "--cov=app", "--cov-report=html", "--html=test_report.html"]) 