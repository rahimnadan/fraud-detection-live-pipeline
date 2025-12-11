import pandas as pd 
import shap
import numpy as np
from traceback import format_exc
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


def generate_shap_description(shap_values, feature_names, base_value, prediction):
    description = f"The model predicts a {prediction * 100:.2f}% probability of fraud. Key factors contributing to this prediction are:"
    contributions = []
    # print(f"shap_values : {shap_values}")
    shap_values = shap_values.flatten()
    for feature, value in zip(feature_names, shap_values):
        # print(feature)
        # print("value",value)
        if value == 0.0:
            contributions.append(f"The {feature} dose not contributed the likelihood of fraud by {abs(value):.2f}.")
        
        elif value > 0.0:
            contributions.append(f"The {feature} increased the likelihood of fraud by {abs(value):.2f}.")
        
        else:
            contributions.append(f"The {feature} decreased the likelihood of fraud by {abs(value):.2f}.")
    
    return description, contributions

def predict_fraud_with_rf(new_transaction, rf_model):

    try:
        # print("predict_fraud_with dt ")

        prediction = rf_model.predict(new_transaction)[0]
        fraud_probability = rf_model.predict_proba(new_transaction)[:, 1][0] 
        # is_fraud  = prediction
        is_fraud = 1 if fraud_probability > 0.5 else 0

        # print(f"Fraud Probability: {fraud_probability:.4f}")
        # print(f"Fraud prediction: {prediction:.4f}")
        # print(f"Is Fraudulent? {'Yes' if is_fraud else 'No'}")

        # **SHAP Explanation**
        explainer = shap.Explainer(rf_model, new_transaction)
        shap_values = explainer(new_transaction)
        # Plot SHAP values
        # shap.summary_plot(shap_values, new_transaction)

        # # Save the figure as PNG
        # plt.savefig("shap_summary.png", bbox_inches="tight", dpi=300)
        # plt.close()  # Close the plot to free memory
        # Extracting specific values
        shap_values_array = shap_values.values
        base_values = shap_values.base_values
        data_values = shap_values.data 

        feature_names = new_transaction.columns.tolist()

        description, contributions = generate_shap_description(shap_values_array, feature_names, base_values, fraud_probability)

        features_importance = {"shap_values" : shap_values_array, "base_values" : base_values, "data_values" : data_values, "description" : description, "features_contributions" : contributions}

        # print("SHAP Explanation:")
        # print(features_importance)
        # shap.summary_plot(shap_values, new_transaction)

        return fraud_probability, features_importance
        # return fraud_probability
    
    except Exception as e:
        print(f"Error in  predict_fraud_with_dt : {format_exc()}")
        return 0.2, None 
    
def predict_fraud_ann(new_transaction, model):

    try:

        predictions = model.predict(new_transaction)
        # print("Predictions values:", predictions)

        fraud_probability = predictions[0][0]
        # print("fraud_probability response : ",fraud_probability)
        verdict = "Fraud" if fraud_probability > 0.7 else "Legitimate"
      
            # **SHAP Explanation**
        explainer = shap.Explainer(model, new_transaction)
        shap_values = explainer(new_transaction)
        
        shap_values_array = shap_values.values
        base_values = shap_values.base_values
        data_values = shap_values.data
        feature_names = new_transaction.columns.tolist()

        description, contributions = generate_shap_description(shap_values_array[0], feature_names, base_values, fraud_probability)

        features_importance = {"shap_values" : shap_values_array.tolist(), "base_values" : base_values.tolist(), "data_values" : data_values.tolist(), "description" : description, "features_contributions" : contributions}

        # print(f"features_importance : {features_importance}")
        return fraud_probability, features_importance
        # return verdict, fraud_probability, features_importance

    except Exception as e:
        # print(f"Error in ANN : {format_exc()}")
        # print(f"predict_fraud_ann : {e}")
        return  0.20, None

# def detect_chargeback(transaction):
#     if transaction['is_chargeback'] and transaction['amount'] > 1000:
#         return "High Risk"
#     return "Low Risk"

def insert_at_index(d, index, key, value):
    items = list(d.items())  # Convert dict to list of tuples
    items.insert(index, (key, value))  # Insert new key-value pair at index
    return dict(items)  # Convert back to dictionary

def inference_code(values, users, model, rf_model, scaler):
    
    try:

        if users:
            # Convert SQLAlchemy objects to dictionaries
            users_dict = [t.__dict__ for t in users]

            # Remove SQLAlchemy internal attributes like `_sa_instance_state`
            for t in users_dict:
                t.pop('_sa_instance_state', None)
        
            # Convert to DataFrame
            df = pd.DataFrame(users_dict)
            
            values['amount'] = np.log1p(values['amount'])
            amount_sum = df['amount'].sum() + values.get("amount")
            rolling_avg_amount = amount_sum/df.shape[0]
            values = insert_at_index(values, 3, "rolling_avg_amount", rolling_avg_amount)

            # print("rolling_avg_amount added  exist previous record: ", values['rolling_avg_amount'])
        else:
            rolling_avg_amount = values['amount']
            values = insert_at_index(values, 3, "rolling_avg_amount", rolling_avg_amount)
            # values['rolling_avg_amount'] = values['amount']
            # print("rolling_avg_amount added  amount: ", values['rolling_avg_amount'])

        # print(f"values : {values}")
        merchant_category = {"groceries": 0, "bank": 1, "electronics": 2, "atm": 3, "restaurant": 4, "luxury goods": 5}
        transaction_type = {"refund" : 0,  "transfer": 1, "purchase": 2, "withdrawal": 3}

        new_transaction = pd.DataFrame([values])
        # new_transaction['merchant_category'] = new_transaction['merchant_category'].map(merchant_category)
        new_transaction['merchant_category'] = merchant_category[values['merchant_category'].lower()]

        # print(f"merchant_category : {new_transaction['merchant_category']}")
        new_transaction['transaction_type'] = transaction_type[values['transaction_type'].lower()]

        # print(f"transaction values in inference code. : {new_transaction}")

        columns_to_normalize = ['amount', 'rolling_avg_amount']
        new_transaction[columns_to_normalize] = scaler.fit_transform(new_transaction[columns_to_normalize])

        # ann_prob, feature_imp_ann = predict_fraud_ann(new_transaction.copy(), model)
        # dt_prob, feature_imp_dt = predict_fraud_with_rf(new_transaction.copy(), rf_model)
        # Run in parallel
        with ThreadPoolExecutor() as executor:
            future_ann = executor.submit(predict_fraud_ann, new_transaction, model)
            future_dt = executor.submit(predict_fraud_with_rf, new_transaction, rf_model)

            ann_prob, feature_imp_ann = future_ann.result()
            dt_prob, feature_imp_dt = future_dt.result()
            # print("ann_prob : ",ann_prob)
        # **Alternative: Weighted Prediction**
        weight_dt = 0.4  # Decision Tree weight
        weight_ann = 0.6  # ANN weight (since ANN might perform better)
        final_weighted_proba = (weight_dt * dt_prob) + (weight_ann * ann_prob)

        final_weighted_pred = 1 if final_weighted_proba > 0.7 else 0

        classes = ["Legitimate", "Fraud"]
        # print(f"final_weighted_proba : {final_weighted_proba}")
        return classes[final_weighted_pred], final_weighted_proba, feature_imp_ann
    
    except Exception as e:
        print(f"Error in inerence_code : {format_exc()}")
        return "Legitimate", 0.20, None
     
