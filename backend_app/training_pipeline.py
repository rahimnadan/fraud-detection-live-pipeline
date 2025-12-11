# import numpy as np 
# import pandas as pd 
# import matplotlib.pyplot as plt 
# from tensorflow import keras
# from tensorflow.keras import layers, callbacks
# from sklearn.model_selection import train_test_split
# import re
# from tensorflow.keras.optimizers import Adam
# import seaborn as sns
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import confusion_matrix, classification_report
# from scipy.stats import ks_2samp
# import psycopg2
# from scipy.spatial.distance import jensenshannon
# from traceback import format_exc

# def load_training_data(from_date, to_date):
#     try:
#         df = pd.read_csv('training_data/clean_data_pre.csv')
        
#         print(f"Training_data : {df.shape}")
#         print(f'column : {df.columns}')
#             # Define the SQL query with placeholders for the dates
#             # Create a connection to the PostgreSQL database
#         DATABASE_URL = 'postgresql://postgres:root@localhost/transactions'
#         connection = psycopg2.connect(DATABASE_URL)
    
#         # Create a cursor object to interact with the database
#         cursor = connection.cursor()
        
#         query = """
#         SELECT * FROM transactions
#         WHERE created_at BETWEEN %s AND %s;
#         """
        
        
#         # Execute the query with the provided date range
#         cursor.execute(query, (from_date, to_date))
            
#         # Fetch all the results
#         records = cursor.fetchall()
        
#         # Get the column names from the cursor description
#         columns = [desc[0] for desc in cursor.description]
#         print(f"columns : {columns}")
#         # Convert the data into a DataFrame
#         prod_data = pd.DataFrame(records, columns=columns)
#         print("************************************")
#         print(f"prod_data shape : {prod_data.shape}")
#         print("************************************")
#         print(f"prod_data : {prod_data.columns}")
#         if prod_data.shape[0] != 0:
#             print("production data : ",prod_data.shape)
#             labels = {'Legitimate' : 0, 'Fraud' : 1}  
#             prod_data['is_fraudulent']  = prod_data['is_fraudulent'].map(labels)
#             prod_data['rolling_avg_amount'] = prod_data.groupby('user_id')['amount'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
#             print(f"prod_data labels : {prod_data['is_fraudulent'].value_counts()}")
#             prod_data.drop(columns=['created_at', 'id','user_id', 'transaction_id', 'prediction_prob'], axis=1, inplace=True)

#     except Exception as e:
#         print(f"Error in load_training_data : {format_exc()}")
#         df = None
#         prod_data = None 

#     finally:
#         return df, prod_data



# def split_data(df_clean):

#     Y = df_clean['is_fraudulent']
#     X = df_clean.drop(columns=['is_fraudulent'], axis=1)   
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)

#     # Print dataset sizes
#     print(f"Training Set: {X_train.shape[0]} rows")
#     print(f"Testing Set: {X_test.shape[0]} rows")  

#     return X_train, X_test, y_train, y_test 

# def build_ann(X_train):
#     model = keras.Sequential([
#         layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)), 
#         layers.BatchNormalization(),
#         layers.Dropout(0.5),

#         layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
#         layers.BatchNormalization(),
#         layers.Dropout(0.5),

#         layers.Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
#         layers.BatchNormalization(),
#         layers.Dropout(0.4),

#         layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
#     ])
    
#     # Compile model
#     model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def train_model(X_train, y_train, model):

#     # Early stopping to prevent overfitting
#     early_stop = callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True)

#     # Train the model
#     history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=1)
#     model.save("trained_model_weights/fraud_detection_model_latest.h5")  # Saves in HDF5 format
#     return history 

# def graphs(history, model, X_test, y_test):

#     # Plot training history
#     plt.figure(figsize=(10, 5))
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Val Loss')
#     plt.legend()
#     plt.title('Model Loss Over Epochs')
#     plt.savefig("model_performance_graphs/model_loss.png")
#     # plt.show()

#     # Plot training history
#     plt.figure(figsize=(10, 5))
#     plt.plot(history.history['accuracy'], label='Train Loss')
#     plt.plot(history.history['val_accuracy'], label='Val Loss')
#     plt.legend()
#     plt.title('Model Loss Over Epochs')
#     plt.savefig("model_performance_graphs/model_training.png")
#     # plt.show()

#     # Predict fraud probabilities
#     y_pred = model.predict(X_test)
#     y_pred_labels = (y_pred > 0.5).astype(int)  # Convert to binary labels

#     # Confusion matrix

#     print("Classification Report:\n", classification_report(y_test, y_pred_labels))

#     sns.heatmap(confusion_matrix(y_test, y_pred_labels), annot=True, fmt='d', cmap='Blues')
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.savefig("model_performance_graphs/confusion_matrix.png")
#     # plt.show()

# def check_fraud_rate_drift(df_prev, df_new):

#     try:

#         # Compute fraud rate in old & new data
#         prev_fraud_rate = df_prev["is_fraudulent"].mean()  # % of transactions predicted as fraud
#         new_fraud_rate = df_new["is_fraudulent"].mean()

#         print(f"Old Fraud Rate: {prev_fraud_rate:.2%}")
#         print(f"New Fraud Rate: {new_fraud_rate:.2%}")
#         fraud_rate = abs(prev_fraud_rate - new_fraud_rate)

#         # Alert if fraud detection rate changes significantly
#         if fraud_rate > 0.05:  # If drift > 5%
#             print("🚨 ALERT: Significant change in fraud detection rate!")
#     except Exception as e:
#         print(f"Error in check_fraud_rate_drift : {format_exc()}")
#         fraud_rate = None 
#     finally:
#         return fraud_rate

# def check_feature_drift(feature, df_prev, df_new):

#     stat, p_value = ks_2samp(df_prev[feature], df_new[feature])
#     print(f"Feature: {feature} | KS-Test : {p_value:.5f}")
        
#     if p_value < 0.05:
#         print(f"🚨 ALERT: Feature '{feature}' has drifted!")
#     else:
#         print(f"✅ Feature '{feature}' is stable.")
#     return p_value


# def jensenshannon_method(old_data, new_data):
#     historical_jsd_values = []

#     features = ['amount', 'rolling_avg_amount', 'hour_of_day', 'lon', 'transaction_ratio']
#     for feature in features:

#         old_hist, _ = np.histogram(old_data[feature], bins=50, density=True)
#         new_hist, _ = np.histogram(new_data[feature], bins=50, density=True)

#         jsd_value = jensenshannon(old_hist, new_hist)
#         historical_jsd_values.append(jsd_value)

#     # Compute statistics
#     mean_jsd = np.mean(historical_jsd_values)
#     std_jsd = np.std(historical_jsd_values)
#     max_jsd = np.max(historical_jsd_values)

#     print(f"Mean JSD: {mean_jsd:.4f}, Std Dev: {std_jsd:.4f}, Max JSD: {max_jsd:.4f}")
#     JSD_THRESHOLD = mean_jsd + 2 * std_jsd  # Adjust multiplier based on sensitivity
#     print(f"Recommended JSD threshold: {JSD_THRESHOLD:.4f}")

#     return JSD_THRESHOLD

# def run_training_pipeline(from_date, to_date):

#    try:
#         df, prod_data = load_training_data(from_date, to_date)
#         print("prod_data shape",prod_data.shape)

#         fraud_rate = check_fraud_rate_drift(df, prod_data)
#         print(f"fraud_rate : {fraud_rate}")
#         threshold = jensenshannon_method(df, prod_data)
#         if threshold <= 0.38:
#             df = pd.concat([df, prod_data], ignore_index=True)
#             X_train, X_test, y_train, y_test  = split_data(df)
#             model = build_ann(X_train)
#             history = train_model(X_train, y_train, model)
#             graphs(history, model, X_test, y_test) 
#         else:
#             pass

#    except Exception as e:
#       print(f"Error in run_training_pipeline : {format_exc()}")



# # from datetime import date, timedelta

# # today = date.today()
# # prev_day = today - timedelta(days=1)
# # load_training_data(prev_day,today,)

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import ks_2samp
import seaborn as sns
import psycopg2
from scipy.spatial.distance import jensenshannon
from traceback import format_exc

# Database connection
DATABASE_URL = 'postgresql://postgres:root@localhost/transactions'

def load_training_data(from_date, to_date):
    try:
        df = pd.read_csv('training_data/clean_data_pre.csv')
        print(f"Training Data: {df.shape}")

        # Connect to database
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor()

        query = "SELECT * FROM transactions WHERE created_at BETWEEN %s AND %s;"
        cursor.execute(query, (from_date, to_date))
        records = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        prod_data = pd.DataFrame(records, columns=columns)
        print(f"Production Data Shape: {prod_data.shape}")

        if prod_data.empty:
            print("No production data found for the given date range.")
            return df, None  # Return existing data if no new data

        labels = {'Legitimate': 0, 'Fraud': 1}
        prod_data['is_fraudulent'] = prod_data['is_fraudulent'].map(labels)
        
        # Compute rolling average amount
        prod_data['rolling_avg_amount'] = (
            prod_data.groupby('user_id')['amount']
            .rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
        )

        print(f"Fraud Label Distribution: {prod_data['is_fraudulent'].value_counts()}")

        # Drop unnecessary columns
        cols_to_drop = ['created_at', 'id', 'user_id', 'transaction_id', 'prediction_prob']
        prod_data.drop(columns=[col for col in cols_to_drop if col in prod_data.columns], inplace=True)

        return df, prod_data

    except Exception as e:
        print(f"Error in load_training_data: {format_exc()}")
        return None, None


def split_data(df_clean):
    """Splits the dataset into training and testing sets."""
    Y = df_clean['is_fraudulent']
    X = df_clean.drop(columns=['is_fraudulent'])  

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42, stratify=Y
    )

    print(f"Training Set: {X_train.shape[0]} rows")
    print(f"Testing Set: {X_test.shape[0]} rows")

    return X_train, X_test, y_train, y_test 


def build_ann(X_train):
    """Builds a simple ANN model."""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)), 
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(X_train, y_train, model):
    """Trains the ANN model with early stopping."""
    early_stop = callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True)

    history = model.fit(
        X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=1
    )
    
    model.save("trained_model_weights/fraud_detection_model_latest.h5")
    return history


def check_fraud_rate_drift(df_prev, df_new):
    """Checks for significant changes in fraud detection rate."""
    if df_new is None:
        return None

    prev_fraud_rate = df_prev["is_fraudulent"].mean()
    new_fraud_rate = df_new["is_fraudulent"].mean()
    print(f"Old Fraud Rate: {prev_fraud_rate:.2%}, New Fraud Rate: {new_fraud_rate:.2%}")

    fraud_rate_diff = abs(prev_fraud_rate - new_fraud_rate)

    if fraud_rate_diff > 0.05:
        print("🚨 ALERT: Significant fraud rate change detected!")
    
    return fraud_rate_diff


def jensenshannon_method(old_data, new_data):
    """Computes Jensen-Shannon distance for feature drift detection."""
    if new_data is None:
        return None

    features = ['amount', 'rolling_avg_amount', 'hour_of_day', 'lon', 'transaction_ratio']
    historical_jsd_values = []

    for feature in features:
        if feature in old_data and feature in new_data:
            old_hist, _ = np.histogram(old_data[feature], bins=50, density=True)
            new_hist, _ = np.histogram(new_data[feature], bins=50, density=True)
            jsd_value = jensenshannon(old_hist, new_hist)
            historical_jsd_values.append(jsd_value)

    mean_jsd = np.mean(historical_jsd_values)
    std_jsd = np.std(historical_jsd_values)
    threshold = mean_jsd + 2 * std_jsd  # Adjust sensitivity

    print(f"JSD Threshold: {threshold:.4f}")
    return threshold


def run_training_pipeline(from_date, to_date):
    """Main function to load data, check drift, and train model if conditions are met."""
    try:
        df, prod_data = load_training_data(from_date, to_date)

        if prod_data is None:
            print("No new data available for training.")
            return

        fraud_rate_diff = check_fraud_rate_drift(df, prod_data)
        threshold = jensenshannon_method(df, prod_data)

        if threshold is not None and threshold <= 0.38:
            df = pd.concat([df, prod_data], ignore_index=True)
            X_train, X_test, y_train, y_test = split_data(df)
            model = build_ann(X_train)
            history = train_model(X_train, y_train, model)
        else:
            print("Data drift not detected, skipping training.")
    
    except Exception as e:
        print(f"Error in run_training_pipeline: {format_exc()}")


# from datetime import date, timedelta
# today = date.today()
# prev_day = today - timedelta(days=1)

# run_training_pipeline(prev_day, today)