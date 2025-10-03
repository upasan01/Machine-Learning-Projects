import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib # Used for saving and loading the model

# --- 1. DATA SIMULATION (REPLACE WITH REAL DATA LOADING) ---
# NOTE: In a real project, you would replace this section with:
# df = pd.read_csv('laptop_dataset.csv')
# For demonstration purposes, we create a small DataFrame.
print("--- 1. Generating Simulated Laptop Data ---")

data = {
    'Brand': ['HP', 'Dell', 'Apple', 'Asus', 'HP', 'Apple', 'Dell', 'Lenovo'],
    'CPU_Type': ['Intel Core i5', 'Intel Core i7', 'Apple M1', 'Intel Core i5', 'Intel Core i3', 'Apple M2', 'Intel Core i7', 'AMD Ryzen 5'],
    'RAM_GB': [8, 16, 8, 16, 4, 16, 32, 8],
    'Storage_GB': [256, 512, 256, 512, 128, 1000, 1000, 256],
    'Screen_Size_Inches': [14.0, 15.6, 13.3, 15.6, 14.0, 13.6, 17.0, 15.0],
    'Price_USD': [650, 1200, 1350, 950, 450, 1800, 2500, 700]
}
df = pd.DataFrame(data)

# Define feature (X) and target (y)
X = df.drop('Price_USD', axis=1)
y = df['Price_USD']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split: Train samples={len(X_train)}, Test samples={len(X_test)}")


# --- 2. PREPROCESSING PIPELINE SETUP ---

# Define categorical and numerical features
categorical_features = ['Brand', 'CPU_Type']
numerical_features = ['RAM_GB', 'Storage_GB', 'Screen_Size_Inches']

# Create preprocessing steps
numerical_transformer = StandardScaler() # Scaling numerical data is important
categorical_transformer = OneHotEncoder(handle_unknown='ignore') # Convert categories to numbers

# Create a ColumnTransformer to apply the correct transformations to the correct columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (none in this case, but good practice)
)

# Create the full machine learning pipeline
# Step 1: Preprocessing, Step 2: Model (RandomForestRegressor)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# --- 3. MODEL TRAINING ---
print("\n--- 3. Training Model (RandomForestRegressor) ---")
model_pipeline.fit(X_train, y_train)
print("Training complete.")


# --- 4. EVALUATION ---
print("\n--- 4. Evaluating Model Performance ---")
y_pred = model_pipeline.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"R-squared (R2 Score): {r2:.4f}")


# --- 5. PREDICTION FUNCTION ---

def predict_laptop_price(model, brand, cpu_type, ram_gb, storage_gb, screen_size):
    """
    Predicts the price of a laptop based on its specifications.

    Args:
        model (Pipeline): The trained scikit-learn pipeline.
        brand (str): Laptop brand (e.g., 'Dell').
        cpu_type (str): CPU specification (e.g., 'Intel Core i7').
        ram_gb (int): RAM size in GB (e.g., 16).
        storage_gb (int): Storage size in GB (e.g., 512).
        screen_size (float): Screen size in inches (e.g., 15.6).
    """
    # Create a DataFrame for a single new prediction instance
    new_data = pd.DataFrame({
        'Brand': [brand],
        'CPU_Type': [cpu_type],
        'RAM_GB': [ram_gb],
        'Storage_GB': [storage_gb],
        'Screen_Size_Inches': [screen_size]
    })

    # Predict the price
    predicted_price = model.predict(new_data)[0]
    return predicted_price

# Example prediction
example_brand = 'Dell'
example_cpu = 'Intel Core i7'
example_ram = 16
example_storage = 512
example_screen = 15.6

predicted_price = predict_laptop_price(
    model_pipeline,
    example_brand,
    example_cpu,
    example_ram,
    example_storage,
    example_screen
)

print(f"\n--- 5. Example Prediction ---")
print(f"Specifications: {example_brand}, {example_cpu}, {example_ram}GB RAM, {example_storage}GB SSD, {example_screen}\" Screen")
print(f"Predicted Price: ${predicted_price:.2f}")


# --- 6. MODEL PERSISTENCE (Optional but recommended) ---
MODEL_FILENAME = 'laptop_price_model.joblib'
joblib.dump(model_pipeline, MODEL_FILENAME)
print(f"\nModel saved successfully to {MODEL_FILENAME}")

# To load the model later:
# loaded_model = joblib.load(MODEL_FILENAME)
# print(f"Model loaded successfully.")
