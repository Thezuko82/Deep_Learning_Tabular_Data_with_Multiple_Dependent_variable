import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model

st.set_page_config(page_title="Enhanced Tabular Deep Learning", layout="wide")
st.title("🔬 Deep Learning on Tabular Data with Full Control")

# Upload dataset
st.sidebar.header("Step 1: Upload CSV Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Dataset")
    st.dataframe(df.head())

    # Select dependent and independent variables
    st.sidebar.header("Step 2: Variable Selection")
    all_columns = df.columns.tolist()
    target_cols = st.sidebar.multiselect("Select Dependent (Target) Variable(s)", all_columns)
    feature_cols = st.sidebar.multiselect("Select Independent (Feature) Variables", [col for col in all_columns if col not in target_cols])

    if target_cols and feature_cols:
        st.sidebar.header("Step 3: Data Splitting")
        train_size = st.sidebar.slider("Training Set %", 0.1, 0.9, 0.7)
        val_size = st.sidebar.slider("Validation Set %", 0.05, 0.4, 0.15)
        test_size = 1.0 - train_size - val_size

        st.sidebar.header("Step 4: Model Structure")
        num_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 2)
        units_per_layer = st.sidebar.slider("Neurons per Layer", 4, 128, 16)

        if st.sidebar.button("Train Model"):
            # Split numerical and categorical columns
            X = df[feature_cols]
            y = df[target_cols]

            categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
            numeric_cols = [col for col in X.columns if col not in categorical_cols]

            # Build preprocessing pipeline
            preprocessor = ColumnTransformer([
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
            ])

            X_processed = preprocessor.fit_transform(X)

            # Split data
            X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y.values, train_size=train_size, random_state=42)
            relative_val_size = val_size / (val_size + test_size)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - relative_val_size, random_state=42)

            # Build model
            model = Sequential()
            model.add(Dense(units_per_layer, activation='relu', input_shape=(X_processed.shape[1],)))
            for _ in range(num_layers - 1):
                model.add(Dense(units_per_layer, activation='relu'))
            model.add(Dense(len(target_cols), activation='linear'))
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # Train
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, verbose=0)

            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.subheader("📊 Model Evaluation")
            st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
            st.write(f"**R² Score:** {r2:.4f}")

            st.write("### 📈 Training Curves")
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            ax[0].plot(history.history['loss'], label='Train Loss')
            ax[0].plot(history.history['val_loss'], label='Val Loss')
            ax[0].set_title("Loss over Epochs")
            ax[0].legend()

            ax[1].plot(history.history['mae'], label='Train MAE')
            ax[1].plot(history.history['val_mae'], label='Val MAE')
            ax[1].set_title("MAE over Epochs")
            ax[1].legend()
            st.pyplot(fig)

            # Option to save model
            st.sidebar.header("Step 5: Save Trained Model")
            model_name = st.sidebar.text_input("Model File Name", value="my_tabular_model.h5")
            if st.sidebar.button("Save Model"):
                save_model(model, model_name)
                st.success(f"Model saved as `{model_name}` in current directory.")

