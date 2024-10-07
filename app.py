import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load the pickled model
with open('random_forest_model.pkl', 'rb') as f:
    random_forest = pickle.load(f)

# Load and prepare the data
df = pd.read_excel('/Users/ahmadazhar/Desktop/Dashboard/data.xlsx')

# Column mapping and renaming
column_mapping = {
    'CustNo': 'customer_number', 'CustName': 'customer_name', 'Address': 'customer_address', 
    'BarangayName': 'district', 'ProvinceName': 'province', 'SalesManTerritory': 'salesman_territory', 
    'InvNo': 'invoice_number', 'InvDt': 'invoice_date', 'DoDt': 'delivery_order_date', 
    'DeliveryDate': 'delivery_date', 'Salesman': 'salesman_name', 'DueDate': 'due_date', 
    'UOM': 'unit_of_measurement', 'ItemName': 'product_name', 'SubTotal': 'subtotal_amount', 
    'GstAmt': 'gst_amount', 'TotalAmt': 'total_amount', 'PaidAmt': 'paid_amount', 
    'Price': 'product_price', 'SubAmt': 'sub_amount', 'LineNo': 'line_number', 'ItemNo': 'item_number', 
    'Qty': 'quantity', 'BaseQty': 'base_quantity', 'ts': 'timestamp'
}
df.rename(columns=column_mapping, inplace=True)

target_column = ['quantity']
target_scaler = MinMaxScaler()
target_scaler.fit(df[target_column])

df['due_date'] = pd.to_datetime(df['due_date'])

df['due_day_of_week'] = df['due_date'].dt.dayofweek
df['due_month'] = df['due_date'].dt.month
df['due_day_of_month'] = df['due_date'].dt.day

# Initialize and fit the scaler
columns_to_scale = ['base_quantity', 'product_price', 'total_amount', 'gst_amount']
scaler = MinMaxScaler()
scaler.fit(df[columns_to_scale])

# Streamlit app
def main():
    st.title("Stock Quantity Prediction App")
    
    st.write("Please provide the input features to predict the quantity of stock.")

    # Creating a form for user inputs
    with st.form(key='prediction_form'):
        customer_number = st.number_input("Customer Number", min_value=0, step=1)
        item_number = st.number_input("Item Number", min_value=0, step=1)
        date = st.date_input("Date")
        base_quantity = st.number_input("Base Quantity", min_value=0.0, step=0.1)
        product_price = st.number_input("Product Price", min_value=0.0, step=0.1)
        total_amount = st.number_input("Total Amount", min_value=0.0, step=0.1)
        gst_amount = st.number_input("GST Amount", min_value=0.0, step=0.1)
        
        submit_button = st.form_submit_button(label='Predict')
    
    # When the form is submitted
    if submit_button:
        due_day_of_week = date.weekday()
        due_month = date.month
        due_day_of_month = date.day
        input_features_scaled = np.array([[base_quantity, product_price, gst_amount, total_amount]])
        scaled_features = scaler.transform(input_features_scaled)
        input_features_non_scaled = np.array([[customer_number, item_number, due_day_of_week, due_month, due_day_of_month]])
        final_input_features = np.concatenate([scaled_features, input_features_non_scaled], axis=1)
        scaled_prediction = random_forest.predict(final_input_features)
        scaled_prediction = scaled_prediction.reshape(-1, 1)
        original_prediction = target_scaler.inverse_transform(scaled_prediction)
        original_prediction = np.abs(original_prediction)
        rounded_prediction = np.round(original_prediction)
        st.write(f"Predicted Stock Quantity: {rounded_prediction[0][0]}")
        
    
if __name__ == '__main__':
    main()
