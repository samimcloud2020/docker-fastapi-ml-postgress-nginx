# train_model.py (ML Model Training)
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import boto3

# Sample Data
data = pd.DataFrame({
    'area': [1000, 1500, 2000],
    'room': [3, 4, 5],
    'price': [300000, 400000, 500000]
})

X = data[['area', 'room']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'linear_model.pkl')

# AWS S3 Config
S3_BUCKET = "sambucket9434"
S3_KEY = "linear_model.pkl"
s3 = boto3.client('s3', aws_access_key_id='', aws_secret_access_key='')

# Upload to S3
s3.upload_file('linear_model.pkl', S3_BUCKET, 'linear_model.pkl')
