# main.py (FastAPI App)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os
import tempfile
from typing import List
import psycopg2
import socket

app = FastAPI()

# AWS S3 Config
S3_BUCKET = "sambucket9434"
S3_KEY = "linear_model.pkl"
s3 = boto3.client('s3', aws_access_key_id='', aws_secret_access_key='')

# Download model from S3
def load_model():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        s3.download_fileobj(S3_BUCKET, S3_KEY, tmp)
        tmp.flush()
        model = joblib.load(tmp.name)
    return model

model = load_model()


# PostgreSQL connection setup
conn = psycopg2.connect(
    dbname="ml_db",
    user="postgres",
    password="postgres",
    host="db",
    port="5432"
)
cursor = conn.cursor()

# Ensure table exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS houses (
        id SERIAL PRIMARY KEY,
        area FLOAT NOT NULL,
        room INT NOT NULL,
        price FLOAT NOT NULL
    )
''')
conn.commit()

class House(BaseModel):
    area: float
    room: int
    price: float

@app.get("/")
def read_api():
    hostname = socket.gethostname()  # ← define it here
    ip_address = socket.gethostbyname(hostname)
    return f"This Hostname is {hostname} and IP Address is {ip_address}"

@app.get("/houses", response_model=List[House])
def read_houses():
    cursor.execute("SELECT area, room, price FROM houses")
    rows = cursor.fetchall()
    return [House(area=row[0], room=row[1], price=row[2]) for row in rows]

@app.post("/houses")
def create_house(house: House):
    cursor.execute("INSERT INTO houses (area, room, price) VALUES (%s, %s, %s)",
                   (house.area, house.room, house.price))
    conn.commit()
    return {"status": "created"}

@app.put("/houses/{id}")
def update_house(id: int, house: House):
    cursor.execute("UPDATE houses SET area=%s, room=%s, price=%s WHERE id=%s",
                   (house.area, house.room, house.price, id))
    conn.commit()
    return {"status": "updated"}

@app.delete("/houses/{id}")
def delete_house(id: int):
    cursor.execute("DELETE FROM houses WHERE id=%s", (id,))
    conn.commit()
    return {"status": "deleted"}

@app.post("/predict")
def predict(house: House):
    features = [[house.area, house.room]]
    predicted_price = model.predict(features)
    return {"predicted_price": predicted_price[0]}



