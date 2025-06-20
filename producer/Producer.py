import requests
import json
import time
import os
from dotenv import load_dotenv
from confluent_kafka import Producer

# ตั้งค่า Kafka Producer
p = Producer({'bootstrap.servers': 'localhost:8097,localhost:8098,localhost:8099'})

def acked(err, msg):
    if err is not None:
        print(f"Failed to deliver message: {msg}: {err}")
    else:
        print(f"Message produced: key = {msg.key().decode()} | value = {msg.value().decode()}")

# AviationStack API key
load_dotenv()
API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

# ดึงข้อมูลเที่ยวบิน (flight departures)
def get_flight_data():
    url = f'http://api.aviationstack.com/v1/flights?access_key={API_KEY}&flight_status=active'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('data', [])
    else:
        print(f"Error fetching data: {response.status_code} {response.text}")
        return []

def main():
    while True:
        flights = get_flight_data()
        for flight in flights:
            flight_id = flight.get('flight', {}).get('iata') or flight.get('flight', {}).get('icao') or 'unknown'
            value_json = json.dumps(flight)
            p.produce(
                topic='flightDelay',  # เปลี่ยนชื่อ topic ตามต้องการ
                key=flight_id.encode('utf-8'),
                value=value_json.encode('utf-8'),
                callback=acked
            )
            p.poll(0)
        p.flush()
        time.sleep(60)  # ดึงข้อมูลทุก 1 นาที

if __name__ == '__main__':
    main()
