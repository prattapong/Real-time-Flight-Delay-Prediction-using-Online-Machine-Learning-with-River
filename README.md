
# ✈️ Real-time Flight Delay Prediction using Online Machine Learning with River

Predict flight delays in real-time using streaming data, Apache Kafka, online machine learning via [River](https://riverml.xyz/), and an interactive dashboard with Streamlit. The system learns continuously from real flight data, improving prediction quality over time.

## 📌 Project Highlights

- 🔁 **Online Learning**: Uses logistic regression that updates on each flight.
- 🌐 **Real-time Data**: Live flight data via [AviationStack API](https://aviationstack.com/).
- ⚡ **Streaming with Kafka**: Producer-consumer architecture using Kafka.
- 📊 **Dashboard with Streamlit**: Visualize predictions, flight info, and model performance live.
- 🔄 **Self-updating model**: Learns from actual outcomes when available with Online Machine Learning from River.

---

## 📁 Architecture Overview

![image](https://github.com/user-attachments/assets/fdfaeb40-4991-4b68-b7ff-f99b91a1bb3f)


---

## 🚀 Quickstart

### ⚙️ Requirements

- Python 3.8+
- Kafka (tested with 3-broker cluster)
- Docker (optional for Kafka deployment)
- `.env` file with your `AVIATIONSTACK_API_KEY`

### 🐳 Kafka Setup (with Docker Compose)

You can deploy Kafka with this multi-broker template:

```yaml
version: '2'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper
    ...
  kafka1:
    image: confluentinc/cp-kafka
    ports:
      - "8097:9092"
    ...
  kafka2:
    ...
  kafka3:
    ...
```

> Use topic name: `flightDelay`

---

## 🔧 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/prattapong/Real-time-Flight-Delay-Prediction-using-Online-Machine-Learning-with-River.git
cd Real-time-Flight-Delay-Prediction-using-Online-Machine-Learning-with-River
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> Or manually: `pip install river streamlit confluent_kafka python-dotenv altair pandas numpy`

### 3. Set API Key

Create a `.env` file:

```
AVIATIONSTACK_API_KEY=your_api_key_here
```

### 4. Start Kafka with Docker Compose

```bash
docker-compose up -d
```

Wait until all Kafka services are up and running.

### 5. Start Kafka Producer

```bash
python Producer.py
```

This script fetches live flights from AviationStack and sends them to Kafka.

### 6. Run Streamlit Consumer Dashboard

```bash
streamlit run Consumer.py
```

> 📈 Live predictions and analytics will appear once Kafka messages are ingested.

---

## 🧠 Machine Learning Pipeline

Implemented using [River](https://riverml.xyz/):

```python
model = (
    preprocessing.OneHotEncoder() |
    preprocessing.StandardScaler() |
    linear_model.LogisticRegression()
)
```

- **OneHotEncoder**: Encodes categorical features like airport codes and airline names.
- **StandardScaler**: Normalizes numerical features.
- **LogisticRegression**: Binary classifier for delay prediction.
- **Online Update**: Model is updated if the actual delay label is available.

---

## 📊 Dashboard Features (Streamlit)

- **Live Predictions**: Delay probability shown for each flight.
- **Recent Flights Table**: Shows scheduled/actual times, delay status, etc.
- **Metrics Cards**: Accuracy, delay rate, precision, recall, average delay.
- **Charts**:
  - Top departure/arrival airports
  - Delay heatmap by hour
  - Airline performance comparison
  - Delay probability time series

---

## 🛠️ Feature Engineering

Each Kafka message (flight) is converted to features:

- `dep_airport`, `arr_airport`
- `dep_terminal`, `arr_terminal`
- `airline`, `route`
- `hour`, `day_of_week`
- Label: `1` if arrival delay > 0 else `0`

---

## ✅ Performance Metrics

Metrics are updated in real-time as labeled flight data becomes available:

- **Accuracy**: % of correct predictions
- **Precision**: % of predicted delays that were actually delayed
- **Recall**: % of actual delays that were correctly predicted
- **Delay Rate**: % of total flights delayed

---

## 📈 Sample API Message (Kafka Producer)

```json
{
  "key": "EK313",
  "value": {
    "flight_date": "2025-06-18",
    "flight_status": "active",
    "departure": {
      "airport": "Haneda Airport", "timezone": "Asia/Tokyo", "iata": "HND",
      "icao": "RJTT", "terminal": "3", "gate": "109","delay": 37,
      "scheduled": "2025-06-18T00:05:00+00:00",
      "estimated": "2025-06-18T00:05:00+00:00",
      "actual": "2025-06-18T00:42:00+00:00",
      "estimated_runway": "2025-06-18T00:42:00+00:00",
      "actual_runway": "2025-06-18T00:42:00+00:00"
    },
    "arrival": {
      "airport": "Dubai", "timezone": "Asia/Dubai", "iata": "DXB",
      "icao": "OMDB", "terminal": "3", "gate": null, "delay": 3, "baggage": "1",
      "scheduled": "2025-06-18T05:45:00+00:00",
      "estimated": "2025-06-18T05:48:00+00:00",
      "actual": null, "estimated_runway": null, "actual_runway": null
    },
    "airline": {
      "name": "Emirates", "iata": "EK", "icao": "UAE"
    },
    "flight": {
      "number": "313", "iata": "EK313", "icao": "UAE313", "codeshared": null
    },
    "aircraft": null,
    "live": null
  }
}
```

---

## 🔮 Future Improvements

- 🧪 Try advanced classifiers: `HoeffdingTree`, `SGDClassifier`, etc.
- 🧠 Optimizers: Experiment with `Adam`, `RMSProp`, etc.
- 🔧 Hyperparameter tuning with `SuccessiveHalvingClassifier`
- 🌤 Integrate external features like weather, holidays
- 📦 Store model state and historical predictions

---

## 🧑‍💻 Contributors

- Rattapong Pojpatinya  
- Tuksaporn Chaiyaraks  
- Pakawat Naktubtee  
- Kanpitcha Panbualuang  
- Kittitat Wattanasuntikul

---

## 📜 License

This project is for educational and research purposes. Contact the authors for permission if you'd like to use this in production.
