"""Smart Home Occupancy Prediction — Author: Adham Aboulkheir"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from dataclasses import dataclass

@dataclass
class SensorReading:
    hour: int; day_of_week: int; temperature: float; humidity: float
    co2_ppm: float; light_lux: float; motion: int; door_open: int

def generate_sensor_data(n_days=60, seed=42):
    np.random.seed(seed)
    readings, labels = [], []
    for day in range(n_days):
        for hour in range(24):
            if 7 <= hour <= 8 or 18 <= hour <= 23: occupied = int(np.random.random() > 0.2)
            elif 9 <= hour <= 17: occupied = int(np.random.random() > 0.8)
            else: occupied = int(np.random.random() > 0.95)
            readings.append(SensorReading(
                hour=hour, day_of_week=day%7,
                temperature=20+3*np.sin(hour/12*np.pi)+np.random.normal(0,0.5),
                humidity=50+10*np.random.random(),
                co2_ppm=400+(300 if occupied else 0)+np.random.normal(0,20),
                light_lux=500 if (8<=hour<=20 and occupied) else 10+np.random.random()*50,
                motion=int(occupied and np.random.random()>0.3),
                door_open=int(occupied and np.random.random()>0.7)
            ))
            labels.append(occupied)
    return readings, labels

def readings_to_features(readings):
    return np.array([[r.hour, r.day_of_week, r.temperature, r.humidity,
                      r.co2_ppm, r.light_lux, r.motion, r.door_open,
                      np.sin(2*np.pi*r.hour/24), np.cos(2*np.pi*r.hour/24),
                      np.sin(2*np.pi*r.day_of_week/7)] for r in readings])

class OccupancyPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        self.scaler = StandardScaler()
    def fit(self, readings, labels):
        X = readings_to_features(readings)
        self.model.fit(self.scaler.fit_transform(X), labels)
        return self
    def predict_proba(self, reading):
        X = readings_to_features([reading])
        return float(self.model.predict_proba(self.scaler.transform(X))[0, 1])
    def evaluate(self, readings, labels):
        X = readings_to_features(readings)
        y_pred = self.model.predict(self.scaler.transform(X))
        return {"accuracy": accuracy_score(labels, y_pred), "f1": f1_score(labels, y_pred, average="weighted")}

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    readings, labels = generate_sensor_data(n_days=60)
    train_r, test_r, train_l, test_l = train_test_split(readings, labels, test_size=0.2, random_state=42)
    predictor = OccupancyPredictor()
    predictor.fit(train_r, train_l)
    metrics = predictor.evaluate(test_r, test_l)
    print(f"Accuracy: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f}")
