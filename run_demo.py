"""Smart Home Automation Demo — Author: Adham Aboulkheir"""
import numpy as np, matplotlib, os, sys
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from ml.occupancy_predictor import OccupancyPredictor, generate_sensor_data, SensorReading
from sklearn.model_selection import train_test_split

def main():
    print("Smart Home Automation Demo")
    os.makedirs("outputs", exist_ok=True)
    readings, labels = generate_sensor_data(n_days=60)
    train_r, test_r, train_l, test_l = train_test_split(readings, labels, test_size=0.2, random_state=42)
    predictor = OccupancyPredictor()
    predictor.fit(train_r, train_l)
    metrics = predictor.evaluate(test_r, test_l)
    print(f"  Accuracy: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f}")
    hours = list(range(24))
    probs = [predictor.predict_proba(SensorReading(h, 1, 21, 55, 500, 200, 1, 0)) for h in hours]
    savings = [35 if p < 0.2 else 15 if p < 0.5 else 0 for p in probs]
    print(f"  Avg energy savings: {np.mean(savings):.0f}%")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0d1117")
    for ax in axes: ax.set_facecolor("#161b22")
    axes[0].bar(hours, probs, color=["#00c9b1" if p>0.5 else "#161b22" for p in probs], alpha=0.85, edgecolor="#00c9b1", linewidth=0.5)
    axes[0].axhline(y=0.5, color="#f4a261", linestyle="--", linewidth=1.5)
    axes[0].set_title("Occupancy Probability (24h)", color="white")
    axes[0].set_xlabel("Hour", color="white"); axes[0].tick_params(colors="white"); axes[0].grid(alpha=0.3, color="#21262d")
    axes[1].plot(hours, savings, color="#00c9b1", linewidth=2, marker="o", markersize=5)
    axes[1].fill_between(hours, savings, alpha=0.2, color="#00c9b1")
    axes[1].set_title("Energy Savings by Hour", color="white")
    axes[1].set_xlabel("Hour", color="white"); axes[1].tick_params(colors="white"); axes[1].grid(alpha=0.3, color="#21262d")
    plt.tight_layout()
    plt.savefig("outputs/smart_home_results.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("  Saved: outputs/smart_home_results.png")

if __name__ == "__main__":
    main()
