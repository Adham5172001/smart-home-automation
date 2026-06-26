"""
Smart Home Energy Optimisation
Author: Adham Aboulkheir
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class EnergySchedule:
    hour: int
    device: str
    action: str  # "on" or "off"
    expected_saving_kwh: float
    reason: str


class EnergyOptimiser:
    """
    Rule-based + ML energy optimisation for smart home.
    Reduces energy consumption by scheduling devices based on occupancy predictions.
    """

    DEVICE_POWER_KW = {
        "hvac":        3.5,
        "water_heater": 4.0,
        "washing_machine": 2.0,
        "dishwasher":  1.8,
        "ev_charger":  7.2,
        "lighting":    0.3,
        "standby":     0.15,
    }

    def __init__(self, tariff_peak_hours: List[int] = None):
        self.tariff_peak_hours = tariff_peak_hours or list(range(7, 10)) + list(range(17, 21))

    def optimise_schedule(self, occupancy_probs: List[float],
                           devices: List[str] = None) -> List[EnergySchedule]:
        """
        Generate optimised device schedule based on occupancy predictions.
        """
        if devices is None:
            devices = ["hvac", "water_heater", "washing_machine", "lighting"]

        schedule = []

        for hour, occ_prob in enumerate(occupancy_probs):
            for device in devices:
                power_kw = self.DEVICE_POWER_KW.get(device, 1.0)
                is_peak = hour in self.tariff_peak_hours

                if occ_prob < 0.2:
                    # No one home — turn off most devices
                    if device in ["hvac", "lighting"]:
                        saving = power_kw * 1.0  # 1 hour saving
                        schedule.append(EnergySchedule(
                            hour=hour, device=device, action="off",
                            expected_saving_kwh=saving,
                            reason=f"Occupancy probability {occ_prob:.0%} — no one home"
                        ))
                elif occ_prob < 0.5 and is_peak:
                    # Low occupancy during peak tariff
                    if device in ["washing_machine", "dishwasher", "ev_charger"]:
                        schedule.append(EnergySchedule(
                            hour=hour, device=device, action="off",
                            expected_saving_kwh=power_kw * 0.5,
                            reason=f"Peak tariff + low occupancy ({occ_prob:.0%})"
                        ))

        return schedule

    def compute_savings(self, schedule: List[EnergySchedule]) -> dict:
        """Compute total energy and cost savings from a schedule."""
        total_kwh = sum(s.expected_saving_kwh for s in schedule)
        peak_savings = sum(s.expected_saving_kwh for s in schedule
                           if s.hour in self.tariff_peak_hours)

        return {
            "total_kwh_saved": round(total_kwh, 2),
            "peak_kwh_saved": round(peak_savings, 2),
            "estimated_cost_saving_gbp": round(total_kwh * 0.28, 2),  # 28p/kWh
            "co2_saved_kg": round(total_kwh * 0.233, 2),  # UK grid factor
            "n_actions": len(schedule),
        }


if __name__ == "__main__":
    print("Energy Optimiser Demo")
    optimiser = EnergyOptimiser()

    # Simulate 24-hour occupancy predictions
    np.random.seed(42)
    hours = list(range(24))
    occupancy_probs = []
    for h in hours:
        if 7 <= h <= 8 or 18 <= h <= 23:
            occupancy_probs.append(np.random.uniform(0.6, 0.9))
        elif 9 <= h <= 17:
            occupancy_probs.append(np.random.uniform(0.1, 0.3))
        else:
            occupancy_probs.append(np.random.uniform(0.0, 0.1))

    schedule = optimiser.optimise_schedule(occupancy_probs)
    savings = optimiser.compute_savings(schedule)

    print(f"Schedule actions: {savings['n_actions']}")
    print(f"Energy saved: {savings['total_kwh_saved']} kWh")
    print(f"Cost saved: £{savings['estimated_cost_saving_gbp']}")
    print(f"CO2 saved: {savings['co2_saved_kg']} kg")
    print(f"\nSample actions:")
    for action in schedule[:5]:
        print(f"  Hour {action.hour:02d}:00 — {action.device} {action.action} "
              f"(save {action.expected_saving_kwh:.2f} kWh)")
