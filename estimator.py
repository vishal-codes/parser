# https://sustainablewebdesign.org/estimating-digital-emissions/#:~:text=0.081%20kWh/GB-,The%20final%20values,-we%20obtain%20for

DATA_CENTER_KWH_PER_GB = 0.055
NETWORK_KWH_PER_GB     = 0.059
DEVICE_KWH_PER_GB      = 0.080
CO2_PER_KWH            = 494 

def estimate(total_bytes: int) -> dict:
    """
    Estimate grams CO2 per view given total_bytes transferred.
    """
    gb = total_bytes / (1024 ** 3)
    energy_kwh = gb * (DATA_CENTER_KWH_PER_GB + NETWORK_KWH_PER_GB + DEVICE_KWH_PER_GB)
    grams = energy_kwh * CO2_PER_KWH
    return {"carbon_per_view": grams, "notes": f"{grams:.2f} g CO₂ per view"}
