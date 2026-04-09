import random
import csv
from pathlib import Path

INTERVAL_S = 300  # 5 minutes per data point

def simulate_db_disk(
  initial_gb: float = 500.0,
  target_days: float = 90.0,
  noise_pct: float = 0.30,
  regain_ratio: float = 0.30,
  regain_chance: float = 0.30,
  output_csv: str = "disk_sim_5min.csv",
):
  total_steps = int(target_days * 86400 / INTERVAL_S)  # 25,920

  # Auto-calibrate: solve for base depletion so E[net loss] = initial_gb
  net_factor = 1 - (regain_ratio * regain_chance * 0.5)
  base_dep = initial_gb / (total_steps * net_factor)

  print(f"Steps: {total_steps:,}  |  Base depletion: {base_dep * 288:.3f} GB/day")

  disk = initial_gb
  rows = [(0, 0, 0.0, round(disk, 4))]

  for step in range(1, total_steps + 1):
    if disk <= 0:
      rows.append((step, step * INTERVAL_S,
                  round(step * INTERVAL_S / 86400, 4), 0.0))
      print(f"Depleted at step {step} (day {step*INTERVAL_S/86400:.1f})")
      break

    noise   = 1 + (random.random() * 2 - 1) * noise_pct
    deplete = base_dep * noise
    disk   -= deplete

    if random.random() < regain_chance:
      disk += deplete * regain_ratio * random.random()

    disk = max(0.0, disk)
    rows.append((step, step * INTERVAL_S,
                round(step * INTERVAL_S / 86400, 4), round(disk, 4)))

  path = Path(output_csv)

  with path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "elapsed_s", "elapsed_days", "disk_gb"])
    w.writerows(rows)

  print(f"Written {len(rows):,} rows → {path}")
  return rows


if __name__ == "__main__":
  simulate_db_disk(
    initial_gb=500,
    target_days=90,
    noise_pct=0.30,
    regain_ratio=0.30,
    regain_chance=0.30,
  )