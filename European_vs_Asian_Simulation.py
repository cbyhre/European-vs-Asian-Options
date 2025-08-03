import numpy as np
import matplotlib.pyplot as plt

#Assumptions
S0 = 100
K = 100
r = 0.0
sigma = 1.5
total_days = 30
steps_per_day = 288
dt = 1 / 365 / steps_per_day
N = 100_000
batch_size = 1_000
asian_window_minutes = 120

european_values = []
asian_values = []
days_remaining = list(range(total_days, 0, -1))  # 30 → 1

for remaining_days in days_remaining:
    steps_remaining = remaining_days * steps_per_day
    asian_window_steps = int(asian_window_minutes * (steps_per_day / 1440))
    if steps_remaining < asian_window_steps:
        continue

    euro_payoffs = []
    asian_payoffs = []

    for _ in range(N // batch_size):
        z = np.random.randn(batch_size, steps_remaining).astype(np.float32)
        increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        log_paths = np.cumsum(increments, axis=1)
        paths = S0 * np.exp(log_paths)

        final_prices = paths[:, -1]
        euro = np.maximum(final_prices - K, 0)
        euro_payoffs.append(np.mean(euro))

        asian_avg = np.mean(paths[:, -asian_window_steps:], axis=1)
        asian = np.maximum(asian_avg - K, 0)
        asian_payoffs.append(np.mean(asian))

    european_values.append(np.mean(euro_payoffs))
    asian_values.append(np.mean(asian_payoffs))

percent_diff = [(e - a) / e * 100 if e != 0 else 0 for e, a in zip(european_values, asian_values)]

print(f"{'T to Maturity':>14} | {'European':>10} | {'Asian':>10} | {'% Difference':>13}")
print("-" * 52)
for t, euro, asian, pct in zip(days_remaining[:len(european_values)], european_values, asian_values, percent_diff):
    print(f"{t:14d} | ${euro:9.4f} | ${asian:9.4f} | {pct:10.4f}%")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(days_remaining[:len(european_values)], european_values, label="European Option", linewidth=2)
ax1.plot(days_remaining[:len(asian_values)], asian_values, label="Asian (TWAP) Option", linestyle='--', linewidth=2)
ax1.set_ylabel("Option Price")
ax1.set_title(f"Option Price vs Time to Maturity (Volatility = {sigma}, Asian Window = {asian_window_minutes} Minutes)")
ax1.legend()
ax1.grid(True)

ax2.plot(days_remaining[:len(percent_diff)], percent_diff, color='purple', linewidth=2)
ax2.set_ylabel("Percent Difference")
ax2.set_title("European Overpricing vs Asian")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))
ax2.set_xlabel("Time to Maturity (Days)")
ax2.grid(True)
ax2.invert_xaxis()

plt.tight_layout()
plt.show()
