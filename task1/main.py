import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import os
import numpy as np

import_data = pd.read_excel('~/Documents/test-task-data-analisys/task1/data/import.xlsx', header = 3, thousands="\xa0", decimal=",")
export_data = pd.read_excel('~/Documents/test-task-data-analisys/task1/data/export.xlsx', header = 3, thousands="\xa0", decimal=",")

# у автомобилей в экспорте ед. измерения в штуках, а все остальное в тысячах
years = np.array(import_data.columns[1:], dtype=float)
t = np.arange(1, len(years) + 1)
cars = export_data["Unnamed: 0"] == "Легковые автомобили, штук "
export_data.loc[cars, years] = export_data.loc[cars, years] / 1000

print(import_data["Unnamed: 0"])
print(export_data["Unnamed: 0"])


for i in range(len(import_data)):
    values = import_data.iloc[i, 1:].astype(float).to_numpy()

    slope, intercept = np.polyfit(t, values, 1)
    trend = intercept + slope * t

    r2 = 1 - (
        ((values - trend) ** 2).sum()
        / ((values - values.mean()) ** 2).sum()
    )

    plt.figure(figsize=(10, 5))
    plt.plot(years, values, label="Значения")
    plt.plot(years, trend, "--", label=f"Тренд, R² = {r2:.2f}")

    plt.title(import_data.iloc[i, 0])
    plt.xlabel("Год")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"task1/plots/linear/import/import_{i}.png", dpi=150)
    plt.close()

for i in range(len(export_data)):
    values = export_data.iloc[i, 1:].astype(float).to_numpy()

    slope, intercept = np.polyfit(t, values, 1)
    trend = intercept + slope * t

    r2 = 1 - (
        ((values - trend) ** 2).sum()
        / ((values - values.mean()) ** 2).sum()
    )

    plt.figure(figsize=(10, 5))
    plt.plot(years, values, label="Значения")
    plt.plot(years, trend, "--", label=f"Тренд, R² = {r2:.2f}")

    plt.title(export_data.iloc[i, 0])
    plt.xlabel("Год")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"task1/plots/linear/export/export_{i}.png", dpi=150)
    plt.close()

pairs = [(4, 4), (2, 11), (11, 8), (12, 9)]

for i, j in pairs:
    x = import_data.iloc[i, 1:].astype(float)
    y = export_data.iloc[j, 1:].astype(float)

    r, p = pearsonr(x, y)

    print(
        import_data.iloc[i, 0],
        "r =", round(r, 2),
        "p =", round(p, 4)
    )