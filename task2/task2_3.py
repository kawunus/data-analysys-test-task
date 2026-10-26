import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

trade_data = pd.read_excel("task2/data/data.xlsx", sheet_name="Годовые данные")
usd_data = pd.read_excel("task2/data/usd.xlsx", sheet_name="Годовые средние")

data = pd.merge(trade_data, usd_data, on="Год")
data = data[data["Год"].between(2008, 2021)]

Path("task2/plots/correlation").mkdir(parents=True, exist_ok=True)

pairs = [
    ("Импорт, тыс. долл. США", "Импорт арбузов", "import"),
    ("Экспорт, тыс. долл. США", "Экспорт арбузов", "export")
]

for column, name, file_name in pairs:
    x = data["Курс BYN/USD"].astype(float).to_numpy()
    y = data[column].astype(float).to_numpy()

    r, p = pearsonr(x, y)

    x_change = data["Курс BYN/USD"].diff().dropna()
    y_change = data[column].diff().dropna()
    r_change, p_change = pearsonr(x_change, y_change)

    print(name, "r =", round(r, 3), "p =", round(p, 4))
    print(name, "изменения: r =", round(r_change, 3), "p =", round(p_change, 4))

    slope, intercept = np.polyfit(x, y, 1)
    order = np.argsort(x)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="Годовые значения")
    plt.plot(x[order], (intercept + slope * x)[order], "--", label="Линейный тренд")

    plt.title(f"{name} и курс доллара\nr = {r:.2f}, p = {p:.4f}")
    plt.xlabel("Курс BYN/USD")
    plt.ylabel("Стоимость, тыс. долларов США")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"task2/plots/correlation/{file_name}.png", dpi=150)
    plt.close()