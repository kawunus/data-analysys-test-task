import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_excel("task2/data/data.xlsx", sheet_name="Месячные данные")
columns = data.columns[2:6]

Path("task2/plots").mkdir(parents=True, exist_ok=True)

year_positions = data.groupby("Год").head(1).index.tolist()
year_labels = data.loc[year_positions, "Год"].astype(str).tolist()
month_labels = data["Месяц"].iloc[:12].str[:3].tolist()

for i, column in enumerate(columns, start=1):
    values = data[column].astype(float)

    additive = seasonal_decompose(values, model="additive", period=12)
    multiplicative = seasonal_decompose(values, model="multiplicative", period=12)

    additive_fit = additive.trend + additive.seasonal
    multiplicative_fit = multiplicative.trend * multiplicative.seasonal

    additive_mad = (values - additive_fit).abs().dropna().mean()
    multiplicative_mad = (values - multiplicative_fit).abs().dropna().mean()

    if additive_mad <= multiplicative_mad:
        result = additive
        fitted = additive_fit
        model_name = "аддитивная"
        mad = additive_mad
    else:
        result = multiplicative
        fitted = multiplicative_fit
        model_name = "мультипликативная"
        mad = multiplicative_mad

    residuals = values - fitted

    print(column, model_name, "MAD =", round(mad, 2))

    figure, axes = plt.subplots(2, 2, figsize=(16, 9))

    axes[0, 0].plot(values)
    axes[0, 0].set_title("Исходный ряд")

    axes[0, 1].plot(result.trend)
    axes[0, 1].set_title("Тренд")

    axes[1, 0].plot(range(12), result.seasonal.iloc[:12], marker="o")
    axes[1, 0].set_title("Сезонность за один год")
    axes[1, 0].set_xticks(range(12), month_labels)

    axes[1, 1].plot(residuals, marker=".", linewidth=0.7)
    axes[1, 1].axhline(0, color="red", linestyle="--")
    axes[1, 1].set_title("Остатки")
    axes[1, 1].set_ylabel("Отклонение")

    for axis in [axes[0, 0], axes[0, 1], axes[1, 1]]:
        axis.set_xticks(year_positions, year_labels, rotation=45)
        axis.grid()

    axes[1, 0].grid()

    figure.suptitle(f"{column} — {model_name} модель, MAD = {mad:.2f}")
    figure.tight_layout()
    figure.savefig(f"task2/plots/decomposition_{i}.png", dpi=150)
    plt.close(figure)