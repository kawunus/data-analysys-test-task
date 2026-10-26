import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


def xgboost_forecast(values, months):
    values = np.asarray(values, dtype=float)

    features = []
    target = []

    for j in range(12, len(values)):
        features.append([j % 12 + 1, values[j - 1], values[j - 12]])
        target.append(values[j])

    model = XGBRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.03,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    )
    model.fit(features, target)

    history = list(values)
    forecast = []

    for j in range(len(history), len(history) + months):
        month = j % 12 + 1
        next_value = model.predict([[month, history[-1], history[-12]]])[0]
        forecast.append(next_value)
        history.append(next_value)

    return np.array(forecast)


data = pd.read_excel("task2/data/data.xlsx", sheet_name="Месячные данные")
columns = data.columns[2:6]

Path("task2/plots").mkdir(parents=True, exist_ok=True)

year_positions = data.groupby("Год").head(1).index.tolist()
year_labels = data.loc[year_positions, "Год"].astype(str).tolist()

future_positions = np.arange(len(data), len(data) + 36)
year_positions += future_positions[::12].tolist()
year_labels += ["2022", "2023", "2024"]

for i, column in enumerate(columns, start=1):
    values = data[column].astype(float).to_numpy()

    train_values = values[:-36]
    actual = values[-36:].reshape(3, 12).sum(axis=1)

    test_model = SARIMAX(
        train_values,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )
    test_result = test_model.fit(disp=False)

    sarima_test = np.asarray(
        test_result.get_forecast(steps=36).predicted_mean
    )
    sarima_test = sarima_test.reshape(3, 12).sum(axis=1)
    sarima_mad = np.abs(actual - sarima_test).mean()

    xgb_test = xgboost_forecast(train_values, 36)
    xgb_test = xgb_test.reshape(3, 12).sum(axis=1)
    xgb_mad = np.abs(actual - xgb_test).mean()

    print(column, "SARIMA MAD =", round(sarima_mad, 2), "XGBoost MAD =", round(xgb_mad, 2))

    model = SARIMAX(
        values,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )
    result = model.fit(disp=False)

    forecast = result.get_forecast(steps=36)
    sarima_values = np.asarray(forecast.predicted_mean)
    interval = np.asarray(forecast.conf_int())

    xgb_values = xgboost_forecast(values, 36)

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(values)), values, label="Исходные данные")
    plt.plot(future_positions, sarima_values, label="Прогноз SARIMA")
    plt.fill_between(
        future_positions,
        interval[:, 0],
        interval[:, 1],
        alpha=0.2,
        label="95% прогнозный интервал"
    )
    plt.axvline(len(data), color="gray", linestyle=":")
    plt.title(column + " — SARIMA")
    plt.xlabel("Год")
    plt.xticks(year_positions, year_labels, rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"task2/plots/forecast_sarima_{i}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(values)), values, label="Исходные данные")
    plt.plot(future_positions, xgb_values, label="Прогноз XGBoost")
    plt.axvline(len(data), color="gray", linestyle=":")
    plt.title(column + " — XGBoost")
    plt.xlabel("Год")
    plt.xticks(year_positions, year_labels, rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"task2/plots/forecast_xgboost_{i}.png", dpi=150)
    plt.close()

    sarima_annual = sarima_values.reshape(3, 12).sum(axis=1)
    xgb_annual = xgb_values.reshape(3, 12).sum(axis=1)

    for year, sarima_value, xgb_value in zip(range(2022, 2025), sarima_annual, xgb_annual):
        print(year, "SARIMA =", round(sarima_value, 2), "XGBoost =", round(xgb_value, 2))