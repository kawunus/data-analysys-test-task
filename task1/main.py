import pandas as pd
import matplotlib.pyplot as plt

import_data = pd.read_excel('~/Documents/test-task-data-analisys/task1/data/import.xlsx', header = 3, thousands="\xa0", decimal=",")
export_data = pd.read_excel('~/Documents/test-task-data-analisys/task1/data/export.xlsx', header = 3, thousands="\xa0", decimal=",")

# у автомобилей в экспорте ед. измерения в штуках, а все остальное в тысячах
years = export_data.columns[1:]
cars = export_data["Unnamed: 0"] == "Легковые автомобили, штук "
export_data.loc[cars, years] = export_data.loc[cars, years] / 1000

print(import_data["Unnamed: 0"])
print(export_data["Unnamed: 0"])

print("\nИзменение импорта:")
for _, row in import_data.iterrows():
        change = (row[2021] / row[2000] - 1) * 100
        print(row["Unnamed: 0"], "изменение на", round(change, 1),"%")

print("\nИзменение экспорта:")
for _, row in export_data.iterrows():
        change = (row[2021] / row[2000] - 1) * 100
        print(row["Unnamed: 0"], "изменение на", round(change, 1),"%")