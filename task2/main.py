from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def to_number(value):
    value = str(value).replace(" ", "")
    return pd.to_numeric(value, errors="coerce")


files = sorted(Path("task2/data").glob("*.xls"))
rows = []

for file in files:
    table = pd.read_excel(file, header=7)

    codes = table.iloc[:, 1].astype(str)
    codes = codes.str.replace(".0", "", regex=False)
    codes = codes.str.zfill(6)

    product = table[codes == "080711"]

    if product.empty:
        continue

    product = product.iloc[0]

    direction = file.stem.split()[0]
    years = file.stem.split()[1].split("-")
    unit = str(product.iloc[3]).strip()

    quantity_1 = to_number(product.iloc[4])
    value_1 = to_number(product.iloc[5])
    quantity_2 = to_number(product.iloc[7])
    value_2 = to_number(product.iloc[8])

    if unit == "кг":
        quantity_1 = quantity_1 / 1000
        quantity_2 = quantity_2 / 1000

    rows.append({
        "направление": direction,
        "год": int(years[0]),
        "количество, тонн": quantity_1,
        "стоимость, тыс. долларов": value_1
    })

    rows.append({
        "направление": direction,
        "год": int(years[1]),
        "количество, тонн": quantity_2,
        "стоимость, тыс. долларов": value_2
    })


result = pd.DataFrame(rows)

result = result.drop_duplicates(
    subset=["направление", "год"],
    keep="last"
)

result.loc[
    (result["направление"] == "импорт") &
    (result["год"] == 2008),
    ["количество, тонн", "стоимость, тыс. долларов"]
] = [4013.990, 745.5]

result.loc[
    (result["направление"] == "экспорт") &
    (result["год"] == 2008),
    ["количество, тонн", "стоимость, тыс. долларов"]
] = [3.212, 1.0]

result.loc[
    (result["направление"] == "импорт") &
    (result["год"] == 2012),
    ["количество, тонн", "стоимость, тыс. долларов"]
] = [10353.554, 2319.1]

result.loc[
    (result["направление"] == "экспорт") &
    (result["год"] == 2012),
    ["количество, тонн", "стоимость, тыс. долларов"]
] = [68.256, 26.6]

result = result.sort_values(["год", "направление"])

print(result.to_string(index=False))

import_data = result[result["направление"] == "импорт"]
export_data = result[result["направление"] == "экспорт"]

plt.figure(figsize=(10, 5))
plt.plot(import_data["год"], import_data["количество, тонн"], marker="o")
plt.title("Импорт арбузов")
plt.xlabel("Год")
plt.ylabel("Количество, тонн")
plt.grid()
plt.tight_layout()
plt.savefig("task2/plots/import_quantity.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(export_data["год"], export_data["количество, тонн"], marker="o")
plt.title("Экспорт арбузов")
plt.xlabel("Год")
plt.ylabel("Количество, тонн")
plt.grid()
plt.tight_layout()
plt.savefig("task2/plots/export_quantity.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(import_data["год"], import_data["стоимость, тыс. долларов"], marker="o")
plt.title("Стоимость импорта арбузов")
plt.xlabel("Год")
plt.ylabel("Тысяч долларов")
plt.grid()
plt.tight_layout()
plt.savefig("task2/plots/import_value.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(export_data["год"], export_data["стоимость, тыс. долларов"], marker="o")
plt.title("Стоимость экспорта арбузов")
plt.xlabel("Год")
plt.ylabel("Тысяч долларов")
plt.grid()
plt.tight_layout()
plt.savefig("task2/plots/export_value.png", dpi=150)
plt.close()