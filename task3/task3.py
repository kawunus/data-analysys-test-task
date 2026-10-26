import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

data = pd.DataFrame([
    ["Россия", 15, 7, 25, 18, 15, 10, 4, 6],
    ["Казахстан", 18, 10, 24, 16, 13, 9, 4, 6],
    ["Армения", 14, 8, 22, 19, 16, 10, 5, 6],
    ["Сербия", 13, 8, 23, 20, 16, 10, 4, 6],
    ["Турция", 20, 6, 18, 18, 15, 13, 4, 6],
    ["Азербайджан", 16, 9, 21, 18, 15, 12, 4, 5],
    ["Кыргызстан", 17, 10, 22, 17, 13, 10, 4, 7],
    ["Таджикистан", 19, 9, 20, 16, 12, 11, 4, 9],
    ["Бразилия", 37, 15, 8, 9, 6, 8, 4, 13],
    ["Индия", 32, 12, 10, 11, 7, 12, 6, 10],
    ["Узбекистан", 34, 13, 9, 10, 6, 10, 5, 13],
    ["Грузия", 33, 12, 8, 11, 7, 10, 6, 13],
    ["Аргентина", 38, 15, 7, 8, 5, 9, 4, 14],
    ["Вьетнам", 35, 10, 10, 12, 7, 11, 5, 10],
    ["Египет", 36, 12, 9, 9, 6, 10, 5, 13],
    ["Индонезия", 34, 13, 9, 12, 7, 11, 5, 9],
    ["Япония", 8, 4, 12, 31, 16, 15, 8, 6],
    ["Южная Корея", 9, 4, 13, 29, 15, 17, 7, 6],
    ["Китай", 10, 5, 18, 27, 14, 16, 5, 5],
    ["ОАЭ", 10, 5, 15, 22, 17, 20, 5, 6],
    ["Иран", 15, 8, 14, 22, 13, 17, 6, 5],
    ["Монголия", 24, 18, 14, 12, 8, 10, 4, 10],
    ["Пакистан", 30, 10, 12, 12, 8, 13, 5, 10],
    ["Таиланд", 28, 12, 10, 13, 9, 14, 5, 9]
], columns=[
    "Страна-партнёр",
    "Продовольствие и сельхозпродукция",
    "Древесина и изделия из неё",
    "Металлы и металлоизделия",
    "Машины и оборудование",
    "Автомобили и автокомпоненты",
    "Химическая продукция и полимеры",
    "Лекарства и медизделия",
    "Мебель и прочие товары"
])

products = data.columns[1:]
values = data[products]

print("Сумма долей по странам:")
print(data[["Страна-партнёр"]].assign(Сумма=values.sum(axis=1)).to_string(index=False))

scores = []

for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(values)
    score = silhouette_score(values, labels)
    scores.append(score)
    print("Кластеров:", k, "silhouette:", round(score, 2))

best_k = scores.index(max(scores)) + 2
print("Выбрано кластеров:", best_k)

model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
data["Кластер"] = model.fit_predict(values) + 1

print("\nСтраны по кластерам:")
print(data[["Страна-партнёр", "Кластер"]].to_string(index=False))

print("\nСредние доли товаров по кластерам:")
print(data.groupby("Кластер")[products].mean().round(1))

points = PCA(n_components=2).fit_transform(values)

for cluster in sorted(data["Кластер"].unique()):
    mask = data["Кластер"].to_numpy() == cluster
    plt.scatter(points[mask, 0], points[mask, 1], label=f"Кластер {cluster}")

for i, country in enumerate(data["Страна-партнёр"]):
    plt.annotate(country, (points[i, 0], points[i, 1]), fontsize=8)

plt.title(f"Структура торговли Беларуси со странами-партнёрами")
plt.xlabel("Ось 1 (PCA)")
plt.ylabel("Ось 2 (PCA)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("task3/plots/clusters.png", dpi=150)
plt.close()