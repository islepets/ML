import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

df = pd.read_csv('nysk.csv')

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Метод снижает размерность, находит новые ортогональные оси в данных, первая компонента
# объясняет максимальную дисперсию, вторая следующую и тд
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_scaled)


#Для оценки моделей использовалось 3 метрики:
# Silhouette Score - значения лежат от -1 до 1, формула (b - a) / max(a, b)
# где а - среднее расстояние до объектов в том же кластере
# b - среднее расстояние до объектов в ближайшем кластере

#Calinski-Harabasz Score
# Отношение межкластерной дисперсии к внутрикластерной

# Davies-Bouldin Score
# среднее сходство между кластерами

#=======================================================================================================================
# K-Means
#=======================================================================================================================
inertia = [] # сумма квадратов расстояний до ближайшего центроида
silhouette_scores = []
calinski_harabasz_scores = []
davies_bouldin_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_scaled, labels))
    calinski_harabasz_scores.append(calinski_harabasz_score(df_scaled, labels))
    davies_bouldin_scores.append(davies_bouldin_score(df_scaled, labels))

optimal_k_silhouette_scores = k_range[np.argmax(silhouette_scores)]
optimal_k_calinski_harabasz_scores = k_range[np.argmax(calinski_harabasz_scores)]
optimal_k_davies_bouldin_scores = k_range[np.argmin(davies_bouldin_scores)]

print("K-Means")
print(f"Оптимальное k (Silhouette): {optimal_k_silhouette_scores}")
print(f"Оптимальное k (Calinski-Harabasz): {optimal_k_calinski_harabasz_scores}")
print(f"Оптимальное k (Davies-Bouldin): {optimal_k_davies_bouldin_scores}")


kmeans_params = [2, 3, 4, 5]
labels_for_k = []
fig, axes = plt.subplots(2, 2, figsize=(13, 6))
for i, k in enumerate(kmeans_params):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_temp.fit_predict(df_scaled)
    labels_for_k.append(labels)

    row = i // 2
    col = i % 2

    scatter = axes[row, col].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='spring')
    axes[row, col].set_title(f'KMeans k={k}')
    axes[row, col].set_xlabel('PC1')
    axes[row, col].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[row, col])

plt.tight_layout()
plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
# Silhouette scores
ax1.plot(k_range, silhouette_scores, 'ro-')
ax1.set_xlabel('Количество кластеров')
ax1.set_ylabel('Silhouette Score')
ax1.set_title('Silhouette Score для K-Means')
ax1.grid(True)

# Calinski-Harabasz scores
ax2.plot(k_range, calinski_harabasz_scores, 'go-')
ax2.set_xlabel('Количество кластеров')
ax2.set_ylabel('Calinski-Harabasz Score')
ax2.set_title('Calinski-Harabasz Score для K-Means')
ax2.grid(True)

# Davies-Bouldin scores
ax3.plot(k_range, davies_bouldin_scores, 'mo-')
ax3.set_xlabel('Количество кластеров')
ax3.set_ylabel('Davies-Bouldin Score')
ax3.set_title('Davies-Bouldin Score для K-Means')
ax3.grid(True)

plt.tight_layout()
plt.show()


#=======================================================================================================================
# DBSCAN
#=======================================================================================================================

dbscan_params = [
    {'eps': 0.3, 'min_samples': 1},
    {'eps': 0.5, 'min_samples': 1},
    {'eps': 0.7, 'min_samples': 1},
    {'eps': 0.3, 'min_samples': 2},
    {'eps': 0.5, 'min_samples': 3},
    {'eps': 0.7, 'min_samples': 5},
]

silhouette_scores_dbscan = []
calinski_harabasz_scores_dbscan = []
davies_bouldin_scores_dbscan = []
labels_dbscan = []
cluster_dbscan = []
ratios = []
valid_indices = []

# Инициализируем списки с значениями по умолчанию для всех параметров
for i in range(len(dbscan_params)):
    silhouette_scores_dbscan.append(-1)  # silhouette_score не может быть < -1
    calinski_harabasz_scores_dbscan.append(0)
    davies_bouldin_scores_dbscan.append(float('inf'))

for i, params in enumerate(dbscan_params):
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    labels = dbscan.fit_predict(df_scaled)

    uniq_labels = set(labels)
    clusters = len(uniq_labels) - (1 if -1 in uniq_labels else 0)
    n_noise = list(labels).count(-1)
    ratio = n_noise / len(labels)

    labels_dbscan.append(labels)
    cluster_dbscan.append(clusters)
    ratios.append(ratio)

    if clusters > 1:
        silhouette_scores_dbscan[i] = silhouette_score(df_scaled, labels)
        calinski_harabasz_scores_dbscan[i] = calinski_harabasz_score(df_scaled, labels)
        davies_bouldin_scores_dbscan[i] = davies_bouldin_score(df_scaled, labels)
        valid_indices.append(i)

if valid_indices:
    best_silhouette_idx = valid_indices[np.argmax([silhouette_scores_dbscan[i] for i in valid_indices])]
    best_calinski_idx = valid_indices[np.argmax([calinski_harabasz_scores_dbscan[i] for i in valid_indices])]
    best_davies_idx = valid_indices[np.argmin([davies_bouldin_scores_dbscan[i] for i in valid_indices])]

    print("DBSCAN")
    print(f"Оптимальные параметры (Silhouette): {dbscan_params[best_silhouette_idx]}")
    print(f"Оптимальные параметры (Calinski-Harabasz): {dbscan_params[best_calinski_idx]}")
    print(f"Оптимальные параметры (Davies-Bouldin): {dbscan_params[best_davies_idx]}")


fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for i, params in enumerate(dbscan_params):
    row = i // 3
    col = i % 3
    labels = labels_dbscan[i]

    scatter = axes[row, col].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
    axes[row, col].set_title(f'DBSCAN: eps={params["eps"]}, min_samples={params["min_samples"]}\n'
                             f'Кластеров: {cluster_dbscan[i]}, Шум: {ratios[i]:.1%}')
    axes[row, col].set_xlabel('PC1')
    axes[row, col].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[row, col])

plt.tight_layout()
plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Создаем подписи для оси X
param_labels = [f"eps={p['eps']}\nmin={p['min_samples']}" for p in dbscan_params]

# Silhouette scores
ax1.plot(range(len(dbscan_params)), silhouette_scores_dbscan, 'ro-')
ax1.set_xticks(range(len(dbscan_params)))
ax1.set_xticklabels(param_labels, rotation=45)
ax1.set_xlabel('Параметры DBSCAN')
ax1.set_ylabel('Silhouette Score')
ax1.set_title('Silhouette Score для DBSCAN')
ax1.grid(True)

# Calinski-Harabasz scores
ax2.plot(range(len(dbscan_params)), calinski_harabasz_scores_dbscan, 'go-')
ax2.set_xticks(range(len(dbscan_params)))
ax2.set_xticklabels(param_labels, rotation=45)
ax2.set_xlabel('Параметры DBSCAN')
ax2.set_ylabel('Calinski-Harabasz Score')
ax2.set_title('Calinski-Harabasz Score для DBSCAN')
ax2.grid(True)

# Davies-Bouldin scores
ax3.plot(range(len(dbscan_params)), davies_bouldin_scores_dbscan, 'mo-')
ax3.set_xticks(range(len(dbscan_params)))
ax3.set_xticklabels(param_labels, rotation=45)
ax3.set_xlabel('Параметры DBSCAN')
ax3.set_ylabel('Davies-Bouldin Score')
ax3.set_title('Davies-Bouldin Score для DBSCAN')
ax3.grid(True)

plt.tight_layout()
plt.show()

#=======================================================================================================================
# Agglomerative  clustering
#=======================================================================================================================

silhouette_scores_agglomerative = []
calinski_harabasz_scores_agglomerative = []
davies_bouldin_scores_agglomerative = []
k_range_agglomerative = range(2, 11)

for k in k_range_agglomerative:
    hierarchical = AgglomerativeClustering(n_clusters=k)
    labels = hierarchical.fit_predict(df_scaled)
    silhouette_scores_agglomerative.append(silhouette_score(df_scaled, labels))
    calinski_harabasz_scores_agglomerative.append(calinski_harabasz_score(df_scaled, labels))
    davies_bouldin_scores_agglomerative.append(davies_bouldin_score(df_scaled, labels))

optimal_k_silhouette_scores_agglomerative = k_range_agglomerative[np.argmax(silhouette_scores_agglomerative)]
optimal_k_calinski_harabasz_scores_agglomerative = k_range_agglomerative[np.argmax(calinski_harabasz_scores_agglomerative)]
optimal_k_davies_bouldin_scores_agglomerative = k_range_agglomerative[np.argmin(davies_bouldin_scores_agglomerative)]

print("Agglomerative")
print(f"Оптимальное k (Silhouette): {optimal_k_silhouette_scores_agglomerative}")
print(f"Оптимальное k (Calinski-Harabasz): {optimal_k_calinski_harabasz_scores_agglomerative}")
print(f"Оптимальное k (Davies-Bouldin): {optimal_k_davies_bouldin_scores_agglomerative}")


hierarchical_params = [2, 3, 4, 5]
fig, axes = plt.subplots(2, 2, figsize=(13, 6))
for i, k in enumerate(hierarchical_params):
    hierarchical_temp = AgglomerativeClustering(n_clusters=k)
    labels = hierarchical_temp.fit_predict(df_scaled)

    row = i // 2
    col = i % 2

    scatter = axes[row, col].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='plasma')
    axes[row, col].set_title(f'Agglomerative k={k}')
    axes[row, col].set_xlabel('PC1')
    axes[row, col].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[row, col])

plt.tight_layout()
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
# Silhouette scores
ax1.plot(k_range_agglomerative, silhouette_scores_agglomerative, 'ro-')
ax1.set_xlabel('Количество кластеров')
ax1.set_ylabel('Silhouette Score')
ax1.set_title('Silhouette Score для Agglomerative')
ax1.grid(True)

# Calinski-Harabasz scores
ax2.plot(k_range_agglomerative, calinski_harabasz_scores_agglomerative, 'go-')
ax2.set_xlabel('Количество кластеров')
ax2.set_ylabel('Calinski-Harabasz Score')
ax2.set_title('Calinski-Harabasz Score для Agglomerative')
ax2.grid(True)

# Davies-Bouldin scores
ax3.plot(k_range_agglomerative, davies_bouldin_scores_agglomerative, 'mo-')
ax3.set_xlabel('Количество кластеров')
ax3.set_ylabel('Davies-Bouldin Score')
ax3.set_title('Davies-Bouldin Score для Agglomerative')
ax3.grid(True)

plt.tight_layout()
plt.show()

#=======================================================================================================================
# Сравнение
#=======================================================================================================================

# K-Means лучшие результаты
kmeans_best_silhouette = max(silhouette_scores)
kmeans_best_calinski = max(calinski_harabasz_scores)
kmeans_best_davies = min(davies_bouldin_scores)

print(f"K-Means:")
print(f"  Silhouette: {kmeans_best_silhouette:.4f}")
print(f"  Calinski-Harabasz: {kmeans_best_calinski:.4f}")
print(f"  Davies-Bouldin: {kmeans_best_davies:.4f}")

# DBSCAN лучшие результаты
if valid_indices:
    dbscan_best_silhouette = max([silhouette_scores_dbscan[i] for i in valid_indices])
    dbscan_best_calinski = max([calinski_harabasz_scores_dbscan[i] for i in valid_indices])
    dbscan_best_davies = min([davies_bouldin_scores_dbscan[i] for i in valid_indices])

    print(f"DBSCAN:")
    print(f"  Silhouette: {dbscan_best_silhouette:.4f}")
    print(f"  Calinski-Harabasz: {dbscan_best_calinski:.4f}")
    print(f"  Davies-Bouldin: {dbscan_best_davies:.4f}")

# Agglomerative лучшие результаты
agglomerative_best_silhouette = max(silhouette_scores_agglomerative)
agglomerative_best_calinski = max(calinski_harabasz_scores_agglomerative)
agglomerative_best_davies = min(davies_bouldin_scores_agglomerative)

print(f"Agglomerative Clustering:")
print(f"  Silhouette: {agglomerative_best_silhouette:.4f}")
print(f"  Calinski-Harabasz: {agglomerative_best_calinski:.4f}")
print(f"  Davies-Bouldin: {agglomerative_best_davies:.4f}")


# Создаем словарь для сравнения
algorithms_comparison = {
    'K-Means': {
        'silhouette': kmeans_best_silhouette,
        'calinski': kmeans_best_calinski,
        'davies': kmeans_best_davies,
        'params': f"n_clusters={optimal_k_silhouette_scores}, random_state=42, n_init=10"
    },
    'Agglomerative': {
        'silhouette': agglomerative_best_silhouette,
        'calinski': agglomerative_best_calinski,
        'davies': agglomerative_best_davies,
        'params': f"n_clusters={optimal_k_silhouette_scores}"
    }
}

# Добавляем DBSCAN если есть валидные результаты
if valid_indices:
    algorithms_comparison['DBSCAN'] = {
        'silhouette': dbscan_best_silhouette,
        'calinski': dbscan_best_calinski,
        'davies': dbscan_best_davies,
        'params': f"eps={dbscan_params[best_silhouette_idx]['eps']}, min_samples={dbscan_params[best_silhouette_idx]['min_samples']}"
    }

# Выбираем лучший алгоритм по комбинации метрик
best_algorithm = None
best_score = -float('inf')

for algo, scores in algorithms_comparison.items():
    # Комбинированная оценка (чем выше, тем лучше)
    combined_score = (scores['silhouette'] +
                      scores['calinski'] / 1000 +  # нормализуем Calinski-Harabasz
                      (1 - scores['davies']))  # инвертируем Davies-Bouldin

    if combined_score > best_score:
        best_score = combined_score
        best_algorithm = algo

print(f"\nЛучший алгоритм: {best_algorithm}")
print(f"Параметры: {algorithms_comparison[best_algorithm]['params']}")
print(f"Метрики: ")
print(f"   • Silhouette Score: {algorithms_comparison[best_algorithm]['silhouette']:.4f}")
print(f"   • Calinski-Harabasz Score: {algorithms_comparison[best_algorithm]['calinski']:.4f}")
print(f"   • Davies-Bouldin Score: {algorithms_comparison[best_algorithm]['davies']:.4f}")
