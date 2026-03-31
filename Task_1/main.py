import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris
from scipy.stats import mode

iris = load_iris()

species_names = iris.target_names
print(species_names)
print(len(species_names)) # 3 grupy

X = iris.data
y_true = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
print("Statistics:")
print(df.describe())

# n_clusters=3 - 3 grupy
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

labels = np.zeros_like(y_kmeans)
# przypisywanie grupom numerów zgodnych z oryginalnymi danymi
for i in range(3):
    mask = (y_kmeans == i)
    if np.any(mask):
        labels[mask] = mode(y_true[mask])[0]

# average='macro' - policzenie średniej dla wszystkich gatunków
print("\nMETRICS")
print(f"Accuracy:  {accuracy_score(y_true, labels):.2f}")
print(f"Precision: {precision_score(y_true, labels, average='macro'):.2f}")
print(f"Recall:    {recall_score(y_true, labels, average='macro'):.2f}")
print(f"F1-score:  {f1_score(y_true, labels, average='macro'):.2f}")

# n_components=2 - spłaszczenie wymiarów do 2
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X) # przeliczenie danych na X i Y

plt.figure(figsize=(10, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_kmeans)
plt.savefig("vis.png")
plt.show()