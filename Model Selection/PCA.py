from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def fit_pca(dataset, n_components=8, sample_size=20000):
    pixels = []
    indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)

    for i in indices:
        img, _ = dataset[i]  # (13, 64, 64)
        flat = img.reshape(13, -1).T.numpy()
        pixels.append(flat)

    pixels = np.vstack(pixels)
    print(f"Fitting PCA on {pixels.shape[0]} pixels...")

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(pixels)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"Explained variance: {cum_var}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_components + 1), cum_var, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.show()

    return pca