import numpy as np
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def initialize_centroids(X, k, method='random'):
    """初始化质心"""
    if method == 'random':
        indices = np.random.choice(X.shape[0], k, replace=False)
        centroids = X[indices]
    elif method == 'kmeans++':
        centroids = [X[np.random.randint(0, X.shape[0])]]
        for _ in range(1, k):
            distances = np.min([np.linalg.norm(X - c, axis=1)**2 for c in centroids], axis=0)
            probabilities = distances / np.sum(distances)
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids.append(X[j])
                    break
        centroids = np.array(centroids)
    return centroids

def k_means(X, k, max_iters=100, tol=1e-4, init_method='random'):
    """K-Means 算法"""
    centroids = initialize_centroids(X, k, method=init_method)
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return labels, centroids



def initialize_gmm(X, k, method='random'):
    """初始化 GMM 参数"""
    n, d = X.shape
    if method == 'random':
        means = X[np.random.choice(n, k, replace=False)]
    elif method == 'kmeans++':
        means = initialize_centroids(X, k, method='kmeans++')
    elif method == 'kmeans':
        # 使用 K-Means 初始化
        labels_kmeans, centroids_kmeans = k_means(X, k=k, init_method='kmeans++')
        means = centroids_kmeans  # 用 K-Means 的聚类中心初始化均值
    covariances = np.array([np.eye(d) for _ in range(k)])  # 初始化协方差矩阵
    weights = np.ones(k) / k  # 初始化权重
    return means, covariances, weights

def gmm_em(X, k, max_iters=100, tol=1e-4, cov_type='full', init_method='random'):
    n, d = X.shape
    means, covariances, weights = initialize_gmm(X, k, method=init_method)
    epsilon = 1e-6  # 正则化常数
    
    for _ in range(max_iters):
        # E-Step
        responsibilities = np.zeros((n, k))
        for i in range(k):
            try:
                responsibilities[:, i] = weights[i] * multivariate_normal.pdf(X, mean=means[i], cov=covariances[i])
            except np.linalg.LinAlgError:
                covariances[i] += epsilon * np.eye(d)  # 修正协方差矩阵为正定
                responsibilities[:, i] = weights[i] * multivariate_normal.pdf(X, mean=means[i], cov=covariances[i])
        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities_sum[responsibilities_sum == 0] = epsilon  # 避免除以 0
        responsibilities /= responsibilities_sum
        
        # M-Step
        N_k = responsibilities.sum(axis=0)
        weights = N_k / n
        means = np.dot(responsibilities.T, X) / N_k[:, None]
        
        if cov_type == 'diag':
            covariances = [(np.diag(np.dot(responsibilities[:, i] * (X - means[i]).T, (X - means[i])) / N_k[i]) + epsilon * np.eye(d)) for i in range(k)]
        elif cov_type == 'tied':
            cov = np.sum([np.dot(responsibilities[:, i] * (X - means[i]).T, (X - means[i])) for i in range(k)], axis=0) / n
            cov += epsilon * np.eye(d)
            covariances = np.tile(cov, (k, 1, 1))
        else:  # full
            covariances = [(np.dot(responsibilities[:, i] * (X - means[i]).T, (X - means[i])) / N_k[i] + epsilon * np.eye(d)) for i in range(k)]
        
        # 检查收敛条件
        if np.linalg.norm(weights - weights) < tol:
            break
    
    return responsibilities.argmax(axis=1), means, covariances



def clustering_accuracy(true_labels, predicted_labels):
    """计算聚类精度"""
    n_clusters = len(np.unique(true_labels))
    cost_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int32)
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost_matrix[i, j] = np.sum((true_labels == i) & (predicted_labels == j))
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    return cost_matrix[row_ind, col_ind].sum() / true_labels.size

def load_data(train_file, test_file):
    train_data = np.loadtxt(train_file, delimiter=',', skiprows=1)
    test_data = np.loadtxt(test_file, delimiter=',', skiprows=1)
    X_train = train_data[:, 1:]  # 特征
    y_train = train_data[:, 0]   # 标签
    X_test = test_data[:, 1:]    # 特征
    y_test = test_data[:, 0]     # 标签
    return X_train, y_train, X_test, y_test




# 加载数据
X_train, y_train, X_test, y_test = load_data('mnist_train.csv', 'mnist_test.csv')
# 数据归一化
X_train = X_train / 255.0
X_test = X_test / 255.0
#K-Means
repeats = 10
k = 10
train_accuracies_random = []
train_accuracies_kmeanspp = []

# for i in range(repeats):
#     # Random 初始化
#     labels_train_random, _ = k_means(X_train, k=k, init_method='random')
#     acc_train_random = clustering_accuracy(y_train, labels_train_random)
#     train_accuracies_random.append(acc_train_random)
#     print(f"Iteration {i+1}, K-Means Accuracy, Random Init: {acc_train_random:.4f}")
    
#     # K-Means++ 初始化
#     labels_train_kmeanspp, _ = k_means(X_train, k=k, init_method='kmeans++')
#     acc_train_kmeanspp = clustering_accuracy(y_train, labels_train_kmeanspp)
#     train_accuracies_kmeanspp.append(acc_train_kmeanspp)
#     print(f"Iteration {i+1}, K-Means Accuracy, K-Means++ Init: {acc_train_kmeanspp:.4f}")

# # 结果可视化
# import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 6))
# plt.plot(range(1, repeats + 1), train_accuracies_random, label='Random Init', marker='o')
# plt.plot(range(1, repeats + 1), train_accuracies_kmeanspp, label='K-Means++ Init', marker='s')
# plt.title('K-Means Training Accuracy')
# plt.xlabel('Iteration')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()
    





# #GMM
# iterations = [10, 30, 50]

# # 存储实验结果
# results_iterations = {it: [] for it in iterations}
# results_initializations = {'random': [], 'kmeans': []}
# results_covariances = {'diag': [], 'full': [], 'tied': []}

# # 探究不同迭代次数
# print("Experimenting with different iterations:")
# for it in iterations:
#     print(f"--- Iterations: {it} ---")
#     for i in range(repeats):
#         labels_gmm, _, _ = gmm_em(X_train, k=k, cov_type='full', init_method='kmeans', max_iters=it)
#         acc = clustering_accuracy(y_train, labels_gmm)
#         results_iterations[it].append(acc)
#         print(f"Run {i + 1}/{repeats}, Accuracy: {acc:.4f}")
#     print(f"Average Accuracy for {it} Iterations: {np.mean(results_iterations[it]):.4f}\n")

# # 探究不同初始化方法
# print("Experimenting with different initialization methods:")
# for init_method in results_initializations.keys():
#     print(f"--- Initialization Method: {init_method} ---")
#     for i in range(repeats):
#         labels_gmm, _, _ = gmm_em(X_train, k=k, cov_type='full', init_method=init_method, max_iters=30)
#         acc = clustering_accuracy(y_train, labels_gmm)
#         results_initializations[init_method].append(acc)
#         print(f"Run {i + 1}/{repeats}, Accuracy: {acc:.4f}")
#     print(f"Average Accuracy for {init_method.capitalize()} Initialization: {np.mean(results_initializations[init_method]):.4f}\n")

# # 探究不同协方差矩阵
# print("Experimenting with different covariance matrices:")
# for cov_type in results_covariances.keys():
#     print(f"--- Covariance Matrix: {cov_type} ---")
#     for i in range(repeats):
#         labels_gmm, _, _ = gmm_em(X_train, k=k, cov_type=cov_type, init_method='kmeans', max_iters=30)
#         acc = clustering_accuracy(y_train, labels_gmm)
#         results_covariances[cov_type].append(acc)
#         print(f"Run {i + 1}/{repeats}, Accuracy: {acc:.4f}")
#     print(f"Average Accuracy for {cov_type.capitalize()} Covariance: {np.mean(results_covariances[cov_type]):.4f}\n")

# # 可视化结果
# plt.figure(figsize=(16, 8))

# # 1. 不同迭代次数
# plt.subplot(1, 3, 1)
# for it, accs in results_iterations.items():
#     plt.plot(range(1, repeats + 1), accs, label=f'{it} Iterations', marker='o')
# plt.title('GMM Accuracy with Different Iterations')
# plt.xlabel('Experiment Index')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(alpha=0.3)

# # 2. 不同初始化方法
# plt.subplot(1, 3, 2)
# for init_method, accs in results_initializations.items():
#     plt.plot(range(1, repeats + 1), accs, label=f'{init_method.capitalize()} Init', marker='s')
# plt.title('GMM Accuracy with Different Initializations')
# plt.xlabel('Experiment Index')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(alpha=0.3)

# # 3. 不同协方差矩阵
# plt.subplot(1, 3, 3)
# for cov_type, accs in results_covariances.items():
#     plt.plot(range(1, repeats + 1), accs, label=f'{cov_type.capitalize()} Covariance', marker='^')
# plt.title('GMM Accuracy with Different Covariance Matrices')
# plt.xlabel('Experiment Index')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(alpha=0.3)

# plt.tight_layout()
# plt.show()








import matplotlib.pyplot as plt

def plot_clusters(X, labels, title, pca_components=2):
    """
    可视化聚类结果
    :param X: 数据 (n_samples, n_features)
    :param labels: 聚类标签 (n_samples,)
    :param title: 图标题
    :param pca_components: 降维后的维度 (默认降到2维)
    """
    # 使用PCA降维
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X)
    
    # 绘制聚类结果
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=5)
    plt.colorbar(scatter, label='Cluster Label')
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(alpha=0.3)
    plt.show()

# 绘制K-Means聚类结果
labels_kmeans, _ = k_means(X_train, k=10, init_method='kmeans++')
plot_clusters(X_train, labels_kmeans, "K-Means Clustering (PCA)")

# # 绘制GMM聚类结果 (以随机初始化为例)
# labels_gmm_random, _, _ = gmm_em(X_train, k=10, cov_type='full', init_method='random')
# plot_clusters(X_train, labels_gmm_random, "GMM Clustering (Random Init, PCA)")

# 绘制GMM聚类结果 (以K-Means初始化为例)
# labels_gmm_kmeans, _, _ = gmm_em(X_train, k=10, cov_type='full', init_method='kmeans')
# plot_clusters(X_train, labels_gmm_kmeans, "GMM Clustering (K-Means Init, PCA)")



# results_covariances = {'diag': [], 'full': [], 'tied': []}

# # 使用 PCA 降维
# pca = PCA(n_components=50)  # 降维到50维
# X_train_pca = pca.fit_transform(X_train)

# for cov_type in results_covariances.keys():
#     print(f"--- Covariance Matrix: {cov_type} ---")
#     for i in range(repeats):
#         labels_gmm, _, _ = gmm_em(X_train_pca, k=k, cov_type=cov_type, init_method='kmeans', max_iters=50)
#         acc = clustering_accuracy(y_train, labels_gmm)
#         results_covariances[cov_type].append(acc)
#         print(f"Run {i + 1}/{repeats}, Accuracy: {acc:.4f}")
#     print(f"Average Accuracy for {cov_type.capitalize()} Covariance: {np.mean(results_covariances[cov_type]):.4f}\n")
# # 3. 不同协方差矩阵
# plt.figure(figsize=(8, 6))
# for cov_type, accs in results_covariances.items():
#     plt.plot(range(1, repeats + 1), accs, label=f'{cov_type.capitalize()} Covariance', marker='^')
# plt.title('GMM Accuracy with Different Covariance Matrices')
# plt.xlabel('Experiment Index')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(alpha=0.3)

# plt.tight_layout()
# plt.show()