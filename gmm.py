import numpy as np

class KMeans():
    def __init__(self, n_clusters = 1, n_init = 1, max_iter = 100):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = 100
    
    def _initialize_params(self, X):
        n_samples, n_features = X.shape
        centroids = X[np.random.choice(n_samples, size=self.n_clusters)]

    def _assign_clusters(centroids, X):
        n_samples, n_features = X.shape
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        assignments = np.zeros_like(distances)
        assignments[np.arange(n_samples),distances.argmin(axis = 1)] = 1
    


class GaussianMixture():
    def __init__(self, n_components=1, max_iter=100):
        self.k = n_components
        self.means, self.covariances,self.resp ,self.weights = None, None,None,None
        self.init_weights()

    def init_weights(self,n_components):
        self.w = np.ones((n_components,))/n_components
        if not np.allclose(self.w.sum(), 1):
           print("Weights not normalized")

    
    def init_distributions(self,X, n_clusters):
        # initializing using k-means initialization
        n_samples, n_features = X.shape
        centroids = X[np.random.choice(n_samples,size=n_clusters)]

        self.means = centroids
        
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        assignments = np.zeros_like(distances)
        assignments[np.arange(n_samples),distances.argmin(axis = 1)] = 1


        self.covariances = np.empty(n_clusters, n_features, n_features)

        for i in range(n_clusters):
            self.covariances[i] = np.cov(X[assignments[:,i] == 1].T)
    
    def E_step(self, X):
        self.resp = np.empty

