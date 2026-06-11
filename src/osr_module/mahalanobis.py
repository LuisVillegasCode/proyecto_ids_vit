import torch
import logging

class OpenSetShield:
    def __init__(self, n_components=128, lambda_mad=3.0, device='cuda'):
        self.n_components = n_components
        self.lambda_mad = lambda_mad
        self.device = device
        
        # Parámetros estadísticos que se guardarán en disco
        self.pca_mean = None
        self.pca_v = None
        self.centroids = {}
        self.inv_covariances = {}
        self.thresholds = {}

    def _robust_inverse(self, cov_matrix, base_epsilon=1e-5, max_iters=5):
        """
        Bloque de rescate SRE. Evita torch.linalg.inv() nativo.
        Aplica Ridge Regularization iterativa. Falla segura a Pseudo-Inversa.
        """
        identity = torch.eye(cov_matrix.size(0), device=self.device, dtype=torch.float64)
        eps = base_epsilon
        
        for _ in range(max_iters):
            try:
                # Estabilización Algebraica: Sigma + eps * I
                reg_cov = cov_matrix + eps * identity
                # Descomposición de Cholesky (más estable que inv directo)
                L = torch.linalg.cholesky(reg_cov)
                return torch.cholesky_inverse(L)
            except RuntimeError:
                # LinAlgError capturado, multiplicamos ruido x10
                eps *= 10.0
                
        logging.warning("[!] Fallo crítico de Cholesky tras iteraciones. Aplicando Pseudo-Inversa (Moore-Penrose).")
        return torch.linalg.pinv(cov_matrix)

    def fit_pca(self, latents):
        """
        Calcula el PCA determinista. Operaciones 100% en GPU y float64.
        """
        self.pca_mean = latents.mean(dim=0, keepdim=True)
        centered = latents - self.pca_mean
        
        cov = (centered.T @ centered) / (centered.size(0) - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # PyTorch eigh devuelve en orden ascendente, requerimos descendente
        eigenvectors = torch.flip(eigenvectors, dims=[1])
        self.pca_v = eigenvectors[:, :self.n_components]

    def transform_pca(self, latents):
        """Proyección al subespacio latente comprimido."""
        centered = latents - self.pca_mean
        return centered @ self.pca_v

    def fit_profiles(self, train_latents, train_labels, class_indices):
        """
        Ajuste del Escudo OSR. Casting defensivo aplicado.
        """
        train_latents = train_latents.to(dtype=torch.float64, device=self.device)
        self.fit_pca(train_latents)
        proj_latents = self.transform_pca(train_latents)

        for c in class_indices:
            mask = (train_labels == c)
            class_latents = proj_latents[mask]
            
            if len(class_latents) < 2:
                continue
                
            mu = class_latents.mean(dim=0)
            centered = class_latents - mu
            cov = (centered.T @ centered) / (centered.size(0) - 1)
            
            inv_cov = self._robust_inverse(cov)
            self.centroids[c] = mu
            self.inv_covariances[c] = inv_cov

    def calculate_distances(self, latents, labels):
        """
        Cálculo Vectorizado de Distancia de Mahalanobis.
        Fórmula: M_k(x) = (f(x) - mu_k)^T Sigma_k^-1 (f(x) - mu_k)
        """
        proj_latents = self.transform_pca(latents.to(dtype=torch.float64, device=self.device))
        distances = torch.zeros(latents.size(0), dtype=torch.float64, device=self.device)
        
        for c in self.centroids.keys():
            mask = (labels == c)
            if not mask.any(): continue
            
            x_c = proj_latents[mask]
            mu = self.centroids[c]
            inv_cov = self.inv_covariances[c]
            
            diff = x_c - mu
            # Multiplicación matricial eficiente en GPU: (x-mu) @ Sigma^-1
            left_term = torch.matmul(diff, inv_cov)
            # Producto punto fila por fila para aislar cada tensor
            dist_sq = torch.sum(left_term * diff, dim=1)
            
            # Protección contra underflow negativo
            distances[mask] = torch.clamp(dist_sq, min=0.0)
            
        return distances

    def calibrate_thresholds(self, val_latents, val_labels):
        """
        Calibración dinámica basada en la Desviación Absoluta de la Mediana (MAD).
        """
        distances = self.calculate_distances(val_latents, val_labels)
        
        for c in self.centroids.keys():
            mask = (val_labels == c)
            if not mask.any(): continue
            
            D_k = distances[mask]
            median = torch.median(D_k)
            mad = torch.median(torch.abs(D_k - median))
            
            # tau_k = Mediana + lambda * MAD
            tau = median + self.lambda_mad * mad
            self.thresholds[c] = tau
            logging.info(f"[*] Escudo Clase {c} calibrado | MAD: {mad:.4f} | tau: {tau:.4f}")

    def detect_anomalies(self, latents, predicted_labels):
        """
        Intercepción de inferencia. Retorna un tensor booleano y las distancias.
        """
        distances = self.calculate_distances(latents, predicted_labels)
        is_anomaly = torch.zeros(latents.size(0), dtype=torch.bool, device=self.device)
        
        for c in self.centroids.keys():
            mask = (predicted_labels == c)
            if not mask.any(): continue
            
            tau = self.thresholds.get(c, float('inf'))
            is_anomaly[mask] = distances[mask] > tau
            
        return is_anomaly, distances

    def save_profiles(self, path):
        # Mover los perfiles a CPU exclusivamente para serialización segura
        torch.save({
            'pca_mean': self.pca_mean.cpu() if self.pca_mean is not None else None,
            'pca_v': self.pca_v.cpu() if self.pca_v is not None else None,
            'centroids': {k: v.cpu() for k, v in self.centroids.items()},
            'inv_covariances': {k: v.cpu() for k, v in self.inv_covariances.items()},
            'thresholds': self.thresholds # Son escalares, no necesitan .cpu()
        }, path)