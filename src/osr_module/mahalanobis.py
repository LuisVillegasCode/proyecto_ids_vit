# src/osr_module/mahalanobis.py
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

    def fit_profiles(self, train_latents, train_labels, class_indices, batch_size=50000):
        """
        Ajuste del Escudo OSR mediante Streaming PCA (Escalabilidad O(1) en VRAM).
        """
        num_samples = train_latents.size(0)
        latent_dim = train_latents.size(1)
        
        # 1. STREAMING MEAN (Calculamos la media global por bloques)
        sum_latents = torch.zeros(latent_dim, dtype=torch.float64, device=self.device)
        for i in range(0, num_samples, batch_size):
            batch = train_latents[i:i+batch_size].to(self.device, dtype=torch.float64)
            sum_latents += batch.sum(dim=0)
        self.pca_mean = (sum_latents / num_samples).unsqueeze(0)
        
        # 2. STREAMING COVARIANCE para PCA (Calculamos X^T * X por bloques)
        cov_sum = torch.zeros((latent_dim, latent_dim), dtype=torch.float64, device=self.device)
        for i in range(0, num_samples, batch_size):
            batch = train_latents[i:i+batch_size].to(self.device, dtype=torch.float64)
            centered_batch = batch - self.pca_mean
            cov_sum += centered_batch.T @ centered_batch
            
        cov = cov_sum / (num_samples - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        self.pca_v = torch.flip(eigenvectors, dims=[1])[:, :self.n_components]

        # Estructuras acumulativas para calcular perfiles de clase en Streaming
        class_counts = {c: 0 for c in class_indices}
        class_sums = {c: torch.zeros(self.n_components, dtype=torch.float64, device=self.device) for c in class_indices}
        class_cov_sums = {c: torch.zeros((self.n_components, self.n_components), dtype=torch.float64, device=self.device) for c in class_indices}

        # 3. STREAMING CENTROIDS (Media por clase)
        for i in range(0, num_samples, batch_size):
            batch = train_latents[i:i+batch_size].to(self.device, dtype=torch.float64)
            batch_labels = train_labels[i:i+batch_size].to(self.device)
            proj_batch = (batch - self.pca_mean) @ self.pca_v
            
            for c in class_indices:
                mask = (batch_labels == c)
                if not mask.any(): continue
                x_c = proj_batch[mask]
                class_sums[c] += x_c.sum(dim=0)
                class_counts[c] += x_c.size(0)

        for c in class_indices:
            if class_counts[c] < 2: continue
            self.centroids[c] = class_sums[c] / class_counts[c]

        # 4. STREAMING MAHALANOBIS COVARIANCE (Covarianza por clase)
        for i in range(0, num_samples, batch_size):
            batch = train_latents[i:i+batch_size].to(self.device, dtype=torch.float64)
            batch_labels = train_labels[i:i+batch_size].to(self.device)
            proj_batch = (batch - self.pca_mean) @ self.pca_v
            
            for c in class_indices:
                mask = (batch_labels == c)
                if not mask.any(): continue
                x_c = proj_batch[mask]
                centered_c = x_c - self.centroids[c]
                class_cov_sums[c] += centered_c.T @ centered_c

        # 5. Inversión final
        for c in class_indices:
            if class_counts[c] < 2: continue
            cov_c = class_cov_sums[c] / (class_counts[c] - 1)
            self.inv_covariances[c] = self._robust_inverse(cov_c)

    def calculate_distances(self, latents, labels, batch_size=50000):
        """
        Cálculo Vectorizado de Distancia de Mahalanobis.
        Fórmula: M_k(x) = (f(x) - mu_k)^T Sigma_k^-1 (f(x) - mu_k)
        """
        num_samples = latents.size(0)
        # Almacenamos el resultado en CPU RAM para no saturar la GPU
        distances = torch.zeros(num_samples, dtype=torch.float64, device='cpu')
        
        for i in range(0, num_samples, batch_size):
            # Subimos solo el lote actual a la GPU
            batch_latents = latents[i:i+batch_size].to(dtype=torch.float64, device=self.device)
            batch_labels = labels[i:i+batch_size].to(device=self.device)
            
            proj_batch = self.transform_pca(batch_latents)
            batch_dists = torch.zeros(batch_latents.size(0), dtype=torch.float64, device=self.device)
            
            for c in self.centroids.keys():
                mask = (batch_labels == c)
                if not mask.any(): continue
                
                x_c = proj_batch[mask]
                mu = self.centroids[c]
                inv_cov = self.inv_covariances[c]
                
                diff = x_c - mu
                left_term = torch.matmul(diff, inv_cov)
                dist_sq = torch.sum(left_term * diff, dim=1)
                
                batch_dists[mask] = torch.clamp(dist_sq, min=0.0)
                
            # Descargamos el resultado del lote a la RAM
            distances[i:i+batch_size] = batch_dists.cpu()
            
            # Solo eliminamos las referencias, NO vaciamos la caché aquí
            del batch_latents, batch_labels, proj_batch, batch_dists
        
        # La limpieza de VRAM se hace UNA VEZ, fuera del bucle    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
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
        # Garantizamos explícitamente que las etiquetas vivan en la RAM (CPU)
        predicted_labels = predicted_labels.cpu()
        
        distances = self.calculate_distances(latents, predicted_labels)
        
        # Sincronización de dispositivos: is_anomaly vive en la RAM
        is_anomaly = torch.zeros(latents.size(0), dtype=torch.bool, device='cpu')
        
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