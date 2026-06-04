import os
import json
import yaml
import h5py
import torch
import logging
import argparse
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import matthews_corrcoef
from einops.layers.torch import Rearrange
from tqdm import tqdm

# ==============================================================================
# CONFIGURACIÓN Y TELEMETRÍA
# ==============================================================================
with open("configs/global_config.yaml", 'r') as f:
    GLOBAL_CONFIG = yaml.safe_load(f)

# Configurar logs forenses
os.makedirs(GLOBAL_CONFIG['paths']['artifacts']['telemetry_logs'], exist_ok=True)
logging.basicConfig(
    filename=os.path.join(GLOBAL_CONFIG['paths']['artifacts']['telemetry_logs'], "phase3_ablation.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# ==============================================================================
# 1. EL PUENTE I/O: DATALOADER DEFENSIVO CON ESCALADO AL VUELO (FR4.1)
# ==============================================================================

def safe_collate(batch):
    # Filtrar los elementos que devolvieron None
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor() # Lote vacío de seguridad
    return torch.utils.data.dataloader.default_collate(batch)
    
class IDS2018Dataset(Dataset):
    def __init__(self, data_dir, scaler_json, n_min, max_bytes=128, mode='prod'):
        self.data_dir = data_dir
        self.n_min = n_min
        self.max_bytes = max_bytes
        self.mode = mode
        
        # Cargar fronteras matemáticas para Anti-Fuga de Datos (FR4.1)
        with open(scaler_json, 'r') as f:
            bounds = json.load(f)
            self.min_e = bounds['entropy_channel']['min']
            self.max_e = bounds['entropy_channel']['max']
            self.min_r = bounds['raw_bytes_channel']['min']
            self.max_r = bounds['raw_bytes_channel']['max']
            
        # Mapeo de Clases dinámico
        self.class_to_idx = {"Benign": 0, "BruteForce": 1, "DoS": 2, "DDoS": 3, "Brute_Force_Web": 4, "Brute_Force_XSS": 5, "SQL_Injection": 6}
        self.index, self.class_counts = self._build_or_load_index()
        
    def _build_or_load_index(self):
        index_file = os.path.join(self.data_dir, f"dataset_index_{self.mode}.pt")
        if os.path.exists(index_file):
            logging.info(f"[*] Cargando índice cacheado desde {index_file}")
            return torch.load(index_file)
            
        logging.info("[*] Construyendo índice maestro HDF5... (Esto tomará unos minutos la primera vez)")
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.hdf5')]
        if self.mode == 'pilot': files = files[:2] # Piloto restringe a 2 archivos
            
        index = []
        class_counts = {v: 0 for v in self.class_to_idx.values()}
        
        for fname in tqdm(files, desc="Indexando"):
            path = os.path.join(self.data_dir, fname)
            try:
                with h5py.File(path, 'r', swmr=True) as hf:
                    for flow_id in hf.keys():
                        lbl_str = hf[flow_id].attrs.get('label', 'Benign')
                        if lbl_str in self.class_to_idx:
                            c_idx = self.class_to_idx[lbl_str]
                            index.append((fname, flow_id, c_idx))
                            class_counts[c_idx] += 1
            except Exception as e:
                logging.error(f"[!] Error indexando {fname}: {str(e)}")
                
        # Piloto estricto: Submuestreo
        if self.mode == 'pilot': index = index[:1000]
            
        torch.save((index, class_counts), index_file)
        return index, class_counts

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fname, flow_id, label = self.index[idx]
        path = os.path.join(self.data_dir, fname)
        
        try:
            with h5py.File(path, 'r', swmr=True) as hf:
                grp = hf[flow_id]
                raw_pkts = grp['raw_packets'][:]
                entropies = grp['blue_channel_entropy'][:]
                
                # Truncamiento Dinámico (Estudio de Ablación)
                current_len = min(len(raw_pkts), self.n_min)
                
                # Ensamblaje de Tensores (Canal 0: Bytes, Canal 1: Entropía)
                img = np.zeros((2, self.n_min, self.max_bytes), dtype=np.float32)
                
                for i in range(current_len):
                    # Padding de bytes
                    pkt_bytes = raw_pkts[i][:self.max_bytes]
                    img[0, i, :len(pkt_bytes)] = pkt_bytes
                    img[1, i, :] = entropies[i] # Broadcast de la entropía a lo largo del paquete
                    
                # FR4.1: Normalización Min-Max Segura
                if self.max_r > self.min_r:
                    img[0] = (img[0] - self.min_r) / (self.max_r - self.min_r)
                if self.max_e > self.min_e:
                    img[1] = (img[1] - self.min_e) / (self.max_e - self.min_e)
                    
                return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            # Tolerancia a fallos: retornar matriz vacía benigna si el flujo colapsa
            logging.error(f"[!] HDF5 Read Error en {fname}, flow {flow_id}: {str(e)}")
            return None

# ==============================================================================
# 2. ARQUITECTURA: VISION TRANSFORMER OSR (FR6)
# ==============================================================================
class ViT_OSR(nn.Module):
    def __init__(self, n_min, max_bytes=128, patch_size=(1, 16), embed_dim=128, depth=4, num_heads=8, num_classes=7):
        super().__init__()
        self.patch_h, self.patch_w = patch_size
        num_patches = (n_min // self.patch_h) * (max_bytes // self.patch_w)
        patch_dim = 2 * self.patch_h * self.patch_w # 2 canales (RGB-E)
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_h, p2=self.patch_w),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, img):
        # Generar embeddings y token CLS
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        
        # Forward pass y extracción de pesos de atención (Simulando compatibilidad FR6)
        # Nota: PyTorch TransformerEncoder nativo no devuelve atención directamente. 
        # Extraemos el token CLS post-encoder que servirá para Mahalanobis.
        x = self.transformer(x)
        
        cls_output = x[:, 0] # Representación latente profunda para OSR
        logits = self.mlp_head(cls_output)
        
        # Devolvemos (Logits, CLS Token, Attention Weights Placeholder)
        return logits, cls_output, None 

# ==============================================================================
# 3. ORQUESTADOR: ESTUDIO DE ABLACIÓN Y ENTRENAMIENTO (FR7, NFR5, FR11)
# ==============================================================================
def train_ablation_study(mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"[*] Acelerador detectado: {device}")
    
    n_min_candidates = [3, 6, 9, 12, 15, 18]
    epochs_per_ablation = 5 if mode == 'pilot' else GLOBAL_CONFIG['training']['epochs']
    batch_size = 32 if mode == 'pilot' else GLOBAL_CONFIG['training']['batch_size']
    
    scaler_json = GLOBAL_CONFIG['paths']['configs']['scaler_bounds']
    train_dir = GLOBAL_CONFIG['paths']['output']['train_val']
    ckpt_dir = GLOBAL_CONFIG['paths']['artifacts']['checkpoints']
    os.makedirs(ckpt_dir, exist_ok=True)
    
    for n_min in n_min_candidates:
        logging.info(f"\n{'='*50}\n[*] INICIANDO ABLACIÓN PARA N_min = {n_min}\n{'='*50}")
        
        dataset = IDS2018Dataset(train_dir, scaler_json, n_min, mode=mode)
        # FR7: Mitigación de Desbalanceo Matemático (Class Weights)
        total_samples = sum(dataset.class_counts.values())
        class_weights = torch.tensor([total_samples / (len(dataset.class_counts) * (count + 1e-6)) for count in dataset.class_counts.values()], dtype=torch.float32).to(device)
        
        # AUTO-TUNING DE HARDWARE (Detección de Cores Dinámica)
        cpu_count = os.cpu_count() or 2
        if mode == 'pilot':
            optimal_workers = 0 if cpu_count <= 2 else 2
        else:
            optimal_workers = max(1, cpu_count - 1)
            
        optimal_prefetch = 2 if optimal_workers > 0 else None
        
        logging.info(f"[*] Auto-Tuning: {optimal_workers} workers | Prefetch: {optimal_prefetch}")
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=optimal_workers, pin_memory=True, prefetch_factor=optimal_prefetch, collate_fn=safe_collate)
        
        model = ViT_OSR(n_min=n_min).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=GLOBAL_CONFIG['training']['learning_rate'], weight_decay=0.01)
        scaler = torch.amp.GradScaler('cuda') # NFR5: Precisión Mixta
        
        # FR11: Carga de Checkpoints (Resiliencia)
        start_epoch = 0
        ckpt_path = os.path.join(ckpt_dir, f"vit_nmin_{n_min}_checkpoint.pt")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scaler.load_state_dict(checkpoint['scaler_state'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"[*] Rescatando entrenamiento desde la Época {start_epoch}")
            
        if start_epoch >= epochs_per_ablation:
            logging.info(f"[*] Ablación N_min={n_min} ya completada. Saltando.")
            continue
            
        # Bucle de Entrenamiento
        for epoch in range(start_epoch, epochs_per_ablation):
            model.train()
            running_loss = 0.0
            all_preds, all_labels = [], []
            
            pbar = tqdm(dataloader, desc=f"Epoca {epoch+1}/{epochs_per_ablation}")
            for inputs, labels in pbar:
                if len(inputs) == 0: continue # Saltar si el lote falló por completo
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                
                # NFR5: AMP
                with torch.amp.autocast('cuda'):
                    logits, _, _ = model(inputs)
                    loss = criterion(logits, labels)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix(loss=loss.item())
                
            # Métricas
            epoch_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
            mcc = matthews_corrcoef(all_labels, all_preds) if len(all_labels) > 0 else 0.0
            logging.info(f"[N_min={n_min}] Época {epoch+1} | Loss: {epoch_loss:.4f} | MCC Train: {mcc:.4f}")
            
            # FR11: Guardado Atómico Atómico
            tmp_ckpt = ckpt_path + ".tmp"
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict(),
                'mcc': mcc
            }, tmp_ckpt)
            os.rename(tmp_ckpt, ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['pilot', 'prod'], required=True)
    args = parser.parse_args()
    train_ablation_study(args.mode)