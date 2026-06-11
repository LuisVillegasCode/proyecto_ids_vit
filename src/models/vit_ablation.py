import os
import json
import yaml
import h5py
import torch
import logging
import argparse
import shutil
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef
from einops.layers.torch import Rearrange
from tqdm import tqdm
import torch
import torch.nn.functional as F

# ==============================================================================
# 0. INYECCIÓN DE ENTORNO Y CONFIGURACIÓN GLOBAL
# ==============================================================================
def inject_pilot_prefix(path_str: str) -> str:
    if not path_str or path_str in ('/', '\\'): return path_str
    clean_path = path_str.rstrip('/\\')
    head, tail = os.path.split(clean_path)
    if tail.startswith('pilot_'): return path_str
    new_path = os.path.join(head, f"pilot_{tail}")
    if path_str.endswith(('/', '\\')): new_path += path_str[-1]
    return new_path

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['pilot', 'prod'], required=True)
args, _ = parser.parse_known_args()

with open("configs/global_config.yaml", 'r') as f:
    GLOBAL_CONFIG = yaml.safe_load(f)

TELEMETRY_LOGS = GLOBAL_CONFIG['paths']['artifacts']['telemetry_logs']
if args.mode == 'pilot':
    TELEMETRY_LOGS = inject_pilot_prefix(TELEMETRY_LOGS)

os.makedirs(TELEMETRY_LOGS, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(TELEMETRY_LOGS, f"phase3_ablation_{args.mode}.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)

# ==============================================================================
# 1. EL PUENTE I/O: DATALOADER OPTIMIZADO (FR4.1 + NFR2)
# ==============================================================================
class IDS2018Dataset(Dataset):
    def __init__(self, data_dir, scaler_json, n_min, max_bytes=128, mode='prod'):
        self.data_dir = data_dir
        self.n_min = n_min
        self.max_bytes = max_bytes
        self.mode = mode
        
        # Caché local por worker para evitar el cuello de botella I/O (NFR2)
        self.worker_file_cache = {}
        
        # Carga de límites globales en RAM (FR4.1 - Anti-Fuga de Datos)
        with open(scaler_json, 'r') as f:
            bounds = json.load(f)
            self.min_e = bounds['entropy_channel']['min']
            self.max_e = bounds['entropy_channel']['max']
            self.min_r = bounds['raw_bytes_channel']['min']
            self.max_r = bounds['raw_bytes_channel']['max']
            
        self.class_to_idx = {"Benign": 0, "BruteForce": 1, "DoS": 2, "DDoS": 3, "Brute_Force_Web": 4, "Brute_Force_XSS": 5, "SQL_Injection": 6}
        self.index, self.class_counts = self._build_or_load_index()
        
    def _build_or_load_index(self):
        index_file = os.path.join(self.data_dir, f"dataset_index_{self.mode}.pt")
        if os.path.exists(index_file):
            logging.info(f"[*] Cargando índice cacheado desde {index_file}")
            return torch.load(index_file)
            
        logging.info("[*] Construyendo índice maestro HDF5... (Solo la primera vez)")
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.hdf5')]
        if self.mode == 'pilot': files = files[:2]
            
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
                
        if self.mode == 'pilot': index = index[:1000]
            
        torch.save((index, class_counts), index_file)
        return index, class_counts

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fname, flow_id, label = self.index[idx]
        path = os.path.join(self.data_dir, fname)
        
        try:
            if path not in self.worker_file_cache:
                if len(self.worker_file_cache) > 20:
                    oldest_path = list(self.worker_file_cache.keys())[0]
                    self.worker_file_cache[oldest_path].close()
                    del self.worker_file_cache[oldest_path]
                self.worker_file_cache[path] = h5py.File(path, 'r', swmr=True)
                
            hf = self.worker_file_cache[path]
            grp = hf[flow_id]
            
            # 1. Extracción directa del tensor consolidado (18, 128, 3)
            tensor_np = grp['rgb_e_tensor'][:]
            
            # 2. Truncamiento temporal adaptativo para el estudio de ablación (eje de paquetes)
            tensor_np = tensor_np[:self.n_min, :self.max_bytes, :]
            if tensor_np.ndim != 3 or tensor_np.shape[-1] != 3:
                raise ValueError(
                    f"Tensor inválido en {fname}/{flow_id}: {tensor_np.shape}"
                )
            # 3. Transposición vectorizada de HWC (N_min, Max_Bytes, 3) a CHW (3, N_min, Max_Bytes) exigido por PyTorch
            img = np.transpose(tensor_np, (2, 0, 1))
            
            # 4. Estandarización Min-Max Global al vuelo optimizada por canales
            if self.max_r > self.min_r:
                img[0] = (img[0] - self.min_r) / (self.max_r - self.min_r)
                img[1] = (img[1] - self.min_r) / (self.max_r - self.min_r)
            if self.max_e > self.min_e:
                img[2] = (img[2] - self.min_e) / (self.max_e - self.min_e)
                
            return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            logging.error(f"[!] HDF5 Read Error en {fname}, flow {flow_id}: {str(e)}")
            return None

# ==============================================================================
# 2. ARQUITECTURA: VISION TRANSFORMER OSR Y XAI (FR6, FR13)
# ==============================================================================

class TransparentTransformerBlock(nn.Module):
    """
    Reemplazo de la "caja negra" nn.TransformerEncoderLayer.
    Exigido por FR13 para retornar explícitamente los mapas de atención (XAI).
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        # need_weights=True permite la Inteligencia Artificial Explicable (XAI)
        attn_out, attn_weights = self.attn(norm_x, norm_x, norm_x, need_weights=True)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights

class ViT_OSR(nn.Module):
    def __init__(self, n_min, max_bytes=128, patch_size=(1, 16), embed_dim=768, depth=12, num_heads=12, num_classes=7):
        super().__init__()
        self.patch_h, self.patch_w = patch_size
        num_patches = (n_min // self.patch_h) * (max_bytes // self.patch_w)
        
        # Actualizado a 3 canales (RGB-E)
        patch_dim = 3 * self.patch_h * self.patch_w
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_h, p2=self.patch_w),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Bloques transparentes (XAI)
        self.layers = nn.ModuleList([
            TransparentTransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            all_attn_weights.append(attn_weights)
            
        cls_output = x[:, 0]  # Vector latente para Distancia de Mahalanobis (Fase 4)
        logits = self.mlp_head(cls_output)
        
        # Se retorna Logits (Softmax Implícito), el Token CLS y la Atención XAI
        return logits, cls_output, all_attn_weights 
    
class FocalLoss(nn.Module):
    """
    Penaliza a los ejemplos 'fáciles' (Benignos) y concentra los gradientes 
    en los ejemplos 'difíciles' (Ataques) usando el factor gamma.
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        # Validación temprana del parámetro de reducción
        assert reduction in ['mean', 'sum', 'none'], "Reduction debe ser 'mean', 'sum' o 'none'"
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. CE puro (SIN pesos) para calcular la probabilidad exacta (pt)
        ce_loss_unweighted = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss_unweighted)
        
        # 2. CE original (CON pesos) para mantener el balanceo base de clases
        ce_loss_weighted = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        # 3. Fórmula final Focal Loss: Modulador * Loss Pesado
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss_weighted
        
        # 4. Manejo correcto de la reducción
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # reduction == 'none'
            return focal_loss
    
# ==============================================================================
# 3. ORQUESTADOR: ESTUDIO DE ABLACIÓN Y ENTRENAMIENTO (FR7, NFR5, FR11)
# ==============================================================================
def train_ablation_study(mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"[*] Acelerador detectado: {device}")
    
    # Estudio de ablación según Metodología 4.2
    n_min_candidates = [9]
    
    train_conf = GLOBAL_CONFIG['training']
    vit_conf = GLOBAL_CONFIG['vit_model']
    
    epochs_per_ablation = 5 if mode == 'pilot' else train_conf['epochs']
    batch_size = 32 if mode == 'pilot' else train_conf['batch_size']
    learning_rate = train_conf['learning_rate']
    ckpt_freq = train_conf.get('checkpoint_frequency', 5)
    
    scaler_json = GLOBAL_CONFIG['paths']['configs']['scaler_bounds']
    train_dir = GLOBAL_CONFIG['paths']['output']['train_val']
    ckpt_dir = GLOBAL_CONFIG['paths']['artifacts']['checkpoints']
    
    if mode == 'pilot':
        scaler_json = inject_pilot_prefix(scaler_json)
        train_dir = inject_pilot_prefix(train_dir)
        ckpt_dir = inject_pilot_prefix(ckpt_dir)
        
    os.makedirs(ckpt_dir, exist_ok=True)
    
    for n_min in n_min_candidates:
        logging.info(f"\n{'='*60}\n[*] INICIANDO ABLACIÓN PARA N_min = {n_min}\n{'='*60}")
        
        dataset = IDS2018Dataset(train_dir, scaler_json, n_min, mode=mode)
        total_samples = sum(dataset.class_counts.values())
        
        # FR7: Manejo de desbalanceo sin remuestreo sintético (Class Weights)
        class_weights = torch.tensor([total_samples / (len(dataset.class_counts) * (count + 1e-6)) 
                                      for count in dataset.class_counts.values()], dtype=torch.float32).to(device)
        
        config_workers = GLOBAL_CONFIG['preprocessing']['multiprocessing_workers']
        optimal_workers = 0 if mode == 'pilot' else config_workers
        optimal_prefetch = 2 if optimal_workers > 0 else None
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=optimal_workers, pin_memory=True, 
                                prefetch_factor=optimal_prefetch, collate_fn=safe_collate)
        
        model = ViT_OSR(
            n_min=n_min,
            patch_size=(1, vit_conf['patch_size']), 
            embed_dim=vit_conf['embed_dim'],
            depth=vit_conf['depth'],
            num_heads=vit_conf['num_heads']
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"[*] ViT Arquitectura: {vit_conf['embed_dim']}d, {vit_conf['depth']} layers, {vit_conf['num_heads']} heads")
        logging.info(f"[*] Parámetros Entrenables: {total_params:,}")
        
        criterion = FocalLoss(weight=class_weights, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scaler = torch.cuda.amp.GradScaler()
        
        start_epoch = 0
        ckpt_path = os.path.join(ckpt_dir, f"vit_nmin_{n_min}_checkpoint.pt")
        
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scaler.load_state_dict(checkpoint['scaler_state'])
            torch.set_rng_state(checkpoint['rng_state'])  # Restaurar semilla atómica (FR11)
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"[*] Rescatando entrenamiento desde la Época {start_epoch}")
            
        if start_epoch >= epochs_per_ablation:
            logging.info(f"[*] Ablación N_min={n_min} ya completada. Saltando.")
            continue
            
        for epoch in range(start_epoch, epochs_per_ablation):
            model.train()
            running_loss = 0.0
            all_preds, all_labels = [], []
            
            pbar = tqdm(dataloader, desc=f"Epoca {epoch+1}/{epochs_per_ablation}")
            for step, (inputs, labels) in enumerate(pbar):
                if len(inputs) == 0: continue
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                
                # NFR5: Eficiencia de VRAM con Precisión Mixta Automática (AMP)
                with torch.cuda.amp.autocast():
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
            
            actual_steps = len(dataloader)  
            epoch_loss = running_loss / actual_steps if actual_steps > 0 else 0.0
            mcc = matthews_corrcoef(all_labels, all_preds) if len(all_labels) > 0 else 0.0
            logging.info(f"[N_min={n_min}] Época {epoch+1} | Loss: {epoch_loss:.4f} | MCC Train: {mcc:.4f}")
            
            # FR11: Resiliencia de apagones (Escritura Atómica)
            tmp_ckpt = ckpt_path + ".tmp"
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict(),
                'rng_state': torch.get_rng_state(), 
                'mcc': mcc
            }, tmp_ckpt)
            os.rename(tmp_ckpt, ckpt_path)
            
            if (epoch + 1) % ckpt_freq == 0:
                hist_path = os.path.join(ckpt_dir, f"vit_nmin_{n_min}_epoch_{epoch+1}.pt")
                shutil.copyfile(ckpt_path, hist_path)
                logging.info(f"  -> Checkpoint histórico guardado: {hist_path}")

if __name__ == "__main__":
    train_ablation_study(args.mode)