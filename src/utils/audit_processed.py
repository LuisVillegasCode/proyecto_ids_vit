import os
import random
import h5py
import numpy as np

def auditar_hdf5(directorio, num_muestras=2):
    print(f"\n{'='*60}")
    print(f"🔍 AUDITANDO DIRECTORIO: {directorio}")
    print(f"{'='*60}")
    
    if not os.path.exists(directorio):
        print(f"❌ El directorio {directorio} no existe.")
        return

    archivos = [f for f in os.listdir(directorio) if f.endswith('.hdf5')]
    if not archivos:
        print("❌ No se encontraron archivos .hdf5.")
        return
    
    print(f"📊 Total de archivos HDF5 encontrados: {len(archivos)}")
    muestras = random.sample(archivos, min(num_muestras, len(archivos)))

    for f_name in muestras:
        f_path = os.path.join(directorio, f_name)
        print(f"\n📄 Archivo Muestra: {f_name}")
        try:
            with h5py.File(f_path, 'r') as hf:
                llaves = list(hf.keys())
                print(f"   Contiene {len(llaves)} elementos (Flujos) en la raíz.")
                
                # Para ser rápidos, evaluamos solo hasta 2 flujos al azar por archivo
                llaves_muestra = random.sample(llaves, min(2, len(llaves)))
                
                for key in llaves_muestra:
                    nodo = hf[key]
                    
                    # 1. Si el elemento es una CARPETA (Group)
                    if isinstance(nodo, h5py.Group):
                        sub_llaves = list(nodo.keys())
                        print(f"   📁 [Grupo: {key[:30]}...] -> Contiene: {sub_llaves}")
                        
                        # Extraemos y evaluamos el primer Dataset dentro de la carpeta
                        if sub_llaves:
                            sub_key = sub_llaves[0]
                            data = nodo[sub_key]
                            auditar_dataset(data, f"{key[:20]}.../{sub_key}")
                            
                    # 2. Si el elemento es un ARCHIVO DIRECTO (Dataset)
                    elif isinstance(nodo, h5py.Dataset):
                        auditar_dataset(nodo, key)
                        
        except Exception as e:
            print(f"   ❌ Error crítico al leer el archivo: {e}")

def auditar_dataset(data, path_nombre):
    """Evalúa la geometría y pureza del tensor real"""
    if not isinstance(data, h5py.Dataset):
        return
    print(f"      📊 [Dataset: {path_nombre}] Shape: {data.shape} | Tipo: {data.dtype}")
    
    # Validaciones anti-colapso para redes neuronales
    if np.issubdtype(data.dtype, np.number):
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        print(f"         ¿Contiene NaNs (Nulos)?: {'Sí ❌ (PELIGRO)' if has_nan else 'No ✅'}")
        print(f"         ¿Contiene Infinitos?:    {'Sí ❌ (PELIGRO)' if has_inf else 'No ✅'}")
        print(f"         Rango de valores:        Min: {np.min(data)} | Max: {np.max(data)}\n")

if __name__ == "__main__":
    auditar_hdf5('data/processed/train_val')
    auditar_hdf5('data/processed/hold_out_test')
    print("\n[🚀] AUDITORÍA FINALIZADA.\n")