# src/utils/config_manager.py
import os
import yaml
import logging
import argparse

class Environment:
    def __init__(self, mode: str, script_name: str):
        """
        Inicializa el entorno recibiendo el modo explícitamente, 
        desacoplando la clase de la lectura de la CLI.
        """
        if mode not in ['pilot', 'prod']:
            raise ValueError("El modo debe ser estrictamente 'pilot' o 'prod'.")
            
        self.mode = mode
        self.script_name = script_name
        
        self._load_config()
        self._setup_logging()

    def _load_config(self):
        config_path = "configs/global_config.yaml"
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            # Mejora 3: Propagación de errores en lugar de sys.exit(1)
            raise RuntimeError(f"[*] FATAL ERROR: Fallo al cargar el archivo de configuración {config_path}.\nDetalle: {e}")

    def inject_pilot_prefix(self, path_str: str) -> str:
        """Aísla las rutas de salida en modo piloto sin duplicar el prefijo."""
        if not path_str or path_str in ('/', '\\'):
            return path_str

        clean_path = path_str.rstrip('/\\')
        head, tail = os.path.split(clean_path)

        if tail == 'pilot' or tail.startswith('pilot_'):
            return path_str

        new_path = os.path.join(head, f"pilot_{tail}")
        if path_str.endswith(('/', '\\')):
            new_path += path_str[-1]

        return new_path

    def get_value(self, *keys):
        """
        Navega de forma segura para obtener valores puros (Fail Fast).
        """
        data = self.config
        for key in keys:
            if not isinstance(data, dict) or key not in data:
                raise KeyError(f"[*] ERROR FATAL: La clave '{key}' no existe en la jerarquía solicitada de global_config.yaml")
            data = data[key]
        return data

    def get_path(self, *keys, ensure_exists=False, is_file=False, apply_pilot=True) -> str:
        """Obtiene una ruta y permite desactivar explícitamente el aislamiento piloto."""
        raw_path = str(self.get_value(*keys))

        if self.mode == 'pilot' and apply_pilot:
            final_path = self.inject_pilot_prefix(raw_path)
        else:
            final_path = raw_path

        if ensure_exists:
            dir_to_create = os.path.dirname(final_path) if is_file else final_path
            if dir_to_create:
                os.makedirs(dir_to_create, exist_ok=True)

        return final_path

    def _setup_logging(self):
        # Aseguramos que la carpeta de logs exista (sabemos que es un directorio, así que is_file=False por defecto)
        log_dir = self.get_path('paths', 'artifacts', 'telemetry_logs', ensure_exists=True)
        log_file = os.path.join(log_dir, f"{self.script_name}_{self.mode}.log")
        
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

# Mejora 1: Inversión de Control. La fábrica recibe los argumentos ya parseados.
def setup_environment(script_name: str, args: argparse.Namespace) -> Environment:
    """
    Fábrica de inicialización. Requiere que el script principal 
    haya parseado sus propios argumentos previamente.
    """
    if not hasattr(args, 'mode'):
        raise AttributeError("El objeto 'args' proporcionado no contiene el atributo obligatorio '--mode'.")
        
    return Environment(mode=args.mode, script_name=script_name)