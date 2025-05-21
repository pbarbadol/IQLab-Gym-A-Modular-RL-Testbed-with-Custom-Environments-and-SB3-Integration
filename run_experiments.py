# run_experiments.py
import os
import json
import time
import random
import numpy as np # Solo si necesitas np aquí, sino en train.py es suficiente

# --- IMPORTANTE: Importar la función del script refactorizado ---
try:
    from train_qlearning import run_training_session
except ImportError as e:
    print(f"Error importando 'run_training_session' desde 'train.py': {e}")
    print("Asegúrate de que 'train.py' esté en el mismo directorio o en el PYTHONPATH.")
    exit()

# --- CONFIGURACIÓN GENERAL DE LA BATERÍA DE PRUEBAS ---
BATTERY_BASE_RESULTS_DIR = "experiment_battery_main" # Nombre para la carpeta raíz de esta batería
NUM_TRIALS_PER_CONFIG = 3  # Número de ejecuciones con diferentes semillas por configuración

# Parámetros base (tus "mejores" encontrados o los de config.py)
# Deben coincidir con las claves que espera `run_training_session`
BASE_EXPERIMENT_PARAMS = {
    "FILAS": 8, "COLUMNAS": 8, "N_ROBOTS": 3,
    "POSICION_INICIAL": None, # o [(0,0), (7,7), (0,7)] etc.
    "ALPHA": 0.05532, "GAMMA": 0.95255,
    "EPSILON_START": 1.0, "EPSILON_END": 0.01, "EPSILON_DECAY": 0.99904,
    "NUM_EPISODIOS": 50000, # Considera reducir para la batería si es muy largo (50000)
    "MAX_STEPS_PER_EPISODE": 64,
    "EPISODIOS_PARA_LOG": 250, # Puede ser mayor para la batería, menos verboso
    "GUARDAR_QTABLES_FILENAME": "q_tables_final.pkl", # Solo el nombre del archivo
    "CARGAR_QTABLES_FILENAME": None, # Normalmente no se carga en la batería
    "VENTANA_PROMEDIO_MOVIL": 100,
    # Controles para la batería (usualmente False para acelerar):
    "GENERAR_GRAFICO_INDIVIDUAL": False, # Los gráficos agregados se harán después
    "EVALUAR_SESION_FINAL": False,       # La evaluación agregada se puede hacer después
    "RENDER_EVALUACION": False,
    "PAUSA_RENDER_EVAL": 0.0,
    "MAX_STEPS_EVALUACION": 128 # O MAX_STEPS_PER_EPISODE * 2
}

# --- DEFINICIÓN DE LOS EXPERIMENTOS ---
# Cada elemento es un diccionario con:
#   'experiment_group_name': Nombre de la carpeta para este grupo de trials.
#   'params_override': Diccionario con parámetros que varían del BASE_EXPERIMENT_PARAMS.
experiments_to_run = []

# 1. Baseline (usando parámetros optimizados)
experiments_to_run.append({
    "experiment_group_name": "Baseline_Optimized_R3_8x8",
    "params_override": {} # Sin cambios
})

# 2. Escalabilidad: Número de Robots
for n_r in [1, 2, 4]: # Probar con N_ROBOTS = 3 ya está en Baseline
    if n_r == BASE_EXPERIMENT_PARAMS["N_ROBOTS"]: continue # Evitar duplicar baseline
    experiments_to_run.append({
        "experiment_group_name": f"Scalability_Robots_{n_r}",
        "params_override": {"N_ROBOTS": n_r, "POSICION_INICIAL": None} # Resetear pos_inicial si N_ROBOTS cambia
    })

# 3. Escalabilidad: Tamaño del Grid
for grid_sz in [6, 10]: # Probar con 8x8 ya está en Baseline
    if grid_sz == BASE_EXPERIMENT_PARAMS["FILAS"]: continue
    experiments_to_run.append({
        "experiment_group_name": f"Scalability_Grid_{grid_sz}x{grid_sz}",
        "params_override": {
            "FILAS": grid_sz, "COLUMNAS": grid_sz,
            "MAX_STEPS_PER_EPISODE": grid_sz * grid_sz # Ajustar max_steps
        }
    })

# 4. Impacto de Posiciones Iniciales (para N_ROBOTS = 3)
if BASE_EXPERIMENT_PARAMS["N_ROBOTS"] == 3:
    pos_configs = {
        "all_0_0": None, # Ya cubierto por baseline si POSICION_INICIAL es None
        "corners_3R": [(0, 0), (BASE_EXPERIMENT_PARAMS["FILAS"] - 1, BASE_EXPERIMENT_PARAMS["COLUMNAS"] - 1), (0, BASE_EXPERIMENT_PARAMS["COLUMNAS"] - 1)],
        "center_3R": [(3,3), (3,4), (4,3)] # Asumiendo 8x8, ajustar si el grid base cambia
    }
    for name, pos_ini_val in pos_configs.items():
        if name == "all_0_0" and BASE_EXPERIMENT_PARAMS["POSICION_INICIAL"] is None: continue # Evitar duplicar
        experiments_to_run.append({
            "experiment_group_name": f"InitialPos_{name}",
            "params_override": {"POSICION_INICIAL": pos_ini_val}
        })
else:
    print("Advertencia: Las pruebas de POSICION_INICIAL están configuradas para 3 robots.")


# --- BUCLE PRINCIPAL DE LA BATERÍA DE PRUEBAS ---
if __name__ == "__main__":
    # Crear directorio raíz para esta batería de experimentos si no existe
    os.makedirs(BATTERY_BASE_RESULTS_DIR, exist_ok=True)
    
    # Generador de semillas para asegurar que cada trial tenga una semilla diferente
    # pero que la secuencia de semillas sea la misma si se vuelve a ejecutar la batería.
    master_seed_rng = random.Random(12345) # Semilla para el generador de semillas

    total_experiment_groups = len(experiments_to_run)
    print(f"--- INICIANDO BATERÍA DE PRUEBAS ---")
    print(f"Se ejecutarán {total_experiment_groups} grupos de experimentos.")
    print(f"Cada grupo se ejecutará {NUM_TRIALS_PER_CONFIG} veces (trials) con diferentes semillas.")
    print(f"Resultados guardados en: {BATTERY_BASE_RESULTS_DIR}")

    for i_group, exp_def in enumerate(experiments_to_run):
        group_name = exp_def["experiment_group_name"]
        params_override = exp_def["params_override"]

        # Crear una copia de los parámetros base y aplicar overrides
        current_group_config = BASE_EXPERIMENT_PARAMS.copy()
        current_group_config.update(params_override)

        # Directorio para este grupo de experimentos
        # (e.g., experiment_battery_main/Baseline_Optimized_R3_8x8)
        group_output_folder = os.path.join(BATTERY_BASE_RESULTS_DIR, group_name)
        os.makedirs(group_output_folder, exist_ok=True)

        # Guardar la configuración consolidada para este grupo
        group_config_summary_path = os.path.join(group_output_folder, "experiment_group_config_summary.json")
        try:
            with open(group_config_summary_path, 'w') as f:
                # Crear una copia para serializar sin funciones o tipos no serializables
                serializable_config = current_group_config.copy()
                # POSICION_INICIAL puede ser una lista de tuplas, que es serializable.
                json.dump(serializable_config, f, indent=4)
        except Exception as e:
            print(f"Error guardando config del grupo {group_name}: {e}")

        print(f"\n[{i_group+1}/{total_experiment_groups}] Iniciando Grupo de Experimento: {group_name}")

        for i_trial in range(NUM_TRIALS_PER_CONFIG):
            trial_seed = master_seed_rng.randint(0, 2**31 - 1) # Generar semilla para este trial
            
            # Nombre para la subcarpeta de este trial específico
            # (e.g., trial_0_seed_12345)
            trial_experiment_name = f"trial_{i_trial}_seed_{trial_seed}"
            
            # Preparar el diccionario de parámetros para este trial
            trial_params = current_group_config.copy()
            trial_params["RANDOM_SEED"] = trial_seed
            trial_params["EXPERIMENT_NAME"] = trial_experiment_name
            trial_params["RESULTS_FOLDERNAME"] = group_output_folder # La carpeta donde se creará la subcarpeta del trial

            print(f"   Iniciando Trial {i_trial+1}/{NUM_TRIALS_PER_CONFIG} (Seed: {trial_seed}) para el grupo '{group_name}'...")
            trial_start_time = time.time()
            
            try:
                # --- LLAMADA A LA FUNCIÓN DE ENTRENAMIENTO ---
                run_training_session(trial_params)
                trial_duration = time.time() - trial_start_time
                print(f"  Trial {i_trial+1} completado en {trial_duration:.2f}s.")
            except Exception as e:
                trial_duration = time.time() - trial_start_time
                print(f"¡ERROR! El Trial {i_trial+1} (Seed: {trial_seed}) del grupo '{group_name}' falló después de {trial_duration:.2f}s.")
                print(f"  Error: {e}")
                # Aquí podrías añadir logging más detallado del error a un archivo.
            
            time.sleep(0.5) # Pequeña pausa opcional entre trials

    print("\n--- BATERÍA DE PRUEBAS COMPLETADA ---")