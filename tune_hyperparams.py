# tune_hyperparams.py

import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Útil para manejar los resultados
import os

# --- Importamos desde nuestros módulos ---
try:
    from tablero import Tablero
    from q_learning_agent import (
        inicializar_q_tables, elegir_acciones, calcular_recompensas, QTableType
    )
    # Importamos la configuración por defecto como base
    import config
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de que 'tablero.py', 'q_learning_agent.py' y 'config.py' estén en el mismo directorio o accesibles.")
    exit()

# --- Función de Entrenamiento Modificada (para devolver métrica clave) ---
# Copiamos la función 'entrenar' de train.py pero la simplificamos
# para que devuelva solo la métrica que usaremos para comparar (e.g., recompensa media final)
# y eliminamos el guardado/carga/logging detallado dentro de esta función.

def run_training_for_tuning(env: Tablero,
                             num_episodios: int,
                             max_steps: int,
                             alpha: float,
                             gamma: float,
                             epsilon_start: float,
                             epsilon_end: float,
                             epsilon_decay: float
                             ) -> float:
    """
    Ejecuta un entrenamiento completo con una configuración dada y devuelve
    una métrica de rendimiento (e.g., recompensa promedio de los últimos N episodios).
    """
    q_tables = inicializar_q_tables(env.num_robots)
    epsilon = epsilon_start
    historico_recompensas_episodio = []

    for episodio in range(num_episodios):
        estado = env.reset()
        recompensa_acumulada_episodio = np.zeros(env.num_robots)
        done = False
        paso = 0

        while not done and paso < max_steps:
            paso += 1
            acciones = elegir_acciones(q_tables, estado, epsilon, env)
            estado_previo = estado
            estado_siguiente, done = env.step(acciones)
            recompensas = calcular_recompensas(estado_previo, estado_siguiente, acciones, done, env)

            for i in range(env.num_robots):
                if estado_previo[2][i]:
                    accion_tomada = acciones[i]
                    recompensa_recibida = recompensas[i]
                    q_antiguo = q_tables[i][estado_previo].get(accion_tomada, 0.0)
                    mejor_q_siguiente = 0.0
                    if not done and estado_siguiente[2][i]:
                        acciones_validas_siguiente = env.get_valid_actions(i)
                        if acciones_validas_siguiente:
                            q_valores_siguientes = q_tables[i][estado_siguiente]
                            q_validos = [q_valores_siguientes.get(a, 0.0) for a in acciones_validas_siguiente]
                            if q_validos:
                                mejor_q_siguiente = max(q_validos)
                    delta = recompensa_recibida + gamma * mejor_q_siguiente - q_antiguo
                    q_tables[i][estado_previo][accion_tomada] = q_antiguo + alpha * delta
                    recompensa_acumulada_episodio[i] += recompensa_recibida
            estado = estado_siguiente

        recompensa_promedio_episodio = np.mean(recompensa_acumulada_episodio)
        historico_recompensas_episodio.append(recompensa_promedio_episodio)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # --- Métrica de Evaluación ---
    # Usaremos la recompensa promedio de los últimos X episodios como métrica
    # Puedes ajustar X o usar otra métrica (e.g., recompensa máxima, paso de convergencia)
    if not historico_recompensas_episodio:
        return -float('inf') # Caso extremo si no hubo episodios
    metric_episodes = min(len(historico_recompensas_episodio), 100) # Usar últimos 100 o menos
    final_performance_metric = np.mean(historico_recompensas_episodio[-metric_episodes:])

    return final_performance_metric


# --- Definición de la Cuadrícula (Grid) de Hiperparámetros ---
param_grid = {
    'alpha': [0.05, 0.1, 0.2],             # Tasa de aprendizaje
    'gamma': [0.9, 0.95],                 # Factor de descuento
    'epsilon_decay': [0.999, 0.9995],     # Decaimiento de Epsilon (más lento = más exploración)
    # Podrías añadir otros como epsilon_end si quieres
    # 'epsilon_end': [0.01, 0.05],
}

# --- Configuración Fija para el Tuning (puede venir de config.py) ---
# Usamos valores de config.py como base, pero los parámetros en param_grid los sobreescribirán
FILAS_TUNE = config.FILAS
COLUMNAS_TUNE = config.COLUMNAS
N_ROBOTS_TUNE = config.N_ROBOTS
POS_INI_TUNE = config.POSICION_INICIAL
NUM_EPISODIOS_TUNE = config.NUM_EPISODIOS // 5 # ¡REDUCIR para tuning! O será muy largo
MAX_STEPS_TUNE = config.MAX_STEPS_PER_EPISODE
EPSILON_START_TUNE = config.EPSILON_START
EPSILON_END_TUNE = config.EPSILON_END # Fijo a menos que esté en param_grid

# --- Ejecución del Grid Search ---
if __name__ == "__main__":
    print("--- Iniciando Grid Search para Hiperparámetros ---")

    # Crear el entorno una vez
    entorno_tune = Tablero(filas=FILAS_TUNE, columnas=COLUMNAS_TUNE,
                           num_robots=N_ROBOTS_TUNE, posicion_inicial=POS_INI_TUNE)

    # Generar todas las combinaciones de parámetros
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = list(itertools.product(*values))
    total_combinations = len(param_combinations)

    print(f"Probando {total_combinations} combinaciones de parámetros.")
    print(f"Parámetros a variar: {list(keys)}")
    print(f"Número de episodios por combinación: {NUM_EPISODIOS_TUNE}")

    results = [] # Lista para guardar (combinación, métrica)
    start_time_grid = time.time()

    # Iterar sobre cada combinación
    for i, combo in enumerate(param_combinations):
        # Crear diccionario con los parámetros de esta combinación
        current_params = dict(zip(keys, combo))
        print(f"\n[{i+1}/{total_combinations}] Probando combinación: {current_params}")
        combo_start_time = time.time()

        # Ejecutar el entrenamiento con esta combinación
        # Asegúrate de pasar todos los parámetros necesarios a la función
        performance = run_training_for_tuning(
            env=entorno_tune,
            num_episodios=NUM_EPISODIOS_TUNE,
            max_steps=MAX_STEPS_TUNE,
            alpha=current_params.get('alpha', config.ALPHA), # Usa valor de combo o default
            gamma=current_params.get('gamma', config.GAMMA),
            epsilon_start=EPSILON_START_TUNE, # Fijo en este ejemplo
            epsilon_end=current_params.get('epsilon_end', EPSILON_END_TUNE),
            epsilon_decay=current_params.get('epsilon_decay', config.EPSILON_DECAY)
        )

        combo_time = time.time() - combo_start_time
        print(f"Resultado (Métrica Rendimiento): {performance:.4f} | Tiempo: {combo_time:.2f}s")
        results.append({'params': current_params, 'performance': performance})

    total_time_grid = time.time() - start_time_grid
    print(f"\n--- Grid Search Finalizado en {total_time_grid:.2f} segundos ---")

    # --- Análisis de Resultados ---
    if not results:
        print("No se obtuvieron resultados.")
    else:
        # Convertir a DataFrame de Pandas para fácil manejo
        results_df = pd.DataFrame(results)
        # Ordenar por rendimiento (descendente, mayor es mejor)
        results_df = results_df.sort_values(by='performance', ascending=False).reset_index(drop=True)

        print("\nMejores Combinaciones Encontradas:")
        print(results_df.head()) # Mostrar las 5 mejores

        # Guardar resultados en un CSV
        results_filename = "grid_search_results.csv"
        try:
            results_df.to_csv(results_filename, index=False)
            print(f"\nResultados completos guardados en '{results_filename}'")
        except Exception as e:
            print(f"\nError al guardar resultados en CSV: {e}")

        # Obtener la mejor combinación
        best_result = results_df.iloc[0]
        best_params = best_result['params']
        best_performance = best_result['performance']

        print("\n---------------------------------------------")
        print("Mejor Combinación:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"Mejor Métrica de Rendimiento: {best_performance:.4f}")
        print("---------------------------------------------")
        print("\n>> ¡Actualiza tu archivo 'config.py' con estos valores si parecen prometedores! <<")
        print("(Recuerda que se entrenó con menos episodios, puede que necesites ajustar NUM_EPISODIOS de nuevo)")