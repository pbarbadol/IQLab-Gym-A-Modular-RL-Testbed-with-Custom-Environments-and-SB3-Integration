# tune_random_search.py

import time
import random # Necesario para Random Search
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

# --- Importar desde nuestros módulos ---
try:
    from tablero import Tablero
    from q_learning_agent import (
        inicializar_q_tables, elegir_acciones, calcular_recompensas, QTableType
    )
    import config
except ImportError as e:
    print(f"Error importando módulos: {e}")
    exit()

# --- Función de Entrenamiento (sin cambios respecto a la versión de Grid Search) ---
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
    (Esta función es idéntica a la usada en Grid Search)
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
                            if q_validos: mejor_q_siguiente = max(q_validos)
                    delta = recompensa_recibida + gamma * mejor_q_siguiente - q_antiguo
                    q_tables[i][estado_previo][accion_tomada] = q_antiguo + alpha * delta
                    recompensa_acumulada_episodio[i] += recompensa_recibida
            estado = estado_siguiente
        recompensa_promedio_episodio = np.mean(recompensa_acumulada_episodio)
        historico_recompensas_episodio.append(recompensa_promedio_episodio)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    if not historico_recompensas_episodio: return -float('inf')
    metric_episodes = min(len(historico_recompensas_episodio), 100)
    final_performance_metric = np.mean(historico_recompensas_episodio[-metric_episodes:])
    return final_performance_metric


# --- Definición de Distribuciones/Rangos para Random Search ---
# Usa funciones lambda para definir cómo muestrear cada parámetro.
param_distributions = {
    # 'alpha': lambda: random.uniform(0.01, 0.3),       # Tasa de aprendizaje (Uniforme)
    'alpha': lambda: 10**random.uniform(-2.0, -0.7), # Tasa de aprendizaje (LogUniforme ~0.01 a 0.2)
    'gamma': lambda: random.uniform(0.85, 0.99),       # Factor de descuento (Uniforme)
    'epsilon_decay': lambda: random.uniform(0.999, 0.9999), # Decaimiento Epsilon (Uniforme)
    # Podrías añadir más aquí si quieres variar otros:
    # 'epsilon_end': lambda: random.uniform(0.005, 0.05),
}

# --- Configuración Fija y Número de Iteraciones ---
# Usamos valores de config.py como base, pero los parámetros muestreados los sobreescribirán
FILAS_TUNE = config.FILAS
COLUMNAS_TUNE = config.COLUMNAS
N_ROBOTS_TUNE = config.N_ROBOTS
POS_INI_TUNE = config.POSICION_INICIAL
# ¡REDUCIR EPISODIOS/PASOS PARA TUNING!
NUM_EPISODIOS_TUNE = config.NUM_EPISODIOS // 10 # Reducir drásticamente para tuning rápido
MAX_STEPS_TUNE = config.MAX_STEPS_PER_EPISODE
EPSILON_START_TUNE = config.EPSILON_START
EPSILON_END_TUNE = config.EPSILON_END # Fijo a menos que esté en param_distributions

# Número de combinaciones aleatorias a probar
N_ITER_RANDOM_SEARCH = 50 # Ajusta este valor según tu presupuesto computacional

# --- Ejecución del Random Search ---
if __name__ == "__main__":
    print("--- Iniciando Random Search para Hiperparámetros ---")

    # Crear el entorno una vez
    entorno_tune = Tablero(filas=FILAS_TUNE, columnas=COLUMNAS_TUNE,
                           num_robots=N_ROBOTS_TUNE, posicion_inicial=POS_INI_TUNE)

    print(f"Probando {N_ITER_RANDOM_SEARCH} combinaciones aleatorias.")
    print(f"Parámetros a variar aleatoriamente: {list(param_distributions.keys())}")
    print(f"Número de episodios por combinación: {NUM_EPISODIOS_TUNE}")

    results = [] # Lista para guardar (combinación, métrica)
    start_time_random = time.time()

    # Iterar N_ITER_RANDOM_SEARCH veces
    for i in range(N_ITER_RANDOM_SEARCH):
        # --- Muestrear una combinación de parámetros ---
        current_params = {key: sampler() for key, sampler in param_distributions.items()}

        # Formatear para impresión legible
        params_str = {k: f'{v:.5f}' if isinstance(v, float) else v for k, v in current_params.items()}
        print(f"\n[{i+1}/{N_ITER_RANDOM_SEARCH}] Probando combinación: {params_str}")
        combo_start_time = time.time()

        # Ejecutar el entrenamiento con esta combinación aleatoria
        performance = run_training_for_tuning(
            env=entorno_tune,
            num_episodios=NUM_EPISODIOS_TUNE,
            max_steps=MAX_STEPS_TUNE,
            # Obtener valores muestreados (o usar default si no se muestreó)
            alpha=current_params.get('alpha', config.ALPHA),
            gamma=current_params.get('gamma', config.GAMMA),
            epsilon_start=EPSILON_START_TUNE,
            epsilon_end=current_params.get('epsilon_end', EPSILON_END_TUNE), # Usar muestreado si existe
            epsilon_decay=current_params.get('epsilon_decay', config.EPSILON_DECAY)
        )

        combo_time = time.time() - combo_start_time
        print(f"Resultado (Métrica Rendimiento): {performance:.4f} | Tiempo: {combo_time:.2f}s")
        # Guardar los parámetros usados y el resultado
        results.append({'params': current_params, 'performance': performance})

    total_time_random = time.time() - start_time_random
    print(f"\n--- Random Search Finalizado en {total_time_random:.2f} segundos ---")

    # --- Análisis de Resultados (igual que en Grid Search) ---
    if not results:
        print("No se obtuvieron resultados.")
    else:
        results_df = pd.DataFrame(results)
        # Desanidar el diccionario 'params' en columnas separadas para mejor análisis
        params_df = pd.json_normalize(results_df['params'])
        results_df = pd.concat([results_df.drop('params', axis=1), params_df], axis=1)
        # Ordenar por rendimiento (descendente)
        results_df = results_df.sort_values(by='performance', ascending=False).reset_index(drop=True)

        print("\nMejores Combinaciones Encontradas:")
        # Mostrar más columnas para ver los parámetros
        print(results_df.head())

        # Guardar resultados en un CSV
        results_filename = "random_search_results.csv"
        try:
            results_df.to_csv(results_filename, index=False)
            print(f"\nResultados completos guardados en '{results_filename}'")
        except Exception as e:
            print(f"\nError al guardar resultados en CSV: {e}")

        # Obtener la mejor combinación
        if not results_df.empty:
            best_result_row = results_df.iloc[0]
            best_performance = best_result_row['performance']
            # Reconstruir el diccionario de los mejores parámetros
            best_params_dict = {col: best_result_row[col] for col in param_distributions.keys() if col in best_result_row}


            print("\n---------------------------------------------")
            print("Mejor Combinación Encontrada:")
            for key, value in best_params_dict.items():
                 print(f"  {key}: {value:.5f}" if isinstance(value, float) else f"  {key}: {value}")
            print(f"Mejor Métrica de Rendimiento: {best_performance:.4f}")
            print("---------------------------------------------")
            print("\n>> ¡Actualiza tu archivo 'config.py' con estos valores si parecen prometedores! <<")
            print("(Recuerda que se entrenó con menos episodios, puede que necesites ajustar NUM_EPISODIOS de nuevo)")
        else:
             print("El DataFrame de resultados está vacío.")