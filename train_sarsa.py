# train_sarsa.py

import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import deque
import os
import json

# --- Importar desde nuestros módulos ---
try:
    from tablero import Tablero, StateType
    # Renombrar si se hizo: from rl_agent_utils import ...
    from q_learning_agent import (
        inicializar_q_tables, elegir_acciones, calcular_recompensas, QTableType
    )
    import config
except ImportError as e:
    print(f"Error importando módulos: {e}")
    exit()

# --- Tipo para las métricas (sin cambios) ---
MetricsDict = dict[str, list]

# --- Función de Entrenamiento SARSA ---
def entrenar_sarsa(env: Tablero,
                   # --- Parámetros (igual que Q-Learning) ---
                   num_episodios: int, max_steps: int, alpha: float, gamma: float,
                   epsilon_start: float, epsilon_end: float, epsilon_decay: float,
                   # --- Control y Guardado (igual que Q-Learning) ---
                   log_interval: int = 100,
                   results_foldername: str = "results_sarsa", # Carpeta diferente para SARSA
                   experiment_name: str | None = None,
                   cargar_desde: str | None = None,
                   guardar_q_tables_file: str | None = "q_tables_sarsa.pkl"
                   ) -> tuple[QTableType | None, MetricsDict, str]:
    """
    Entrena agentes usando SARSA y guarda resultados.
    """
    # --- Crear Carpeta de Resultados (similar a Q-Learning) ---
    if experiment_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        experiment_name = f"SARSA_R{env.num_robots}_M{env.filas}x{env.columnas}_E{num_episodios}_{timestamp}"
    output_dir = os.path.join(results_foldername, experiment_name)
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Guardando resultados SARSA en: {output_dir}")
    except OSError as e:
        print(f"Error al crear directorio '{output_dir}': {e}")
        return None, {}, ""

    # --- Guardar Configuración Inicial (igual que Q-Learning) ---
    config_summary = {
        "algoritmo": "SARSA", # Indicar algoritmo
        "entorno": {"filas": env.filas, "columnas": env.columnas, "num_robots": env.num_robots},
        "hiperparametros": {"alpha": alpha, "gamma": gamma, "epsilon_start": epsilon_start, "epsilon_end": epsilon_end, "epsilon_decay": epsilon_decay},
        "entrenamiento": {"num_episodios": num_episodios, "max_steps_per_episode": max_steps},
        # ... añadir otros detalles si se quiere ...
    }
    summary_filepath = os.path.join(output_dir, "summary.txt")
    try:
        with open(summary_filepath, 'w') as f: json.dump(config_summary, f, indent=4)
        print(f"Resumen SARSA guardado en '{summary_filepath}'")
    except IOError as e: print(f"Error guardando resumen: {e}")

    # --- Inicialización (Q-Tables, Epsilon, Métricas - igual que Q-Learning) ---
    q_tables = inicializar_q_tables(env.num_robots) # Ignorar carga por simplicidad
    epsilon = epsilon_start
    metrics: MetricsDict = { # Mismas métricas que antes
        "episodio": [], "recompensa_promedio": [], "pasos_por_episodio": [],
        "epsilon": [], "porcentaje_visitadas": [], "tasa_exito_100ep": [],
        "td_error_promedio": []
    }
    ultimos_100_completados = deque(maxlen=log_interval)

    print(f"--- Iniciando Entrenamiento SARSA ({num_episodios} episodios) ---")
    start_time = time.time()

    # --- Bucle Principal de Entrenamiento SARSA ---
    for episodio in range(num_episodios):
        estado = env.reset()
        # *** SARSA: Elegir la PRIMERA acción ANTES del bucle de pasos ***
        acciones = elegir_acciones(q_tables, estado, epsilon, env)

        recompensa_acumulada_episodio = np.zeros(env.num_robots)
        done = False
        paso = 0
        episodio_completado = False
        td_error_sum = 0.0
        update_count = 0

        # --- Bucle de un episodio ---
        while not done and paso < max_steps:
            paso += 1
            estado_previo = estado
            acciones_previas = acciones # Guardar la acción elegida en el paso anterior

            # 1. Ejecutar las acciones elegidas PREVIAMENTE en el entorno
            estado_siguiente, done = env.step(acciones_previas)

            # 2. Calcular las recompensas recibidas por esas acciones
            recompensas = calcular_recompensas(estado_previo, estado_siguiente, acciones_previas, done, env)

            # *** SARSA: Elegir la SIGUIENTE acción (a') desde el estado siguiente (s') ***
            # Esta acción se usará en la actualización Q y en el *siguiente* paso del bucle
            acciones_siguientes = elegir_acciones(q_tables, estado_siguiente, epsilon, env)

            # 3. Actualizar las Q-tables usando SARSA
            for i in range(env.num_robots):
                if estado_previo[2][i]: # Si estaba activo en s
                    accion_tomada = acciones_previas[i] # a
                    accion_siguiente_elegida = acciones_siguientes[i] # a'
                    recompensa_recibida = recompensas[i] # R

                    q_antiguo = q_tables[i][estado_previo].get(accion_tomada, 0.0)

                    # *** SARSA: Usar Q(s', a') en lugar de max Q(s', a') ***
                    # Obtener el Q-valor para el estado siguiente y la acción *realmente* elegida
                    q_siguiente = 0.0
                    if not done and estado_siguiente[2][i]: # Si no es terminal y sigue activo
                        q_siguiente = q_tables[i][estado_siguiente].get(accion_siguiente_elegida, 0.0)

                    # Fórmula de actualización SARSA
                    delta = recompensa_recibida + gamma * q_siguiente - q_antiguo
                    td_error_sum += abs(delta)
                    update_count += 1
                    q_tables[i][estado_previo][accion_tomada] = q_antiguo + alpha * delta

                    recompensa_acumulada_episodio[i] += recompensa_recibida

            # Actualizar estado y acciones para el siguiente paso
            estado = estado_siguiente
            acciones = acciones_siguientes # La acción a' se convierte en a para la próxima iteración

            if not episodio_completado and len(estado[1]) == env.total_celdas:
                episodio_completado = True

        # --- Fin del episodio - Recopilar Métricas (igual que Q-Learning) ---
        metrics["episodio"].append(episodio + 1)
        metrics["recompensa_promedio"].append(np.mean(recompensa_acumulada_episodio))
        # ... (resto del código de recopilación de métricas igual) ...
        metrics["pasos_por_episodio"].append(paso)
        metrics["epsilon"].append(epsilon)
        metrics["porcentaje_visitadas"].append(len(estado[1]) / env.total_celdas * 100 if env.total_celdas > 0 else 0)
        ultimos_100_completados.append(episodio_completado)
        tasa_exito_actual = sum(ultimos_100_completados) / len(ultimos_100_completados) if ultimos_100_completados else 0.0
        metrics["tasa_exito_100ep"].append(tasa_exito_actual)
        avg_td_error = (td_error_sum / update_count) if update_count > 0 else 0.0
        metrics["td_error_promedio"].append(avg_td_error)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # --- Log Periódico (igual que Q-Learning) ---
        if (episodio + 1) % log_interval == 0:
            tiempo_transcurrido = time.time() - start_time
            print(f"Ep(SARSA): {episodio+1}/{num_episodios} | "
                  f"Exito: {tasa_exito_actual*100:.1f}% | "
                  f"Visit: {np.mean(metrics['porcentaje_visitadas'][-log_interval:]):.1f}% | "
                  f"Eps: {epsilon:.4f} | Rec: {np.mean(metrics['recompensa_promedio'][-log_interval:]):.3f} | "
                  f"T: {tiempo_transcurrido:.1f}s")

    # --- Fin del Entrenamiento SARSA ---
    tiempo_total = time.time() - start_time
    print(f"--- Entrenamiento SARSA Finalizado en {tiempo_total:.2f} segundos ---")

    # --- Guardado de Métricas y Q-Tables (igual, pero en la carpeta SARSA) ---
    metrics_filepath_csv = os.path.join(output_dir, "sarsa_metrics.csv")
    try:
        import pandas as pd
        pd.DataFrame(metrics).to_csv(metrics_filepath_csv, index=False)
        print(f"Métricas SARSA guardadas en '{metrics_filepath_csv}'")
    except Exception as e: print(f"Error guardando métricas CSV: {e}")

    if guardar_q_tables_file:
        q_table_path = os.path.join(output_dir, guardar_q_tables_file)
        try:
            with open(q_table_path, 'wb') as f:
                 q_tables_serializable = [{k: dict(v) for k, v in q_t.items()} for q_t in q_tables]
                 pickle.dump(q_tables_serializable, f)
            print(f"Q-Tables SARSA guardadas en '{q_table_path}'")
        except Exception as e: print(f"Error guardando Q-Tables SARSA: {e}")

    # Añadir resultados finales a summary.txt
    try:
        with open(summary_filepath, 'a') as f:
            f.write("\n\n--- Resultados SARSA ---\n")
            f.write(f"Tiempo total: {tiempo_total:.2f}s\n")
            # ... (añadir métricas finales clave) ...
        print(f"Resultados finales SARSA añadidos a '{summary_filepath}'")
    except IOError as e: print(f"Error añadiendo resultados a resumen: {e}")

    return q_tables, metrics, output_dir


# --- Bloque Principal (Adaptado para SARSA) ---
if __name__ == "__main__":
    # Usar configuración de config.py
    entorno_sarsa = Tablero(filas=config.FILAS, columnas=config.COLUMNAS,
                            num_robots=config.N_ROBOTS, posicion_inicial=config.POSICION_INICIAL)

    # Llamar a la nueva función de entrenamiento SARSA
    q_tables_sarsa, metrics_sarsa, results_dir_sarsa = entrenar_sarsa(
        env=entorno_sarsa,
        num_episodios=config.NUM_EPISODIOS,
        max_steps=config.MAX_STEPS_PER_EPISODE,
        alpha=config.ALPHA, gamma=config.GAMMA,
        epsilon_start=config.EPSILON_START, epsilon_end=config.EPSILON_END,
        epsilon_decay=config.EPSILON_DECAY,
        log_interval=config.EPISODIOS_PARA_LOG,
        results_foldername="results_sarsa", # Guardar en carpeta separada
        guardar_q_tables_file="q_tables_sarsa.pkl"
    )

    # --- Graficar Métricas SARSA (igual que antes, pero con datos SARSA) ---
    if config.GENERAR_GRAFICO and metrics_sarsa and results_dir_sarsa:
        print("\n--- Generando Gráficos SARSA ---")
        # Reutilizar la función plot_and_save_metric (debería estar definida o importada)
        try:
             from train_qlearning import plot_and_save_metric 
             pass
        except NameError:
             # Definirla aquí si es necesario
             def plot_and_save_metric(episodes, data, title, ylabel, filename, window=100, show_avg=True):
                """Función auxiliar para crear y guardar un gráfico individual."""
                plt.figure(figsize=(10, 5))
                plt.plot(episodes, data, alpha=0.6, label=ylabel)
                if show_avg:
                    if len(data) >= window:
                        promedio_movil = np.convolve(data, np.ones(window)/window, mode='valid')
                        plt.plot(episodes[window-1:], promedio_movil, color='red', linewidth=2, label=f'Prom. Móvil {window}ep')
                plt.title(title)
                plt.xlabel('Episodio')
                plt.ylabel(ylabel)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                try:
                    plt.savefig(filename)
                    print(f"Gráfico guardado: {filename}")
                except Exception as e:
                    print(f"Error al guardar gráfico '{filename}': {e}")
                plt.close() # Cerrar la figura para liberar memoria

        ep = metrics_sarsa['episodio']
        win = config.VENTANA_PROMEDIO_MOVIL

        plot_and_save_metric(ep, metrics_sarsa['recompensa_promedio'], 'SARSA Recompensa Promedio',
                             'Recompensa Prom.', os.path.join(results_dir_sarsa, 'plot_sarsa_recompensa.png'), win)
        # ... (Llamar a plot_and_save_metric para las otras métricas)...
        plot_and_save_metric(ep, metrics_sarsa['pasos_por_episodio'], 'SARSA Pasos por Episodio',
                             'Pasos', os.path.join(results_dir_sarsa, 'plot_sarsa_pasos.png'), win)
        plot_and_save_metric(ep, metrics_sarsa['porcentaje_visitadas'], 'SARSA % Celdas Visitadas',
                             '% Visitadas', os.path.join(results_dir_sarsa, 'plot_sarsa_visitadas.png'), win)
        plot_and_save_metric(ep, metrics_sarsa['tasa_exito_100ep'], f'SARSA Tasa de Éxito ({config.EPISODIOS_PARA_LOG}ep)',
                             'Tasa Éxito', os.path.join(results_dir_sarsa, 'plot_sarsa_exito.png'), win, show_avg=False)
        plot_and_save_metric(ep, metrics_sarsa['epsilon'], 'SARSA Decaimiento de Epsilon',
                             'Epsilon', os.path.join(results_dir_sarsa, 'plot_sarsa_epsilon.png'), win, show_avg=False)
        plot_and_save_metric(ep, metrics_sarsa['td_error_promedio'], 'SARSA Error TD Promedio',
                             'Error TD Prom.', os.path.join(results_dir_sarsa, 'plot_sarsa_td_error.png'), win)


    # --- Evaluación SARSA (igual que Q-Learning, pero usando q_tables_sarsa) ---
    if config.EVALUAR_AL_FINAL and q_tables_sarsa:
        print("\n--- Evaluación SARSA (No implementada aquí, requiere `evaluar_politica`) ---")
         