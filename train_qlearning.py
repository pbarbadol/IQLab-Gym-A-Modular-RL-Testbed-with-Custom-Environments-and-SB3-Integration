# train.py (refactorizado)

import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import deque
import os # Necesario para crear carpetas
import json # Otra opción para guardar config/métricas

# --- Importar desde nuestros módulos ---
try:
    from tablero import Tablero, StateType
    from q_learning_agent import (
        inicializar_q_tables, elegir_acciones, calcular_recompensas, QTableType
    )
    import config # Importar toda la configuración
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de que 'tablero.py', 'q_learning_agent.py' y 'config.py' estén en el mismo directorio o accesibles.")
    exit()

# --- Tipo para las métricas (sin cambios) ---
MetricsDict = dict[str, list]

def entrenar(env: Tablero,
             # --- Parámetros pasados explícitamente ---
             num_episodios: int, max_steps: int, alpha: float, gamma: float,
             epsilon_start: float, epsilon_end: float, epsilon_decay: float,
             # --- Parámetros de control y guardado ---
             log_interval: int = 100,
             results_foldername: str = "results", # Carpeta base para resultados
             experiment_name: str | None = None, # Nombre específico para la subcarpeta
             cargar_desde: str | None = None,
             guardar_q_tables_file: str | None = "q_tables.pkl" # Nombre archivo dentro de la carpeta
             ) -> tuple[QTableType | None, MetricsDict, str]: # Devuelve Q-tables, métricas y ruta de resultados
    """
    Entrena agentes IQL, recopila métricas y guarda resultados en una carpeta específica.
    """
    # --- Crear Carpeta de Resultados ---
    if experiment_name is None:
        # Generar nombre automático si no se proporciona
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        experiment_name = f"R{env.num_robots}_M{env.filas}x{env.columnas}_E{num_episodios}_{timestamp}"

    output_dir = os.path.join(results_foldername, experiment_name)
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Guardando resultados en: {output_dir}")
    except OSError as e:
        print(f"Error al crear directorio de resultados '{output_dir}': {e}")
        # Podríamos decidir continuar sin guardar o salir
        return None, {}, "" # Indicar fallo

    # --- Guardar Configuración Inicial ---
    config_summary = {
        "entorno": {
            "filas": env.filas, "columnas": env.columnas, "num_robots": env.num_robots,
            "total_celdas": env.total_celdas, "posicion_inicial": env._posicion_inicial_robots
        },
        "hiperparametros": {
            "alpha": alpha, "gamma": gamma, "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end, "epsilon_decay": epsilon_decay
        },
        "entrenamiento": {
            "num_episodios": num_episodios, "max_steps_per_episode": max_steps,
            "log_interval": log_interval
        },
        "archivos": {
             "cargar_q_tables_desde": cargar_desde,
             "guardar_q_tables_en": os.path.join(output_dir, guardar_q_tables_file) if guardar_q_tables_file else None
        }
    }
    summary_filepath = os.path.join(output_dir, "summary.txt")
    try:
        with open(summary_filepath, 'w') as f:
            # Usar json.dump para un formato legible y fácil de parsear
            json.dump(config_summary, f, indent=4)
            # Alternativa simple con print:
            # f.write("--- Configuración del Experimento ---\n")
            # f.write(f"Nombre: {experiment_name}\n")
            # ... escribir más detalles ...
        print(f"Resumen de configuración guardado en '{summary_filepath}'")
    except IOError as e:
        print(f"Error al guardar resumen de configuración: {e}")


    # --- Inicialización (Q-Tables, Epsilon, Métricas) ---
    q_tables = inicializar_q_tables(env.num_robots) # Ignoramos cargar por simplicidad aquí, añadir si es necesario
    epsilon = epsilon_start
    metrics: MetricsDict = {
        "episodio": [], "recompensa_promedio": [], "pasos_por_episodio": [],
        "epsilon": [], "porcentaje_visitadas": [], "tasa_exito_100ep": [],
        "td_error_promedio": []
    }
    ultimos_100_completados = deque(maxlen=log_interval) # Usar log_interval como ventana

    print(f"--- Iniciando Entrenamiento ({num_episodios} episodios) ---")
    start_time = time.time()

    # --- Bucle Principal de Entrenamiento (igual que antes) ---
    for episodio in range(num_episodios):
        estado = env.reset()
        recompensa_acumulada_episodio = np.zeros(env.num_robots)
        done = False
        paso = 0
        episodio_completado = False
        td_error_sum = 0.0
        update_count = 0

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
                    td_error_sum += abs(delta)
                    update_count += 1
                    q_tables[i][estado_previo][accion_tomada] = q_antiguo + alpha * delta
                    recompensa_acumulada_episodio[i] += recompensa_recibida

            estado = estado_siguiente
            if not episodio_completado and len(estado[1]) == env.total_celdas:
                episodio_completado = True

        # --- Fin del episodio - Recopilar Métricas ---
        metrics["episodio"].append(episodio + 1)
        metrics["recompensa_promedio"].append(np.mean(recompensa_acumulada_episodio))
        metrics["pasos_por_episodio"].append(paso)
        metrics["epsilon"].append(epsilon)
        metrics["porcentaje_visitadas"].append(len(estado[1]) / env.total_celdas * 100)
        ultimos_100_completados.append(episodio_completado)
        tasa_exito_actual = sum(ultimos_100_completados) / len(ultimos_100_completados)
        metrics["tasa_exito_100ep"].append(tasa_exito_actual)
        avg_td_error = (td_error_sum / update_count) if update_count > 0 else 0.0
        metrics["td_error_promedio"].append(avg_td_error)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # --- Log Periódico (sin cambios) ---
        if (episodio + 1) % log_interval == 0:
            tiempo_transcurrido = time.time() - start_time
            print(f"Ep: {episodio+1}/{num_episodios} | "
                  f"Exito: {tasa_exito_actual*100:.1f}% | "
                  f"Visit: {np.mean(metrics['porcentaje_visitadas'][-log_interval:]):.1f}% | "
                  f"Eps: {epsilon:.4f} | Rec: {np.mean(metrics['recompensa_promedio'][-log_interval:]):.3f} | "
                  f"T: {tiempo_transcurrido:.1f}s")

    # --- Fin del Entrenamiento ---
    tiempo_total = time.time() - start_time
    print(f"--- Entrenamiento Finalizado en {tiempo_total:.2f} segundos ---")

    # --- Guardar Métricas (como CSV o JSON) ---
    metrics_filepath_csv = os.path.join(output_dir, "training_metrics.csv")
    try:
        import pandas as pd
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(metrics_filepath_csv, index=False)
        print(f"Datos de métricas guardados en '{metrics_filepath_csv}'")
    except ImportError:
        print("Pandas no instalado. Guardando métricas como JSON.")
        metrics_filepath_json = os.path.join(output_dir, "training_metrics.json")
        try:
            with open(metrics_filepath_json, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"Datos de métricas guardados en '{metrics_filepath_json}'")
        except IOError as e:
            print(f"Error al guardar métricas como JSON: {e}")
    except Exception as e:
        print(f"Error al guardar métricas como CSV: {e}")


    # --- Guardar Q-Tables ---
    if guardar_q_tables_file:
        q_table_path = os.path.join(output_dir, guardar_q_tables_file)
        try:
            with open(q_table_path, 'wb') as f:
                 q_tables_serializable = [{k: dict(v) for k, v in q_t.items()} for q_t in q_tables]
                 pickle.dump(q_tables_serializable, f)
            print(f"Q-Tables guardadas en '{q_table_path}'")
        except Exception as e:
            print(f"Error al guardar Q-Tables: {e}")

    # --- Guardar Resultados Finales en summary.txt ---
    try:
        with open(summary_filepath, 'a') as f: # Abrir en modo 'append'
            f.write("\n\n--- Resultados del Entrenamiento ---\n")
            f.write(f"Tiempo total de entrenamiento: {tiempo_total:.2f} segundos\n")
            # Añadir métricas finales clave
            if metrics["recompensa_promedio"]:
                 f.write(f"Recompensa promedio final ({log_interval} ep): {np.mean(metrics['recompensa_promedio'][-log_interval:]):.4f}\n")
            if metrics["tasa_exito_100ep"]:
                 f.write(f"Tasa de éxito final ({log_interval} ep): {metrics['tasa_exito_100ep'][-1]*100:.2f}%\n")
            if metrics["porcentaje_visitadas"]:
                 f.write(f"Porcentaje visitadas final ({log_interval} ep): {np.mean(metrics['porcentaje_visitadas'][-log_interval:]):.2f}%\n")

        print(f"Resultados finales añadidos a '{summary_filepath}'")
    except IOError as e:
        print(f"Error al añadir resultados al resumen: {e}")


    return q_tables, metrics, output_dir # Devolver ruta para usarla después

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


# --- Bloque Principal ---
if __name__ == "__main__":
    # Crear entorno desde config
    # En train.py y visualize_sim.py
    entorno = Tablero(
        filas=config.FILAS,
        columnas=config.COLUMNAS,
        num_robots=config.N_ROBOTS,
        posicion_inicial=config.POSICION_INICIAL,
        posicion_carga=config.POSICION_CARGA,      # <<< NUEVO
        bateria_maxima=config.BATERIA_MAXIMA,      # <<< NUEVO
        bateria_inicial=config.BATERIA_INICIAL       # <<< NUEVO
    )

    # Entrenar y obtener resultados y ruta
    q_tables_finales, metrics_data, results_dir = entrenar(
        env=entorno,
        num_episodios=config.NUM_EPISODIOS,
        max_steps=config.MAX_STEPS_PER_EPISODE,
        alpha=config.ALPHA, gamma=config.GAMMA,
        epsilon_start=config.EPSILON_START, epsilon_end=config.EPSILON_END,
        epsilon_decay=config.EPSILON_DECAY,
        log_interval=config.EPISODIOS_PARA_LOG,
        # cargar_desde=config.CARGAR_QTABLES_FILENAME, # Añadir si se usa
        guardar_q_tables_file=config.GUARDAR_QTABLES_FILENAME.split('/')[-1] # Solo nombre archivo
    )

    # --- Generar y Guardar Gráficos Individuales ---
    if config.GENERAR_GRAFICO and metrics_data and results_dir:
        print("\n--- Generando Gráficos ---")
        ep = metrics_data['episodio']
        win = config.VENTANA_PROMEDIO_MOVIL

        plot_and_save_metric(ep, metrics_data['recompensa_promedio'], 'Recompensa Promedio por Episodio',
                             'Recompensa Prom.', os.path.join(results_dir, 'plot_recompensa.png'), win)

        plot_and_save_metric(ep, metrics_data['pasos_por_episodio'], 'Pasos por Episodio',
                             'Pasos', os.path.join(results_dir, 'plot_pasos.png'), win)

        plot_and_save_metric(ep, metrics_data['porcentaje_visitadas'], 'Porcentaje de Celdas Visitadas',
                             '% Visitadas', os.path.join(results_dir, 'plot_visitadas.png'), win)

        plot_and_save_metric(ep, metrics_data['tasa_exito_100ep'], f'Tasa de Éxito ({config.EPISODIOS_PARA_LOG}ep)',
                             'Tasa Éxito', os.path.join(results_dir, 'plot_exito.png'), win, show_avg=False) # Avg no aplica bien aquí

        plot_and_save_metric(ep, metrics_data['epsilon'], 'Decaimiento de Epsilon',
                             'Epsilon', os.path.join(results_dir, 'plot_epsilon.png'), win, show_avg=False)

        plot_and_save_metric(ep, metrics_data['td_error_promedio'], 'Error TD Promedio por Episodio',
                             'Error TD Prom.', os.path.join(results_dir, 'plot_td_error.png'), win)

    # --- Evaluación (sin cambios, excepto usar Q-tables devueltas) ---
    if config.EVALUAR_AL_FINAL and q_tables_finales:
        # Reconstruir Q-tables a defaultdict si se cargaron desde pickle y se serializaron
        q_tables_eval = inicializar_q_tables(entorno.num_robots)
        # Asumimos que q_tables_finales son los defaultdicts devueltos por entrenar
        # Si hubieran sido cargados y deserializados como dicts, necesitaríamos convertirlos
        q_tables_eval = q_tables_finales

        # Reutilizamos la función evaluar_politica (definida localmente o importada)
        # ... (código de importar o definir evaluar_politica como en la respuesta anterior) ...
        try:
             from q_learning_agent import evaluar_politica
        except ImportError:
            def evaluar_politica(env: Tablero, q_tables: QTableType, max_steps: int, render: bool = True, pause: float = 0.3):
                    print("\n--- Evaluación de la Política Aprendida (Epsilon = 0) ---")
                    estado = env.reset()
                    if render:
                        try:
                            env.render()
                            time.sleep(pause * 2)
                        except AttributeError:
                            print("Advertencia: El método render() no está implementado en Tablero.")
                            render = False # Desactivar render si no existe

                    done = False
                    paso = 0
                    recompensa_total_eval = np.zeros(env.num_robots)

                    while not done and paso < max_steps:
                        paso += 1
                        acciones = elegir_acciones(q_tables, estado, 0.0, env)
                        estado_siguiente, done = env.step(acciones)
                        recompensas = calcular_recompensas(estado, estado_siguiente, acciones, done, env)
                        recompensa_total_eval += recompensas
                        estado = estado_siguiente

                        if render:
                            print(f"\nPaso {paso} | Acciones: {acciones} | Recompensas: [{', '.join(f'{r:.2f}' for r in recompensas)}]")
                            env.render()
                            if pause > 0: time.sleep(pause)

                    print("\n--- Fin Evaluación ---")
                    print(f"Terminado en {paso} pasos. Done={done}")
                    print(f"Recompensa total: {recompensa_total_eval} (Promedio: {np.mean(recompensa_total_eval):.2f})")
                    print(f"Celdas visitadas: {len(estado[1])}/{env.total_celdas}")

        evaluar_politica(
            env=entorno,
            q_tables=q_tables_eval, # Usar las Q-tables potencialmente convertidas
            max_steps=config.MAX_STEPS_EVALUACION,
            render=config.RENDER_EVALUACION,
            pause=config.PAUSA_RENDER_EVAL
        )