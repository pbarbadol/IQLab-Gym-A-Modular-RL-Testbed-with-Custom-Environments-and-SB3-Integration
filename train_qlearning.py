# train.py (refactorizado para Opción 3)

import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import deque
import os
import json
import random # Importante para fijar semillas

# --- Importar desde nuestros módulos ---
try:
    from tablero import Tablero, StateType # Asumo que StateType está aquí o es un alias conocido
    from q_learning_agent import (
        inicializar_q_tables, elegir_acciones, calcular_recompensas, QTableType,
        evaluar_politica # Importar evaluar_politica directamente
    )
    import config # Para la ejecución directa
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de que 'tablero.py', 'q_learning_agent.py' y 'config.py' estén en el mismo directorio o accesibles.")
    exit()

# --- Tipo para las métricas (sin cambios) ---
MetricsDict = dict[str, list]

# ---------------------------------------------------------------------------
# LA FUNCIÓN entrenar() SE MANTIENE CASI IGUAL INTERNAMENTE
# SOLO ASEGÚRATE DE QUE USE LOS PARÁMETROS PASADOS Y NO config.* DIRECTAMENTE
# Y QUE EL GUARDADO DE RESULTADOS RESPETE results_foldername y experiment_name
# ---------------------------------------------------------------------------
def entrenar(env: Tablero,
             num_episodios: int, max_steps: int, alpha: float, gamma: float,
             epsilon_start: float, epsilon_end: float, epsilon_decay: float,
             log_interval: int = 100,
             results_foldername: str = "results", # Carpeta base para este grupo/tipo de experimento
             experiment_name: str | None = None, # Nombre específico para la subcarpeta de ESTE trial
             cargar_desde: str | None = None,
             guardar_q_tables_file: str | None = "q_tables_final.pkl"
             ) -> tuple[QTableType | None, MetricsDict, str]:
    """
    Entrena agentes IQL, recopila métricas y guarda resultados en una carpeta específica.
    MODIFICADO: Toma results_foldername y experiment_name para un control más fino del guardado.
    """
    # --- Crear Carpeta de Resultados para ESTE TRIAL ---
    if experiment_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        experiment_name = f"R{env.num_robots}_M{env.filas}x{env.columnas}_E{num_episodios}_{timestamp}"

    # El output_dir ahora es la carpeta específica del trial DENTRO de results_foldername
    output_dir_trial = os.path.join(results_foldername, experiment_name)
    try:
        os.makedirs(output_dir_trial, exist_ok=True)
        # No imprimir "Guardando resultados en:" aquí, se hará en run_training_session
    except OSError as e:
        print(f"Error al crear directorio de resultados del trial '{output_dir_trial}': {e}")
        return None, {}, ""

    # --- Guardar Configuración de ESTE TRIAL ---
    config_summary_trial = {
        "entorno": {
            "filas": env.filas, "columnas": env.columnas, "num_robots": env.num_robots,
            "total_celdas": env.total_celdas,
            # Convertir tupla a lista para serialización JSON si es necesario para _posicion_inicial_robots
            "posicion_inicial": list(env._posicion_inicial_robots) if env._posicion_inicial_robots else None
        },
        "hiperparametros": {
            "alpha": alpha, "gamma": gamma, "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end, "epsilon_decay": epsilon_decay
        },
        "entrenamiento_trial": { # Diferenciar de la config general del grupo
            "num_episodios": num_episodios, "max_steps_per_episode": max_steps,
            "log_interval": log_interval
        },
        "archivos_trial": {
             "cargar_q_tables_desde": cargar_desde,
             "guardar_q_tables_en": os.path.join(output_dir_trial, guardar_q_tables_file) if guardar_q_tables_file else None
        }
    }
    summary_filepath_trial = os.path.join(output_dir_trial, "summary_trial.txt") # Renombrado
    try:
        with open(summary_filepath_trial, 'w') as f:
            json.dump(config_summary_trial, f, indent=4)
    except IOError as e:
        print(f"Error al guardar resumen de config del trial: {e}")

    # --- Inicialización (Q-Tables, Epsilon, Métricas) ---
    # Modificación para manejar la carga de Q-tables
    if cargar_desde and os.path.exists(cargar_desde):
        print(f"Cargando Q-Tables desde: {cargar_desde}")
        # Necesitas una función para cargar q_tables que sea compatible con el formato guardado
        # Asumiré que tienes q_learning_agent.cargar_q_tables(num_robots, filename)
        from q_learning_agent import cargar_q_tables # Asegúrate que esta función exista y funcione
        q_tables = cargar_q_tables(env.num_robots, cargar_desde)
    else:
        if cargar_desde: # Si se especificó pero no existe
            print(f"Advertencia: Archivo {cargar_desde} para Q-Tables no encontrado. Inicializando Q-tables vacías.")
        q_tables = inicializar_q_tables(env.num_robots)

    epsilon = epsilon_start
    metrics: MetricsDict = {
        "episodio": [], "recompensa_promedio": [], "pasos_por_episodio": [],
        "epsilon": [], "porcentaje_visitadas": [], "tasa_exito_100ep": [],
        "td_error_promedio": []
    }
    # Usar log_interval como ventana, o un valor fijo si es más apropiado (e.g., 100)
    ultimos_X_completados = deque(maxlen=log_interval if log_interval > 0 else 100)

    # print(f"--- Iniciando Entrenamiento para {experiment_name} ({num_episodios} episodios) ---") # Movido a run_training_session
    start_time_trial = time.time()

    # --- Bucle Principal de Entrenamiento (sin cambios en la lógica interna) ---
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
                if estado_previo[2][i]: # Si el robot estaba activo
                    accion_tomada = acciones[i]
                    recompensa_recibida = recompensas[i]
                    q_antiguo = q_tables[i][estado_previo].get(accion_tomada, 0.0)
                    mejor_q_siguiente = 0.0
                    if not done and estado_siguiente[2][i]: # Si el robot sigue activo en el siguiente estado
                        acciones_validas_siguiente = env.get_valid_actions(i) # Pasar estado_siguiente explícitamente
                        if acciones_validas_siguiente:
                            q_valores_siguientes = q_tables[i][estado_siguiente]
                            # Usar .get(a, 0.0) por si una acción válida aún no tiene entrada
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
        metrics["recompensa_promedio"].append(np.mean(recompensa_acumulada_episodio) if env.num_robots > 0 else 0.0)
        metrics["pasos_por_episodio"].append(paso)
        metrics["epsilon"].append(epsilon)
        metrics["porcentaje_visitadas"].append(len(estado[1]) / env.total_celdas * 100 if env.total_celdas > 0 else 0)
        
        ultimos_X_completados.append(episodio_completado)
        if len(ultimos_X_completados) > 0:
            tasa_exito_actual = sum(ultimos_X_completados) / len(ultimos_X_completados)
        else:
            tasa_exito_actual = 0.0
        metrics["tasa_exito_100ep"].append(tasa_exito_actual) # El nombre de la métrica es histórico, podría ser "tasa_exito_ventana"
        
        avg_td_error = (td_error_sum / update_count) if update_count > 0 else 0.0
        metrics["td_error_promedio"].append(avg_td_error)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if log_interval > 0 and (episodio + 1) % log_interval == 0:
            tiempo_transcurrido_parcial = time.time() - start_time_trial
            # Calcular promedios sobre la ventana de log_interval para el print
            avg_rec_log_win = np.mean(metrics['recompensa_promedio'][-log_interval:])
            avg_visit_log_win = np.mean(metrics['porcentaje_visitadas'][-log_interval:])

            print(f"  Trial: {experiment_name} - Ep: {episodio+1}/{num_episodios} | "
                  f"Éxito (últ.{len(ultimos_X_completados)}ep): {tasa_exito_actual*100:.1f}% | "
                  f"Visit (últ.{log_interval}ep): {avg_visit_log_win:.1f}% | "
                  f"Eps: {epsilon:.4f} | Rec (últ.{log_interval}ep): {avg_rec_log_win:.3f} | "
                  f"T: {tiempo_transcurrido_parcial:.1f}s")

    tiempo_total_trial = time.time() - start_time_trial
    # print(f"--- Entrenamiento del Trial '{experiment_name}' Finalizado en {tiempo_total_trial:.2f} segundos ---") # Movido

    # --- Guardar Métricas (como CSV o JSON) ---
    metrics_filepath_csv = os.path.join(output_dir_trial, "training_metrics.csv")
    try:
        import pandas as pd
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(metrics_filepath_csv, index=False)
    except ImportError:
        metrics_filepath_json = os.path.join(output_dir_trial, "training_metrics.json")
        try:
            with open(metrics_filepath_json, 'w') as f: json.dump(metrics, f)
        except IOError as e: print(f"Error guardando métricas JSON: {e}")
    except Exception as e: print(f"Error guardando métricas CSV: {e}")

    # --- Guardar Q-Tables ---
    if guardar_q_tables_file:
        q_table_path = os.path.join(output_dir_trial, guardar_q_tables_file)
        try:
            # Tu método de serialización de Q-Table (el que tenías está bien)
            q_tables_serializable = []
            for q_t in q_tables:
                # Convertir defaultdict interno a dict normal
                q_t_dict = {k_state: dict(v_actions) for k_state, v_actions in q_t.items()}
                q_tables_serializable.append(q_t_dict)
            with open(q_table_path, 'wb') as f:
                 pickle.dump(q_tables_serializable, f)
        except Exception as e:
            print(f"Error al guardar Q-Tables en {q_table_path}: {e}")

    # --- Añadir Resultados Finales al summary_trial.txt ---
    try:
        with open(summary_filepath_trial, 'a') as f:
            f.write("\n\n--- Resultados del Trial ---\n")
            f.write(f"Tiempo total de entrenamiento del trial: {tiempo_total_trial:.2f} segundos\n")
            window_size = log_interval if log_interval > 0 and len(metrics['recompensa_promedio']) >= log_interval else len(metrics['recompensa_promedio'])
            if window_size > 0:
                f.write(f"Recompensa promedio final ({window_size} ep): {np.mean(metrics['recompensa_promedio'][-window_size:]):.4f}\n")
                f.write(f"Tasa de éxito final ({window_size} ep): {np.mean(metrics['tasa_exito_100ep'][-window_size:])*100:.2f}%\n") # Usa la métrica de tasa de éxito ya calculada
                f.write(f"Porcentaje visitadas final ({window_size} ep): {np.mean(metrics['porcentaje_visitadas'][-window_size:]):.2f}%\n")
            else:
                f.write("No hay suficientes episodios para calcular métricas finales.\n")
    except IOError as e:
        print(f"Error al añadir resultados al resumen del trial: {e}")

    return q_tables, metrics, output_dir_trial


# ---------------------------------------------------------------------------
# FUNCIÓN plot_and_save_metric SE MANTIENE IGUAL
# ---------------------------------------------------------------------------
def plot_and_save_metric(episodes, data, title, ylabel, filename, window=100, show_avg=True):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, data, alpha=0.6, label=ylabel)
    if show_avg and len(data) >= window and window > 0: # Asegurar que window sea positivo
        promedio_movil = np.convolve(data, np.ones(window)/window, mode='valid')
        # Asegurar que episodes[window-1:] tenga la misma longitud que promedio_movil
        start_index = window - 1
        if start_index < len(episodes):
             plt.plot(episodes[start_index:start_index+len(promedio_movil)], promedio_movil, color='red', linewidth=2, label=f'Prom. Móvil {window}ep')
    plt.title(title)
    plt.xlabel('Episodio')
    plt.ylabel(ylabel)
    if show_avg or ylabel: # Solo mostrar leyenda si hay algo que etiquetar
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(filename)
    except Exception as e:
        print(f"Error al guardar gráfico '{filename}': {e}")
    plt.close()


# ---------------------------------------------------------------------------
# NUEVA FUNCIÓN ENVOLTORIO PARA UNA SESIÓN DE ENTRENAMIENTO COMPLETA
# ---------------------------------------------------------------------------
def run_training_session(params: dict):
    """
    Función principal para ejecutar una sesión de entrenamiento completa,
    incluyendo creación de entorno, entrenamiento, ploteo y evaluación opcional.
    """
    session_name = params.get("EXPERIMENT_NAME", f"unnamed_session_{time.strftime('%Y%m%d-%H%M%S')}")
    results_base_folder = params.get("RESULTS_FOLDERNAME", "results_default_session")

    print(f"\n--- Iniciando Sesión de Entrenamiento: {session_name} ---")
    print(f"--- Guardando en subcarpeta de: {results_base_folder} ---")
    print(f"  Config: F={params['FILAS']} C={params['COLUMNAS']} R={params['N_ROBOTS']} | Episodes: {params['NUM_EPISODIOS']}")
    print(f"  Hyperparams: A={params['ALPHA']:.5f} G={params['GAMMA']:.5f} ED={params['EPSILON_DECAY']:.5f}")
    if params.get('RANDOM_SEED') is not None:
        print(f"  Random Seed: {params['RANDOM_SEED']}")

    # --- Configuración de Semilla Aleatoria ---
    seed = params.get("RANDOM_SEED")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        # Si Tablero u otros componentes usan 'random' y necesitan ser sembrados, hacerlo aquí.
        # Ejemplo: if hasattr(Tablero, 'seed_global_random'): Tablero.seed_global_random(seed)

    # --- Crear Entorno ---
    pos_inicial_config = params.get("POSICION_INICIAL")
    if isinstance(pos_inicial_config, str): # Por si se pasa como JSON string desde línea de cmd (no es el caso aquí)
        if pos_inicial_config.lower() == "none": pos_inicial_config = None
        else:
            try: pos_inicial_config = json.loads(pos_inicial_config)
            except: pos_inicial_config = None # Fallback

    entorno = Tablero(filas=params["FILAS"], columnas=params["COLUMNAS"],
                      num_robots=params["N_ROBOTS"], posicion_inicial=pos_inicial_config)

    # --- Ejecutar Entrenamiento ---
    # La función 'entrenar' se encarga de la subcarpeta específica del 'experiment_name'
    q_tables_finales, metrics_data, trial_output_dir = entrenar(
        env=entorno,
        num_episodios=params["NUM_EPISODIOS"],
        max_steps=params["MAX_STEPS_PER_EPISODE"],
        alpha=params["ALPHA"], gamma=params["GAMMA"],
        epsilon_start=params["EPSILON_START"], epsilon_end=params["EPSILON_END"],
        epsilon_decay=params["EPSILON_DECAY"],
        log_interval=params["EPISODIOS_PARA_LOG"],
        results_foldername=results_base_folder, # Carpeta del grupo de experimento
        experiment_name=session_name,           # Nombre del trial/sesión específica
        cargar_desde=params.get("CARGAR_QTABLES_FILENAME"),
        guardar_q_tables_file=params["GUARDAR_QTABLES_FILENAME"].split('/')[-1] if params.get("GUARDAR_QTABLES_FILENAME") else "q_tables_final.pkl"
    )

    print(f"--- Entrenamiento para '{session_name}' finalizado. Resultados en: {trial_output_dir} ---")

    # --- Generar Gráficos Individuales (si se especifica en params) ---
    if params.get("GENERAR_GRAFICO_INDIVIDUAL") and metrics_data and trial_output_dir:
        print(f"--- Generando Gráficos Individuales para '{session_name}' ---")
        ep = metrics_data['episodio']
        win = params.get("VENTANA_PROMEDIO_MOVIL", 100)
        # Usar trial_output_dir que es la carpeta específica de este trial
        plot_and_save_metric(ep, metrics_data['recompensa_promedio'], 'Recompensa Promedio', 'Recompensa', os.path.join(trial_output_dir, 'plot_recompensa.png'), win)
        plot_and_save_metric(ep, metrics_data['pasos_por_episodio'], 'Pasos por Episodio', 'Pasos', os.path.join(trial_output_dir, 'plot_pasos.png'), win)
        plot_and_save_metric(ep, metrics_data['porcentaje_visitadas'], '% Celdas Visitadas', '% Visitadas', os.path.join(trial_output_dir, 'plot_visitadas.png'), win)
        plot_and_save_metric(ep, metrics_data['tasa_exito_100ep'], f'Tasa Éxito (ventana {params.get("EPISODIOS_PARA_LOG",100)}ep)', 'Tasa Éxito', os.path.join(trial_output_dir, 'plot_exito.png'), win, show_avg=False)
        plot_and_save_metric(ep, metrics_data['epsilon'], 'Decaimiento Epsilon', 'Epsilon', os.path.join(trial_output_dir, 'plot_epsilon.png'), win, show_avg=False)
        plot_and_save_metric(ep, metrics_data['td_error_promedio'], 'Error TD Promedio', 'Error TD', os.path.join(trial_output_dir, 'plot_td_error.png'), win)

    # --- Evaluación (si se especifica en params) ---
    if params.get("EVALUAR_SESION_FINAL") and q_tables_finales:
        print(f"\n--- Evaluando Política para '{session_name}' (Epsilon = 0) ---")
        # La función evaluar_politica ya está importada de q_learning_agent
        evaluar_politica(
            env=entorno, # Reutilizar el entorno, se reseteará internamente
            q_tables=q_tables_finales,
            max_steps=params.get("MAX_STEPS_EVALUACION", params["MAX_STEPS_PER_EPISODE"] * 2),
            render=params.get("RENDER_EVALUACION", False), # Usualmente False para la batería
            pause=params.get("PAUSA_RENDER_EVAL", 0.1)
        )
    print(f"--- Sesión '{session_name}' Completada. ---")


# --- Bloque Principal (si train.py se ejecuta directamente) ---
if __name__ == "__main__":
    print("### Ejecutando train.py directamente usando configuración de config.py ###")

    # Construir diccionario de parámetros desde config.py
    # Los nombres de las claves deben coincidir con los que espera run_training_session
    direct_run_params = {
        "FILAS": config.FILAS, "COLUMNAS": config.COLUMNAS, "N_ROBOTS": config.N_ROBOTS,
        "POSICION_INICIAL": config.POSICION_INICIAL,
        "ALPHA": config.ALPHA, "GAMMA": config.GAMMA,
        "EPSILON_START": config.EPSILON_START, "EPSILON_END": config.EPSILON_END,
        "EPSILON_DECAY": config.EPSILON_DECAY,
        "NUM_EPISODIOS": config.NUM_EPISODIOS,
        "MAX_STEPS_PER_EPISODE": config.MAX_STEPS_PER_EPISODE,
        "EPISODIOS_PARA_LOG": config.EPISODIOS_PARA_LOG,
        "GUARDAR_QTABLES_FILENAME": config.GUARDAR_QTABLES_FILENAME,
        "CARGAR_QTABLES_FILENAME": config.CARGAR_QTABLES_FILENAME, # Puede ser None
        "VENTANA_PROMEDIO_MOVIL": config.VENTANA_PROMEDIO_MOVIL,
        # Parámetros específicos de la sesión/batería con defaults para ejecución directa:
        "RESULTS_FOLDERNAME": "results_direct_run", # Carpeta diferente para no mezclar con la batería
        "EXPERIMENT_NAME": f"direct_run_{time.strftime('%Y%m%d-%H%M%S')}", # Nombre único para el trial
        "RANDOM_SEED": None, # O un entero si quieres reproducibilidad en ejecuciones directas
        "GENERAR_GRAFICO_INDIVIDUAL": config.GENERAR_GRAFICO, # Controla plots individuales
        "EVALUAR_SESION_FINAL": config.EVALUAR_AL_FINAL,       # Controla evaluación al final
        "MAX_STEPS_EVALUACION": config.MAX_STEPS_EVALUACION,
        "RENDER_EVALUACION": config.RENDER_EVALUACION,
        "PAUSA_RENDER_EVAL": config.PAUSA_RENDER_EVAL
    }

    # Crear el directorio base para la ejecución directa si no existe
    # La subcarpeta del trial se creará dentro de entrenar() -> run_training_session()
    os.makedirs(direct_run_params["RESULTS_FOLDERNAME"], exist_ok=True)

    run_training_session(direct_run_params)

    print("### Ejecución directa de train.py finalizada. ###")