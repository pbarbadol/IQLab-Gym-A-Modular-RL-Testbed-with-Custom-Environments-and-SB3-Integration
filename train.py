# train.py (modificado)

import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict, deque # Para la tasa de éxito móvil

# --- Importaciones (igual que antes) ---
try:
    from tablero import Tablero, StateType
    from q_learning_agent import (
        inicializar_q_tables, elegir_acciones, calcular_recompensas, QTableType
    )
    import config
except ImportError as e:
    print(f"Error importando módulos: {e}")
    exit()


# --- Tipo para las métricas ---
MetricsDict = dict[str, list] # Diccionario donde las claves son nombres de métricas y los valores son listas

def entrenar(env: Tablero,
             num_episodios: int,
             max_steps: int,
             alpha: float,
             gamma: float,
             epsilon_start: float,
             epsilon_end: float,
             epsilon_decay: float,
             log_interval: int = 100,
             cargar_desde: str | None = None,
             guardar_en: str | None = None
             ) -> tuple[QTableType, MetricsDict]: # Devuelve Q-tables y métricas
    """
    Entrena agentes IQL y recopila métricas detalladas.
    """
    # --- Inicialización (Q-Tables, Epsilon) ---
    if cargar_desde:
        print(f"Intentando cargar Q-Tables desde {cargar_desde}...")
        # ... (lógica de carga, similar a visualize_sim.py) ...
        # q_tables = cargar_q_tables_visual(env.num_robots, cargar_desde) # Reusar si se adaptó
        # if q_tables is None:
        q_tables = inicializar_q_tables(env.num_robots) # Fallback
    else:
        q_tables = inicializar_q_tables(env.num_robots)
    epsilon = epsilon_start

    # --- Inicialización de Métricas ---
    metrics: MetricsDict = {
        "episodio": [],
        "recompensa_promedio": [],
        "pasos_por_episodio": [],
        "epsilon": [],
        "porcentaje_visitadas": [],
        "tasa_exito_100ep": [], # Tasa de éxito en los últimos 100 episodios
        "q_table_size_total": [], # Tamaño total (suma de tamaños individuales)
        "td_error_promedio": [] # Opcional: Error TD promedio
    }
    # Buffer para calcular tasa de éxito móvil
    ultimos_100_completados = deque(maxlen=log_interval)

    print(f"--- Iniciando Entrenamiento ({num_episodios} episodios) ---")
    start_time = time.time()

    for episodio in range(num_episodios):
        estado = env.reset()
        recompensa_acumulada_episodio = np.zeros(env.num_robots)
        done = False
        paso = 0
        episodio_completado = False # Flag para saber si se completó el tablero
        td_error_sum = 0.0         # Para calcular error TD promedio
        update_count = 0           # Contador de actualizaciones Q

        # --- Bucle de un episodio ---
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
                    if not done and estado_siguiente[2][i]:
                        acciones_validas_siguiente = env.get_valid_actions(i)
                        if acciones_validas_siguiente:
                            q_valores_siguientes = q_tables[i][estado_siguiente]
                            q_validos = [q_valores_siguientes.get(a, 0.0) for a in acciones_validas_siguiente]
                            if q_validos:
                                mejor_q_siguiente = max(q_validos)

                    # Calcular TD Error ANTES de actualizar
                    delta = recompensa_recibida + gamma * mejor_q_siguiente - q_antiguo
                    td_error_sum += abs(delta)
                    update_count += 1

                    # Actualizar Q-table
                    q_tables[i][estado_previo][accion_tomada] = q_antiguo + alpha * delta
                    recompensa_acumulada_episodio[i] += recompensa_recibida

            estado = estado_siguiente
            # Comprobar si se completó el tablero en este paso (para tasa de éxito)
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

        # Calcular tamaño Q-table periódicamente (no en cada paso, puede ser lento)
        if (episodio + 1) % log_interval == 0:
             size_total = sum(len(q_t) for q_t in q_tables)
             metrics["q_table_size_total"].append(size_total)
             # Añadir NaNs para los episodios intermedios para mantener longitud de lista
             metrics["q_table_size_total"].extend([np.nan] * (log_interval - 1))
        # Asegurarse de que la lista tenga la longitud correcta al final si no es múltiplo
        if episodio == num_episodios - 1 and len(metrics["q_table_size_total"]) < num_episodios:
             size_total = sum(len(q_t) for q_t in q_tables)
             missing_count = num_episodios - len(metrics["q_table_size_total"])
             metrics["q_table_size_total"].extend([np.nan] * (missing_count - 1) + [size_total])


        # Calcular Error TD promedio
        avg_td_error = (td_error_sum / update_count) if update_count > 0 else 0.0
        metrics["td_error_promedio"].append(avg_td_error)


        # Decaimiento de Epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # --- Imprimir Log ---
        if (episodio + 1) % log_interval == 0:
            tiempo_transcurrido = time.time() - start_time
            print(f"Ep: {episodio+1}/{num_episodios} | "
                  #f"Pasos: {np.mean(metrics['pasos_por_episodio'][-log_interval:]):.1f} | " # Promedio pasos reciente
                  f"Exito: {tasa_exito_actual*100:.1f}% | "
                  f"Visit: {np.mean(metrics['porcentaje_visitadas'][-log_interval:]):.1f}% | "
                  f"Eps: {epsilon:.4f} | Rec: {np.mean(metrics['recompensa_promedio'][-log_interval:]):.3f} | "
                  #f"QSize: {size_total} | " # Mostrar tamaño si se calculó
                  f"T: {tiempo_transcurrido:.1f}s")

    # --- Fin del Entrenamiento ---
    # Asegurarse de que todas las listas de métricas tengan la misma longitud
    final_len = num_episodios
    for key in metrics:
        # Rellenar tamaño de Q-table si es necesario
        if key == "q_table_size_total":
             last_valid_size = metrics[key][-1] if metrics[key] else 0
             while len(metrics[key]) < final_len:
                  metrics[key].append(last_valid_size) # Propagar último tamaño conocido
        # Verificar otras listas (no debería ser necesario con este código)
        # elif len(metrics[key]) < final_len:
        #      print(f"Advertencia: Métrica '{key}' tiene longitud {len(metrics[key])}, esperaba {final_len}")
        #      # Rellenar con NaN o último valor si es apropiado
        #      metrics[key].extend([np.nan] * (final_len - len(metrics[key])))


    tiempo_total = time.time() - start_time
    print(f"--- Entrenamiento Finalizado en {tiempo_total:.2f} segundos ---")

    # --- Guardado (igual que antes) ---
    if guardar_en:
         try:
             with open(guardar_en, 'wb') as f:
                 q_tables_serializable = [{k: dict(v) for k, v in q_t.items()} for q_t in q_tables]
                 pickle.dump(q_tables_serializable, f)
             print(f"Q-Tables guardadas en {guardar_en}")
         except Exception as e:
             print(f"Error al guardar Q-Tables con pickle: {e}")

    return q_tables, metrics


# --- Bloque Principal (modificado para graficar métricas) ---
if __name__ == "__main__":
    entorno = Tablero(filas=config.FILAS, columnas=config.COLUMNAS,
                      num_robots=config.N_ROBOTS, posicion_inicial=config.POSICION_INICIAL)

    q_tables_finales, metrics_data = entrenar(
        env=entorno,
        num_episodios=config.NUM_EPISODIOS,
        max_steps=config.MAX_STEPS_PER_EPISODE,
        alpha=config.ALPHA, gamma=config.GAMMA,
        epsilon_start=config.EPSILON_START, epsilon_end=config.EPSILON_END,
        epsilon_decay=config.EPSILON_DECAY,
        log_interval=config.EPISODIOS_PARA_LOG,
        guardar_en=config.GUARDAR_QTABLES_FILENAME
    )

    # --- Graficar Métricas ---
    if config.GENERAR_GRAFICO:
        num_plots = 5 # Número de métricas a graficar
        fig, axs = plt.subplots(num_plots, 1, figsize=(10, 15), sharex=True) # Compartir eje X

        episodes = metrics_data['episodio']
        ventana = config.VENTANA_PROMEDIO_MOVIL

        # Función auxiliar para promedio móvil
        def moving_average(data, w):
            if len(data) < w: return None # No calcular si no hay suficientes datos
            return np.convolve(data, np.ones(w)/w, mode='valid')

        # 1. Recompensa Promedio
        axs[0].plot(episodes, metrics_data['recompensa_promedio'], alpha=0.5, label='Recompensa Prom.')
        avg = moving_average(metrics_data['recompensa_promedio'], ventana)
        if avg is not None: axs[0].plot(episodes[ventana-1:], avg, color='red', label=f'Prom. Móvil {ventana}ep')
        axs[0].set_ylabel('Recompensa Prom.')
        axs[0].legend()
        axs[0].grid(True)

        # 2. Pasos por Episodio
        axs[1].plot(episodes, metrics_data['pasos_por_episodio'], alpha=0.5, label='Pasos')
        avg = moving_average(metrics_data['pasos_por_episodio'], ventana)
        if avg is not None: axs[1].plot(episodes[ventana-1:], avg, color='red', label=f'Prom. Móvil {ventana}ep')
        axs[1].set_ylabel('Pasos')
        axs[1].legend()
        axs[1].grid(True)

        # 3. Porcentaje Visitadas
        axs[2].plot(episodes, metrics_data['porcentaje_visitadas'], alpha=0.5, label='% Visitadas')
        avg = moving_average(metrics_data['porcentaje_visitadas'], ventana)
        if avg is not None: axs[2].plot(episodes[ventana-1:], avg, color='red', label=f'Prom. Móvil {ventana}ep')
        axs[2].set_ylabel('% Visitadas')
        axs[2].set_ylim(0, 105) # Eje Y de 0 a 100+
        axs[2].legend()
        axs[2].grid(True)

        # 4. Tasa de Éxito
        axs[3].plot(episodes, metrics_data['tasa_exito_100ep'], label=f'Tasa Éxito ({config.EPISODIOS_PARA_LOG}ep)')
        axs[3].set_ylabel('Tasa Éxito (%)')
        axs[3].set_ylim(-0.05, 1.05) # Eje Y de 0 a 1
        axs[3].legend()
        axs[3].grid(True)

        # 5. Epsilon
        axs[4].plot(episodes, metrics_data['epsilon'], label='Epsilon')
        axs[4].set_ylabel('Epsilon')
        axs[4].set_xlabel('Episodio')
        axs[4].legend()
        axs[4].grid(True)

        # # 6. Opcional: Tamaño Q-Table (puede necesitar eje Y separado si escala mucho)
        # ax6 = axs[4].twinx() # Eje Y secundario
        # # Interpolar NaNs para graficar línea continua donde hay datos
        # q_size_series = pd.Series(metrics_data['q_table_size_total']).interpolate(method='linear')
        # ax6.plot(episodes, q_size_series, label='Tamaño Q-Table (Total)', color='purple', linestyle=':')
        # ax6.set_ylabel('Tamaño Q-Table', color='purple')
        # ax6.tick_params(axis='y', labelcolor='purple')
        # ax6.legend(loc='lower right')

        fig.suptitle('Métricas de Entrenamiento Q-Learning', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Ajustar para título
        plt.savefig("training_metrics.png")
        print("\nGráfico de métricas guardado como 'training_metrics.png'")
        # plt.show()

    # --- Evaluación (igual que antes) ---
    if config.EVALUAR_AL_FINAL and q_tables_finales:
        # Reconstruir Q-tables a defaultdict si se cargaron desde pickle
        q_tables_eval = inicializar_q_tables(entorno.num_robots)
        if isinstance(q_tables_finales[0], dict): # Chequea si son dicts normales
             print("Convirtiendo Q-tables cargadas a defaultdict para evaluación...")
             for i in range(entorno.num_robots):
                 for state_tuple, action_values_dict in q_tables_finales[i].items():
                     action_defaultdict = defaultdict(float, action_values_dict)
                     q_tables_eval[i][state_tuple] = action_defaultdict
        else: # Asume que ya son defaultdicts (si no se cargaron)
             q_tables_eval = q_tables_finales

        # Importar función evaluar_politica si la moviste a otro lado, o copiarla aquí
        try:
             # Intenta importar si la pusiste en utils o similar
             from q_learning_agent import evaluar_politica
        except ImportError:
            # Copia la función evaluar_politica aquí si no está importable
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
        # Fin de la copia de evaluar_politica (si fue necesaria)

        evaluar_politica(
            env=entorno,
            q_tables=q_tables_eval, # Usar las Q-tables potencialmente convertidas
            max_steps=config.MAX_STEPS_EVALUACION,
            render=config.RENDER_EVALUACION,
            pause=config.PAUSA_RENDER_EVAL
        )