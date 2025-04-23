# train.py

import time
import numpy as np
import matplotlib.pyplot as plt

# --- Importar desde módulos ---
try:
    from tablero import Tablero, StateType # Importa la clase y el tipo de estado
    from q_learning_agent import (
        inicializar_q_tables,
        elegir_acciones,
        calcular_recompensas,
        guardar_q_tables,
        cargar_q_tables,
        QTableType # Importa el tipo para anotaciones
    )
    # Importar configuración
    import config
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de que 'tablero.py', 'q_learning_agent.py' y 'config.py' estén en el mismo directorio o accesibles.")
    exit()


def entrenar(env: Tablero,
             num_episodios: int,
             max_steps: int,
             alpha: float,
             gamma: float,
             epsilon_start: float,
             epsilon_end: float,
             epsilon_decay: float,
             log_interval: int = 100,
             cargar_desde: str | None = None, # Ruta para cargar Q-tables pre-entrenadas
             guardar_en: str | None = None   # Ruta para guardar Q-tables al final
             ) -> tuple[QTableType, list[float]]:
    """
    Función principal para entrenar los agentes usando Independent Q-Learning (IQL).

    Args:
        env: Instancia del entorno Tablero.
        num_episodios: Número total de episodios a ejecutar.
        max_steps: Máximo número de pasos permitidos por episodio.
        alpha: Tasa de aprendizaje.
        gamma: Factor de descuento.
        epsilon_start: Valor inicial de epsilon.
        epsilon_end: Valor final (mínimo) de epsilon.
        epsilon_decay: Factor de decaimiento de epsilon por episodio.
        log_interval: Cada cuántos episodios imprimir información.
        cargar_desde: Nombre del archivo .pkl desde donde cargar Q-tables existentes (opcional).
        guardar_en: Nombre del archivo .pkl donde guardar las Q-tables finales (opcional).


    Returns:
        Una tupla conteniendo:
        - La lista de Q-tables aprendidas (una por robot).
        - Una lista con la recompensa promedio por episodio durante el entrenamiento.
    """

    if cargar_desde:
        # Intentar cargar Q-tables desde el archivo especificado
        q_tables = cargar_q_tables(env.num_robots, cargar_desde)
        print(f"Funcionalidad de carga no implementada/descomentada. Iniciando Q-tables vacías.")
        q_tables = inicializar_q_tables(env.num_robots)
    else:
        q_tables = inicializar_q_tables(env.num_robots)

    epsilon = epsilon_start
    historico_recompensas_episodio = [] # Guarda recompensa promedio de cada episodio

    print(f"--- Iniciando Entrenamiento ({num_episodios} episodios) ---")
    start_time = time.time()

    for episodio in range(num_episodios):
        estado = env.reset() # Obtiene estado inicial ((pos), frozenset(visit), (activos))
        recompensa_acumulada_episodio = np.zeros(env.num_robots) # Usar numpy array es eficiente
        done = False
        paso = 0

        # Bucle principal de un episodio
        while not done and paso < max_steps:
            paso += 1

            # 1. Elegir acciones para todos los robots activos
            acciones = elegir_acciones(q_tables, estado, epsilon, env)

            # Guardar estado previo antes de ejecutar el paso
            estado_previo = estado

            # 2. Ejecutar las acciones en el entorno
            estado_siguiente, done = env.step(acciones)

            # 3. Calcular las recompensas individuales
            recompensas = calcular_recompensas(estado_previo, estado_siguiente, acciones, done, env)

            # 4. Actualizar las Q-tables para cada robot que estaba activo ANTES del paso
            for i in range(env.num_robots):
                # Solo actualizar si el robot 'i' estaba activo en el estado PREVIO
                if estado_previo[2][i]:
                    accion_tomada = acciones[i]
                    recompensa_recibida = recompensas[i]

                    # Valor Q antiguo
                    q_antiguo = q_tables[i][estado_previo].get(accion_tomada, 0.0) # Usar .get

                    # Calcular el mejor valor Q para el estado siguiente (max_a' Q(s', a'))
                    # Es 0 si el episodio terminó (done=True) o si el robot 'i' ya no está activo en s'
                    mejor_q_siguiente = 0.0
                    if not done and estado_siguiente[2][i]:
                        acciones_validas_siguiente = env.get_valid_actions(i)
                        if acciones_validas_siguiente:
                            q_valores_siguientes = q_tables[i][estado_siguiente]
                            # Obtener Q-valores para acciones válidas, default a 0
                            q_validos = [q_valores_siguientes.get(a, 0.0) for a in acciones_validas_siguiente]
                            if q_validos:
                                mejor_q_siguiente = max(q_validos)

                    # Fórmula de actualización Q-Learning
                    delta = recompensa_recibida + gamma * mejor_q_siguiente - q_antiguo
                    q_tables[i][estado_previo][accion_tomada] = q_antiguo + alpha * delta

                    # Acumular recompensa para estadísticas del episodio
                    recompensa_acumulada_episodio[i] += recompensa_recibida

            # Actualizar estado para el siguiente paso
            estado = estado_siguiente

        # --- Fin del episodio ---
        recompensa_promedio_episodio = np.mean(recompensa_acumulada_episodio)
        historico_recompensas_episodio.append(recompensa_promedio_episodio)

        # Decaimiento de Epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Imprimir progreso
        if (episodio + 1) % log_interval == 0:
            tiempo_transcurrido = time.time() - start_time
            tiempo_estimado_restante = (tiempo_transcurrido / (episodio + 1)) * (num_episodios - (episodio + 1))
            media_reciente = np.mean(historico_recompensas_episodio[-log_interval:])
            print(f"Ep: {episodio+1}/{num_episodios} | "
                  f"Pasos: {paso} | Done: {done} | "
                  f"Eps: {epsilon:.4f} | Rec Reciente: {media_reciente:.3f} | "
                  f"T: {tiempo_transcurrido:.1f}s / ~{(tiempo_transcurrido+tiempo_estimado_restante)/60:.1f}m")

    # --- Fin del Entrenamiento ---
    tiempo_total = time.time() - start_time
    print(f"--- Entrenamiento Finalizado en {tiempo_total:.2f} segundos ---")

    # Guardar Q-Tables
    if guardar_en:
         guardar_q_tables(q_tables, guardar_en)

    return q_tables, historico_recompensas_episodio

def evaluar_politica(env: Tablero,
                       q_tables: QTableType,
                       max_steps: int,
                       render: bool = True,
                       pause: float = 0.3):
    """Ejecuta un episodio usando la política aprendida (epsilon=0)."""
    print("\n--- Evaluación de la Política Aprendida (Epsilon = 0) ---")
    estado = env.reset()
    if render:
        env.render()
        time.sleep(pause * 2) # Pausa inicial

    done = False
    paso = 0
    recompensa_total_eval = np.zeros(env.num_robots)

    while not done and paso < max_steps:
        paso += 1
        # Elegir acciones determinísticamente (epsilon=0)
        acciones = elegir_acciones(q_tables, estado, 0.0, env)

        # Ejecutar paso
        estado_siguiente, done = env.step(acciones)

        # Calcular recompensas solo para info (no se usan para aprender)
        recompensas = calcular_recompensas(estado, estado_siguiente, acciones, done, env)
        recompensa_total_eval += recompensas

        # Actualizar estado
        estado = estado_siguiente

        # Renderizar
        if render:
            print(f"\nPaso {paso} | Acciones: {acciones} | Recompensas: [{', '.join(f'{r:.2f}' for r in recompensas)}]")
            env.render()
            if pause > 0:
                time.sleep(pause)

    print("\n--- Fin Evaluación ---")
    print(f"Episodio terminado en {paso} pasos. Done={done}")
    print(f"Recompensa total acumulada en evaluación: {recompensa_total_eval} (Promedio: {np.mean(recompensa_total_eval):.2f})")
    print(f"Celdas visitadas: {len(estado[1])}/{env.total_celdas}")


# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    # Crear el entorno usando la configuración
    entorno = Tablero(filas=config.FILAS,
                      columnas=config.COLUMNAS,
                      num_robots=config.N_ROBOTS,
                      posicion_inicial=config.POSICION_INICIAL)

    # Entrenar a los agentes
    q_tables_finales, historial_recompensas = entrenar(
        env=entorno,
        num_episodios=config.NUM_EPISODIOS,
        max_steps=config.MAX_STEPS_PER_EPISODE,
        alpha=config.ALPHA,
        gamma=config.GAMMA,
        epsilon_start=config.EPSILON_START,
        epsilon_end=config.EPSILON_END,
        epsilon_decay=config.EPSILON_DECAY,
        log_interval=config.EPISODIOS_PARA_LOG,
        # cargar_desde=config.CARGAR_QTABLES_FILENAME, # Descomentar para cargar
        guardar_en=config.GUARDAR_QTABLES_FILENAME     # Guardará al final
    )

    # Generar gráfico si está habilitado
    if config.GENERAR_GRAFICO:
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(historial_recompensas, alpha=0.6, label='Recompensa Promedio por Episodio')
            # Calcular y graficar promedio móvil
            if len(historial_recompensas) >= config.VENTANA_PROMEDIO_MOVIL:
                promedio_movil = np.convolve(historial_recompensas,
                                             np.ones(config.VENTANA_PROMEDIO_MOVIL)/config.VENTANA_PROMEDIO_MOVIL,
                                             mode='valid')
                plt.plot(np.arange(len(promedio_movil)) + config.VENTANA_PROMEDIO_MOVIL - 1,
                         promedio_movil,
                         label=f'Promedio Móvil ({config.VENTANA_PROMEDIO_MOVIL} ep)',
                         color='red', linewidth=2)

            plt.xlabel('Episodio')
            plt.ylabel('Recompensa Promedio')
            plt.title('Progreso del Entrenamiento Q-Learning')
            plt.legend()
            plt.grid(True)
            plt.ylim(bottom=min(historial_recompensas) - abs(min(historial_recompensas)*0.1) if historial_recompensas else -1) # Ajustar eje Y
            plt.tight_layout()
            plt.savefig("training_rewards.png") # Guardar gráfico
            print("\nGráfico de recompensas guardado como 'training_rewards.png'")
            # plt.show() # Descomentar para mostrar interactivamente
        except Exception as e:
            print(f"\nError al generar/guardar el gráfico: {e}")
            print("Asegúrate de tener matplotlib instalado: pip install matplotlib numpy")

    # Evaluar la política aprendida si está habilitado
    if config.EVALUAR_AL_FINAL and q_tables_finales:
         # Necesitamos convertir las Q-tables guardadas (dicts) de nuevo a defaultdict si las cargamos
         # Si no las cargamos, ya son defaultdicts. Asumimos que son defaultdicts aquí.
         evaluar_politica(
             env=entorno,
             q_tables=q_tables_finales,
             max_steps=config.MAX_STEPS_EVALUACION,
             render=config.RENDER_EVALUACION,
             pause=config.PAUSA_RENDER_EVAL
         )