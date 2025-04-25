import random
import pickle
import time
import numpy as np
from collections import defaultdict
from typing import List


try:
    from tablero import Tablero, StateType, ACTION_TERMINATE_INDEX, NUM_ACTIONS
except ImportError:
    print("Error: No se pudo importar el módulo 'tablero'. Asegúrate de que el archivo 'tablero.py' esté en la misma carpeta que este script.")
    raise

# RECOMPENSAS Y PENALIZACIONES
STEP_PENALTY = -10          # Pequeña penalización por cada paso para fomentar eficiencia.
NEW_CELL_REWARD = 8.0         # Recompensa por descubrir una nueva celda (compartida).
WALL_COLLISION_PENALTY = -0.5 # Penalización por intentar moverse a una celda inválida.
TERMINATE_EARLY_PENALTY = 0.0 # Penalización si un robot termina ANTES de completar el tablero.
TERMINATE_LATE_REWARD = 0.0   # Pequeña recompensa si termina DESPUÉS de completar (opcional).
GOAL_COMPLETED_REWARD = 40.0  # Recompensa grande al completar el tablero (para los activos).
FAILURE_PENALTY = -100.0       # Penalización si el episodio termina sin completar (todos terminan o max_steps).


QTableType = List[defaultdict]

def inicializar_q_tables(num_robots: int) -> QTableType:
    """Inicializa la tabla Q para cada robot."""

    # Cada robot tiene su propia Q-table, que es un defaultdict de defaultdicts
    # El primer defaultdict tiene como claves los estados (tuplas) y como valores otro defaultdict
    # que tiene como claves las acciones (índices) y como valores los Q-valores (float).
    # Esto permite manejar estados y acciones que no han sido visitados aún sin lanzar KeyError.
    return [defaultdict(lambda: defaultdict(float)) for _ in range(num_robots)]

def elegir_acciones(q_tables: QTableType, estado: StateType, epsilon: float, env: Tablero) -> List[int]:
    """
    Elige acciones para cada robot usando la política epsilon-greedy.
    Los robots inactivos siempre devuelven la acción TERMINATE
    
    """

    num_robots = env.num_robots
    robots_activos = estado[2] # Tupla de robos activos
    acciones_elegidas = [ACTION_TERMINATE_INDEX] * num_robots

    for i in range(num_robots):
        if not robots_activos[i]:
            continue

        # Obtenemos las acciones válidas
        acciones_validas = env.get_valid_actions(i)

        if not acciones_validas:
            acciones_elegidas[i] = ACTION_TERMINATE_INDEX
            continue

        # Epsilon-greedy: exploración vs explotación
        if random.random() < epsilon:
            # Exploración: elige una acción aleatoria de las válidas
            acciones_elegidas[i] = random.choice(acciones_validas)
        else:
            # Explotación: elige la acción con el valor Q más alto
            q_valores_estado_actual = q_tables[i][estado]
            # Filtramos las acciones válidas para obtener sus valores Q que aun no tengan una entrada en la Q-table
            q_valores_validos = {accion: q_valores_estado_actual.get(accion, 0.0) for accion in acciones_validas}

            # Elegimos la acción con el valor Q más alto
            if not q_valores_validos:
                # Si (inesperadamente) no hay valores para acciones válidas, exploramos
                acciones_elegidas[i] = random.choice(acciones_validas)
            else:
                max_q_valor = max(q_valores_validos.values())
                # Obtiene todas las acciones que tienen ese máximo Q-valor (manejo de empates)
                mejores_acciones = [accion for accion, q in q_valores_validos.items() if q == max_q_valor]
                # Elige una acción al azar entre las mejores (si hay empate)
                acciones_elegidas[i] = random.choice(mejores_acciones)

    return acciones_elegidas


def calcular_recompensas(estado_previo: StateType,
                         estado_actual: StateType,
                         acciones: List[int],
                         done: bool, # Indica si el episodio TERMINÓ en este paso
                         env: Tablero) -> List[float]:
    """
    Calcula la recompensa para CADA robot basada en la transición de estado.
    Esta función define el objetivo del aprendizaje.

    Args:
        estado_previo: El estado completo ANTES de ejecutar las acciones.
        estado_actual: El estado completo DESPUÉS de ejecutar las acciones.
        acciones: La lista de acciones ejecutadas por cada robot.
        done: True si el episodio ha finalizado en este paso (por cualquier razón).
        env: La instancia del entorno 'Tablero'.

    Returns:
        Una lista de recompensas (float), una para cada robot.
    """
    num_robots = env.num_robots
    recompensas = [0.0] * num_robots

    posiciones_previas, visitadas_previas, activos_previos = estado_previo
    posiciones_actuales, visitadas_actuales, activos_actuales = estado_actual

    # --- Variables útiles para la lógica de recompensa ---
    celdas_nuevas_descubiertas_count = len(visitadas_actuales) - len(visitadas_previas)
    completado_ahora = len(visitadas_actuales) == env.total_celdas

    for i in range(num_robots):
        # Solo calcular recompensa si el robot estaba activo ANTES de este paso
        if not activos_previos[i]:
            continue # Sin recompensa ni penalización si ya estaba inactivo

        accion_tomada = acciones[i]
        pos_prev = posiciones_previas[i]
        pos_actual = posiciones_actuales[i]
        acaba_de_terminar = not activos_actuales[i] # True si terminó en ESTE paso

        # 1. Penalización base por paso
        recompensas[i] += STEP_PENALTY

        # 2. Recompensa por descubrir celdas NUEVAS (compartida)
        #    Se da si *cualquier* robot descubrió algo nuevo.
        if celdas_nuevas_descubiertas_count > 0:
            recompensas[i] += NEW_CELL_REWARD * celdas_nuevas_descubiertas_count # Escalar por nº celdas?


        # 3. Penalización por colisión con pared/borde
        #    (Se detecta si intentó moverse pero sigue en la misma celda y no terminó)
        if pos_prev == pos_actual and accion_tomada != ACTION_TERMINATE_INDEX and not acaba_de_terminar:
             recompensas[i] += WALL_COLLISION_PENALTY

        # 4. Penalización/Recompensa relacionada con 'Terminar'
        if acaba_de_terminar:
            if completado_ahora:
                 # Terminó justo cuando se completó o después
                 recompensas[i] += TERMINATE_LATE_REWARD
            else:
                 # Terminó antes de que se completara el tablero
                 recompensas[i] += TERMINATE_EARLY_PENALTY

        # 5. Recompensa/Penalización GRANDE al final del episodio
        if done: # Si el episodio termina en este paso
            if completado_ahora:
                 # Éxito: El tablero se completó
                 recompensas[i] += GOAL_COMPLETED_REWARD
            else:
                 # Fracaso: Terminó (max_steps o todos inactivos) sin completar
                 recompensas[i] += FAILURE_PENALTY * num_robots # Penaliza a todos los robots por igual (Realmente penaliza uno pero el promedio es el mismo)

    return recompensas



def guardar_q_tables(q_tables: QTableType, filename: str = "q_tables.pkl"):
    """Guarda la lista de Q-tables en un archivo."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump([dict(q_table) for q_table in q_tables], f) # Convertir a dict normal
        print(f"Q-Tables guardadas en {filename}")
    except Exception as e:
        print(f"Error al guardar Q-Tables: {e}")

def cargar_q_tables(num_robots: int, filename: str = "q_tables.pkl") -> QTableType:
    """Carga la lista de Q-tables desde un archivo."""
    try:
        with open(filename, 'rb') as f:
            list_of_dicts = pickle.load(f)
            # Convertir de nuevo a defaultdict anidado
            q_tables = inicializar_q_tables(num_robots)
            for i in range(num_robots):
                if i < len(list_of_dicts):
                    loaded_dict = list_of_dicts[i]
                    for state, action_values in loaded_dict.items():
                        q_tables[i][state] = defaultdict(float, action_values)
            print(f"Q-Tables cargadas desde {filename}")
            return q_tables
    except FileNotFoundError:
        print(f"Advertencia: Archivo {filename} no encontrado. Inicializando Q-tables vacías.")
        return inicializar_q_tables(num_robots)
    except Exception as e:
        print(f"Error al cargar Q-Tables: {e}. Inicializando Q-tables vacías.")
        return inicializar_q_tables(num_robots)
    
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