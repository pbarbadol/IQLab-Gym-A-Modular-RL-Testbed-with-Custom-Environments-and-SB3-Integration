# -*- coding: utf-8 -*-
# Author: Pablo Barbado Lozano
# Date: 2025-04-23
# Description: Entorno Gymnasium para el problema de cobertura de un tablero por múltiples robots.
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple, Set, Optional, Dict, Any

# --- Constantes del Entorno ---

# Mapeo de acciones: (cambio_fila, cambio_columna)
# Se mantiene igual, pero será usado internamente.
_ACTION_MAP = {
    0: (-1, 0),  # Arriba
    1: (1, 0),   # Abajo
    2: (0, -1),  # Izquierda
    3: (0, 1),   # Derecha
    4: (0, 0)    # Terminar (acción interna para un robot)
}
_ACTION_TERMINATE_INDEX = 4 # Índice específico para terminar
_NUM_ATOMIC_ACTIONS = len(_ACTION_MAP) # Número de acciones básicas por robot

# Tipos para anotaciones (Opcional pero útil)
ObsType = Dict[str, np.ndarray]
InfoType = Dict[str, Any]


"""
La lógica del entorno (TableroEnv) debe centrarse en describir cómo funciona el mundo
para Gymnasium: cómo cambian las posiciones, qué celdas se visitan, cuándo termina
un episodio, y qué recompensa se obtiene.
"""
class TableroEnv(gym.Env):
    """
    Entorno Gymnasium para el problema de cobertura de un tablero por múltiples robots.

    El objetivo es que los robots visiten todas las celdas del tablero.

    **Espacio de Observación (`observation_space`):**
    Un diccionario (`spaces.Dict`) con:
        - `robot_positions`: `spaces.Tuple` de N `spaces.Tuple(spaces.Discrete, spaces.Discrete)`.
                           Posición (fila, columna) de cada uno de los N robots.
        - `visited_mask`: `spaces.MultiBinary((filas, columnas))`. Una matriz binaria donde
                          1 indica que la celda ha sido visitada, 0 si no.
        - `active_robots`: `spaces.MultiBinary(N)`. Un vector binario donde 1 indica que
                           el robot i está activo, 0 si ha terminado.

    **Espacio de Acción (`action_space`):**
    `spaces.MultiDiscrete([_NUM_ATOMIC_ACTIONS] * N)`. Un vector donde cada elemento es la
    acción (0-4) para el robot correspondiente.
        - 0: Arriba
        - 1: Abajo
        - 2: Izquierda
        - 3: Derecha
        - 4: Terminar (el robot deja de moverse)

    **Recompensa (`reward`):**
        - +1 por cada *nueva* celda visitada en el paso actual.
        - -0.01 por cada robot activo que *no* elige la acción 'Terminar' (pequeño coste por movimiento).
        - +`filas * columnas` de bonus si todas las celdas son visitadas.
        - -1 si se realiza una acción inválida (intentar moverse fuera del tablero).

    **Terminación (`terminated`):**
        - El episodio termina si todas las celdas del tablero han sido visitadas.
        - El episodio termina si *todos* los robots han elegido la acción 'Terminar'.

    **Truncación (`truncated`):**
        - El episodio se trunca si se alcanza un número máximo de pasos (opcional, se puede añadir).

    **Información (`info`):**
        - `visited_cells_set`: El conjunto (set) de tuplas (fila, col) de celdas visitadas.
        - `valid_actions`: Una lista de listas, donde `valid_actions[i]` contiene los índices
                           de acciones válidas para el robot `i`. (Nota: SB3 por defecto
                           no usa máscaras de acción, pero es útil tener la info).
    """
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, filas: int, columnas: int, num_robots: int,
                 posicion_inicial: Optional[List[Tuple[int, int]]] = None,
                 max_steps: Optional[int] = None,
                 render_mode: Optional[str] = None):

        super().__init__()

        # --- Validación de Parámetros ---
        if not (filas > 0 and columnas > 0):
            raise ValueError("El número de filas y columnas debe ser mayor que 0.")
        if not (num_robots > 0 and num_robots <= filas * columnas):
            raise ValueError("El número de robots debe ser mayor que 0 y menor o igual al número de celdas del tablero.")

        # --- Atributos del Entorno ---
        self.filas = filas
        self.columnas = columnas
        self.num_robots = num_robots
        self.total_celdas = filas * columnas
        self.max_steps = max_steps
        self._current_step = 0

        # --- Posiciones Iniciales ---
        if posicion_inicial:
            if len(posicion_inicial) != num_robots:
                raise ValueError("La longitud de posicion_inicial debe coincidir con num_robots.")
            if any(not self._es_posicion_valida_static(r, c, filas, columnas) for r, c in posicion_inicial):
                 raise ValueError("Alguna posición inicial proporcionada está fuera del tablero.")
            self._posicion_inicial_robots: List[Tuple[int, int]] = list(posicion_inicial)
        else:
            # Todos empiezan en (0,0) por defecto si es válido
            if not self._es_posicion_valida_static(0, 0, filas, columnas) and filas*columnas > 0:
                 raise RuntimeError("La posición inicial (0,0) es inválida con las dimensiones dadas.")
            self._posicion_inicial_robots = [(0, 0)] * num_robots

        self.action_space = spaces.MultiDiscrete([_NUM_ATOMIC_ACTIONS] * self.num_robots)
        low_bounds = np.zeros((self.num_robots, 2), dtype=np.int32)
        # high_bounds será una matriz donde cada fila es [filas-1, columnas-1]
        high_bounds_per_robot = np.array([self.filas - 1, self.columnas - 1], dtype=np.int32)
        high_bounds = np.tile(high_bounds_per_robot, (self.num_robots, 1)) # Repetir para cada robot
        # --- Definición de Espacios Gymnasium ---
        

        self.observation_space = spaces.Dict({
        "robot_positions": spaces.Box(
            low=low_bounds,    # Ahora tiene shape (num_robots, 2)
            high=high_bounds,  # Ahora tiene shape (num_robots, 2)
            shape=(self.num_robots, 2), # Coincide con low/high
            dtype=np.int32
        ),
        "visited_mask": spaces.MultiBinary((self.filas, self.columnas)),
        "active_robots": spaces.MultiBinary(self.num_robots)
    })

        # --- Estado Interno (Privado) ---
        # Usamos guiones bajos para indicar que son variables internas del entorno
        self._posicion_robots: List[Tuple[int, int]] = []
        self._celdas_visitadas: Set[Tuple[int, int]] = set()
        self._robots_activos: List[bool] = [] # True si el robot aún puede moverse

        # --- Renderización ---
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


    @staticmethod
    def _es_posicion_valida_static(fila: int, columna: int, filas_max: int, columnas_max: int) -> bool:
        """Método estático para verificar validez (útil en __init__ antes de self)."""
        return 0 <= fila < filas_max and 0 <= columna < columnas_max

    def _es_posicion_valida(self, fila: int, columna: int) -> bool:
        """Verifica si la posición (fila, columna) es válida dentro del tablero."""
        return 0 <= fila < self.filas and 0 <= columna < self.columnas

    # EN TableroEnv._get_obs

    def _get_obs(self) -> ObsType:
        """Construye la observación en el formato de `observation_space`."""
        visited_mask = np.zeros((self.filas, self.columnas), dtype=np.int8)
        for r, c in self._celdas_visitadas:
            if self._es_posicion_valida(r, c):
                visited_mask[r, c] = 1

        # Convertir lista de tuplas a array NumPy del tipo correcto
        robot_positions_array = np.array(self._posicion_robots, dtype=np.int32)

        return {
            "robot_positions": robot_positions_array, # Ahora es un array NumPy
            "visited_mask": visited_mask,
            "active_robots": np.array(self._robots_activos, dtype=np.int8)
        }

    def _get_info(self) -> InfoType:
        """Genera información adicional sobre el estado actual."""
        return {
            "visited_cells_set": self._celdas_visitadas.copy(), # Copia para evitar modificación externa
            "valid_actions": [self._get_valid_actions_for_robot(i) for i in range(self.num_robots)]
        }

    def _get_valid_actions_for_robot(self, robot_id: int) -> List[int]:
        """Obtiene las acciones atómicas válidas para un robot específico."""
        if not self._robots_activos[robot_id]:
            # Si ya terminó, solo puede seguir "terminando" (o no hacer nada útil)
            # Devolver solo la acción de terminar asegura consistencia.
            return [_ACTION_TERMINATE_INDEX]

        valid = []
        fila_actual, columna_actual = self._posicion_robots[robot_id]

        # Comprobamos movimientos (ARRIBA, ABAJO, IZQUIERDA, DERECHA)
        for action_index, (df, dc) in _ACTION_MAP.items():
            if action_index == _ACTION_TERMINATE_INDEX:
                continue # Saltar la acción de terminar aquí, se añade al final
            fila_nueva, columna_nueva = fila_actual + df, columna_actual + dc
            if self._es_posicion_valida(fila_nueva, columna_nueva):
                valid.append(action_index)

        # Añadir siempre la opción de terminar si el robot está activo
        valid.append(_ACTION_TERMINATE_INDEX)

        return sorted(valid) # Ordenar por consistencia

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, InfoType]:
        """Reinicia el entorno a su estado inicial."""
        super().reset(seed=seed) # Necesario para manejar el generador de números aleatorios (RNG)

        # Reiniciamos el estado interno
        self._posicion_robots = list(self._posicion_inicial_robots)
        self._celdas_visitadas = set(self._posicion_robots) # Celdas iniciales ya están visitadas
        self._robots_activos = [True] * self.num_robots # Todos los robots empiezan activos
        self._current_step = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[ObsType, float, bool, bool, InfoType]:
        """
        Realiza un paso en el entorno usando la acción multi-agente proporcionada.

        Args:
            action: Un array de numpy o tupla con la acción para cada robot.

        Returns:
            observation: El estado del entorno después del paso.
            reward: La recompensa obtenida en este paso.
            terminated: True si el episodio ha terminado (objetivo alcanzado o todos inactivos).
            truncated: True si el episodio se ha truncado (e.g., por límite de pasos).
            info: Información adicional.
        """
        if not self.action_space.contains(action):
             raise ValueError(f"Acción inválida: {action}. Debe estar en {self.action_space}")

        # --- Lógica de Transición del Estado ---
        num_celdas_antes = len(self._celdas_visitadas)
        posicion_siguiente = list(self._posicion_robots) # Copia para cálculo
        reward = 0.0
        num_robots_moviendose = 0

        acciones_list = list(action) # Convertir a lista si es numpy array

        for i in range(self.num_robots):
            if not self._robots_activos[i]:
                continue # Este robot ya terminó, no hace nada

            action_index = acciones_list[i]
            num_robots_moviendose += 1 # Contamos como activo si no ha terminado

            # --- Si la acción es Terminar ---
            if action_index == _ACTION_TERMINATE_INDEX:
                self._robots_activos[i] = False # Marca este robot como inactivo
                # No hay penalización por elegir terminar
                continue # Pasa al siguiente robot

            # --- Si la acción es Moverse ---
            # Penalización por movimiento
            reward -= 0.01

            fila_actual, columna_actual = self._posicion_robots[i]
            try:
                fila_delta, columna_delta = _ACTION_MAP[action_index]
            except KeyError:
                # Esto no debería ocurrir si la validación de action_space funciona
                print(f"Warning: Acción {action_index} inválida recibida para robot {i}. Ignorando.")
                reward -= 1.0 # Penalización fuerte por acción teóricamente imposible
                continue

            fila_nueva = fila_actual + fila_delta
            columna_nueva = columna_actual + columna_delta

            # Comprobamos validez del movimiento
            if self._es_posicion_valida(fila_nueva, columna_nueva):
                posicion_siguiente[i] = (fila_nueva, columna_nueva)
                # La celda visitada se añade después de actualizar todas las posiciones
            else:
                # Acción inválida (chocar contra pared) - penalización
                reward -= 1.0
                # El robot se queda en su sitio (no actualizamos posicion_siguiente[i])

        # Actualizamos posiciones y celdas visitadas DESPUÉS de calcular todos los movimientos
        self._posicion_robots = posicion_siguiente
        # Añadimos TODAS las posiciones actuales de los robots activos a visitadas
        # (Incluso si no se movieron pero están en una celda)
        for i in range(self.num_robots):
             if self._robots_activos[i]: # Solo los activos cuentan para 'visitar' en este paso
                 self._celdas_visitadas.add(self._posicion_robots[i])


        # --- Cálculo de Recompensa por Nuevas Celdas ---
        num_celdas_despues = len(self._celdas_visitadas)
        nuevas_celdas_visitadas = num_celdas_despues - num_celdas_antes
        reward += float(nuevas_celdas_visitadas) # Recompensa por exploración

        # --- Comprobación de Condiciones de Fin ---
        self._current_step += 1

        completado = len(self._celdas_visitadas) == self.total_celdas
        todos_terminados = not any(self._robots_activos)

        terminated = completado or todos_terminados
        truncated = False
        if self.max_steps is not None and self._current_step >= self.max_steps:
            truncated = True
            # Podríamos añadir una penalización aquí si se trunca sin completar
            # if not completado: reward -= self.total_celdas / 2

        # --- Bonus por Completar ---
        if completado:
            reward += float(self.total_celdas) # Gran bonus por terminar el objetivo

        # --- Obtener Obs/Info Finales ---
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info


    def render(self) -> Optional[str]:
        """
        Renderiza el estado actual del entorno.
        - 'human': Imprime en la consola.
        - 'ansi': Devuelve una representación en formato string ANSI.
        """
        if self.render_mode == "ansi":
            return self._render_text()
        elif self.render_mode == "human":
            self._render_frame()
            return None # Human mode suele renderizar en una ventana o imprimir
        else:
            return None # No renderizar

    def _render_frame(self):
         print(self._render_text())


    def _render_text(self) -> str:
        """Genera la representación textual del tablero."""
        # Crear una matriz vacía para el tablero
        grid_repr = [[' ' for _ in range(self.columnas)] for _ in range(self.filas)]

        # Marcar celdas visitadas
        for r, c in self._celdas_visitadas:
            if self._es_posicion_valida(r, c):
                grid_repr[r][c] = '.' # Punto para visitada

        # Marcar posiciones de los robots (con manejo básico de superposición)
        robot_markers_en_celda: Dict[Tuple[int, int], List[str]] = {}
        for i, (r, c) in enumerate(self._posicion_robots):
            if self._es_posicion_valida(r, c):
                marker = str(i) if self._robots_activos[i] else 'x' # ID si activo, 'x' si inactivo
                if (r, c) not in robot_markers_en_celda:
                    robot_markers_en_celda[(r, c)] = []
                robot_markers_en_celda[(r, c)].append(marker)

        # Añadir marcadores de robots al grid
        for (r, c), markers in robot_markers_en_celda.items():
            if len(markers) == 1:
                grid_repr[r][c] = markers[0]
            elif len(markers) > 1:
                 # Si hay varios, mostramos '+' o el número si caben? Mejor '+'
                 grid_repr[r][c] = '+' # Símbolo para múltiples robots
            # Si está visitada '.' y hay un robot, el robot tiene prioridad

        # Construir el string del tablero
        output = "-" * (self.columnas * 2 + 1) + "\n"
        for r in range(self.filas):
            output += "|" + "|".join(grid_repr[r]) + "|\n"
        output += "-" * (self.columnas * 2 + 1) + "\n"
        output += (f"Paso: {self._current_step} | "
                   f"Visitadas: {len(self._celdas_visitadas)}/{self.total_celdas} | "
                   f"Activos: {self._robots_activos}\n")
        # Podríamos añadir las posiciones aquí también si es útil
        # output += f"Posiciones: {self._posicion_robots}\n"

        return output


    def close(self):
        """Cierra el entorno y libera recursos (si los hubiera)."""
        # En este caso, no hay recursos externos como ventanas gráficas que cerrar.
        #print("Cerrando el entorno TableroEnv.")
        pass

# --- Ejemplo de Uso y Verificación con Gymnasium ---
if __name__ == "__main__":
    print("--- Creando Entorno Gymnasium ---")
    # Ejemplo con renderizado en consola ('human') y límite de pasos
    env = TableroEnv(filas=4, columnas=5, num_robots=2, render_mode='human', max_steps=50)
    # O sin renderizado explícito (más rápido para entrenamiento)
    # env = TableroEnv(filas=4, columnas=5, num_robots=2)

    print(f"Espacio de Acción: {env.action_space}")
    print(f"Espacio de Observación: {env.observation_space}")

    # --- Verificación con check_env (muy recomendable) ---
    try:
        from gymnasium.utils.env_checker import check_env
        print("\n--- Verificando compatibilidad con Gymnasium (check_env) ---")
        check_env(env)
        print("¡Entorno compatible con Gymnasium!")
    except ImportError:
        print("Instala gymnasium[mujoco] o gymnasium[all] para usar check_env.")
    except Exception as e:
        print(f"Error en check_env: {e}")


    # --- Ejecución de un Episodio de Ejemplo ---
    print("\n--- Ejecutando un episodio de ejemplo ---")
    obs, info = env.reset()
    print("Estado inicial (observación):")
    for key, value in obs.items():
        print(f"  {key}:")
        print(value)
    print("Información inicial:")
    print(info)

    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    while not terminated and not truncated:
        step_count += 1
        # Tomar una acción aleatoria (una para cada robot)
        # En un caso real, aquí iría la política del agente (modelo de SB3)
        action = env.action_space.sample()
        print(f"\nPaso {step_count}, Acción: {action}")

        obs, reward, terminated, truncated, info = env.step(action)

        print("Nueva Observación:")
        for key, value in obs.items():
             print(f"  {key}:")
             print(value)
        print(f"Recompensa: {reward:.2f}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        # print(f"Info: {info}") # Puede ser muy verboso mostrarlo siempre

        total_reward += reward

        # Pequeña pausa para poder ver el renderizado 'human'
        if env.render_mode == 'human':
            import time
            time.sleep(0.5)


    print("\n--- Episodio Finalizado ---")
    print(f"Razón: {'Terminated' if terminated else 'Truncated'}")
    print(f"Pasos totales: {step_count}")
    print(f"Recompensa total: {total_reward:.2f}")
    print(f"Celdas visitadas al final: {len(info['visited_cells_set'])}/{env.total_celdas}")

    env.close()