# Author: Pablo Barbado Lozano
# Date: 2025-04-23
from typing import List, Tuple, Set, FrozenSet


# Mapeo de acciones:
ACTION_MAP = {
    0: (-1, 0),  # Arriba
    1: (1, 0),   # Abajo
    2: (0, -1),  # Izquierda
    3: (0, 1),    # Derecha
    4: (0, 0)    # Terminar
}
ACTION_TERMINATE_INDEX = 4 # Índice específico para terminar

NUM_ACTIONS = len(ACTION_MAP)

# Definimos el estado. Este será una tupla que contiene Tuple[int, int] (posición del robot) y un conjunto de celdas visitadas (FrozenSet[Tuple[int, int]]).
# Ponemos el tipo FrozenSet para que no se pueda modificar el conjunto de celdas visitadas una vez devuelto el estado.
StateType = Tuple[Tuple[Tuple[int, int], ...], FrozenSet[Tuple[int, int]], Tuple[bool, ...]]

"""
La lógica del entorno (Tablero) debe centrarse en describir cómo funciona el mundo: 
cómo cambian las posiciones, qué celdas se visitan, cuándo termina un episodio. 
La recompensa, en cambio, define el objetivo del agente: qué comportamiento queremos que aprenda.
"""
class Tablero:
    def __init__(self, filas, columnas, num_robots, posicion_inicial=None):

        # Control de errores de inicialización:
        if not (filas > 0 and columnas > 0):
            raise ValueError("El número de filas y columnas debe ser mayor que 0.")
        if not (num_robots > 0 and num_robots <= filas * columnas):
            raise ValueError("El número de robots debe ser mayor que 0 y menor o igual al número de celdas del tablero.")
        
        # Asignación de atributos:
        self.filas = filas
        self.columnas = columnas
        self.num_robots = num_robots
        self.total_celdas = filas * columnas

        # Definir posiciones iniciales
        if posicion_inicial:
            if len(posicion_inicial) != num_robots:
                raise ValueError("La longitud de posicion_inicial debe coincidir con num_robots.")
            # Usamos un método estático auxiliar para validar antes de que 'self' esté listo
            if any(not self._es_posicion_valida_static(r, c, filas, columnas) for r, c in posicion_inicial):
                 raise ValueError("Alguna posición inicial proporcionada está fuera del tablero.")
            self._posicion_inicial_robots = list(posicion_inicial)
        else:
            # Todos empiezan en (0,0) por defecto si es válido
            if not self._es_posicion_valida_static(0, 0, filas, columnas) and filas*columnas > 0:
                 # Esto no debería ocurrir si filas > 0 y cols > 0
                 raise RuntimeError("La posición inicial (0,0) es inválida con las dimensiones dadas.")
            self._posicion_inicial_robots = [(0, 0)] * num_robots
        
        # Obtenemos la posición de los robots:
        self._posicion_robots: List[Tuple[int, int]] = [] # Inicializamos la posición de los robots en una lista vacía
        self._celdas_visitadas: Set[Tuple[int, int]] = set() # Inicializamos las celdas visitadas en un conjunto vacío
        self._robots_activos: List[bool] = [] # Inicializamos los robots activos en una lista vacía

        # Inicializamos el tablero:
        self.reset()

    @staticmethod
    def _es_posicion_valida_static(fila: int, columna: int, filas_max: int, columnas_max: int) -> bool:
        """Método estático para verificar validez (útil en __init__ antes de self)."""
        return 0 <= fila < filas_max and 0 <= columna < columnas_max

    def es_posicion_valida(self, fila: int, columna: int) -> bool:
        """Verifica si la posición (fila, columna) es válida dentro del tablero."""
        return 0 <= fila < self.filas and 0 <= columna < self.columnas
    
    def get_estado(self)  -> StateType:
        return (tuple(self._posicion_robots), frozenset(self._celdas_visitadas), tuple(self._robots_activos))
    
    def get_valid_actions(self, robot_id: int) -> List[int]:

        if not self._robots_activos[robot_id]:
            return [ACTION_TERMINATE_INDEX] # Si ya terminó, solo puede seguir "terminando"

        valid = []
        fila_actual, columna_actual = self._posicion_robots[robot_id]

        # Comprobamos movimientos (ARRIBA, ABAJO, IZQUIERDA, DERECHA)
        for action_index, (df, dc) in ACTION_MAP.items():
            if action_index == ACTION_TERMINATE_INDEX:
                    continue # Saltar la acción de terminar aquí
            fila_nueva, columna_nueva = fila_actual + df, columna_actual + dc
            if self.es_posicion_valida(fila_nueva, columna_nueva):
                valid.append(action_index)

        valid.append(ACTION_TERMINATE_INDEX) # Añadir siempre la opción de terminar

        return valid
    

    def reset(self) -> StateType:
        """Reinicia el tablero a su estado inicial."""
        self._posicion_robots = list(self._posicion_inicial_robots) # Reiniciamos la posición de los robots a la inicial
        self._celdas_visitadas = set(self._posicion_robots)
        self._robots_activos = [True] * self.num_robots
        return self.get_estado() # Devolvemos el estado inicial del tablero


    def step(self, acciones: List[int]) -> Tuple[StateType, bool]:
        """Realiza un paso en el tablero según las acciones de los robots."""
        # Verificamos que el número de acciones sea igual al número de robots:
        if len(acciones) != self.num_robots:
            raise ValueError("El número de acciones debe ser igual al número de robots.")

        # Copia para no modificar directamente mientras se itera
        posicion_siguiente = list(self._posicion_robots)

        # Iteramos sobre cada robot y su acción correspondiente:
        for i in range(self.num_robots):
            if not self._robots_activos[i]:
                continue # Este robot ya terminó, pasa al siguiente

            action_index = acciones[i]
            if action_index not in ACTION_MAP:
                raise ValueError(f"Acción inválida: {action_index}. Debe ser un número entre 0 y {NUM_ACTIONS - 1}.")
            
            # --- Si la acción es Terminar ---
            if action_index == ACTION_TERMINATE_INDEX:
                self._robots_activos[i] = False # Marca este robot como inactivo para futuros pasos
                continue # No hay movimiento ni visita de celda en este paso para este robot

            fila_actual, columna_actual = self._posicion_robots[i]
            fila_delta, columna_delta = ACTION_MAP[action_index]
            fila_nueva = fila_actual + fila_delta
            columna_nueva = columna_actual + columna_delta

            # Comprobamos que la nueva posición sea válida:
            if self.es_posicion_valida(fila_nueva, columna_nueva):
                posicion_siguiente[i] = (fila_nueva, columna_nueva) # Actualizamos la posición del robot
                self._celdas_visitadas.add((fila_nueva, columna_nueva)) # Añadimos la celda visitada al conjunto de celdas visitadas
            

        # Actualizamos la posición de los robots:
        self._posicion_robots = posicion_siguiente

        completado = len(self._celdas_visitadas) == self.total_celdas
        # Comprobamos si todos los robots han terminado:
        todos_terminados = not any(self._robots_activos)

        finalizar = todos_terminados or completado

        return self.get_estado(), finalizar # Devolvemos el estado y si se ha completado el tablero o no
    

    def render(self, modo: str = 'human') -> None:
        """
        Muestra una representación textual simple del estado actual del tablero.
        """
        if modo == 'human':
            # Crear una matriz vacía para el tablero
            grid_repr = [[' ' for _ in range(self.columnas)] for _ in range(self.filas)]

            # Marcar celdas visitadas
            for r, c in self._celdas_visitadas:
                if self.es_posicion_valida(r, c):
                    grid_repr[r][c] = '.' # Punto para visitada

            # Marcar posiciones de los robots (con manejo básico de superposición)
            robot_markers_en_celda = {}
            for i, (r, c) in enumerate(self._posicion_robots):
                if self.es_posicion_valida(r, c):
                    marker = str(i) if self._robots_activos[i] else 'x' # ID si activo, 'x' si inactivo
                    if (r, c) not in robot_markers_en_celda:
                        robot_markers_en_celda[(r, c)] = []
                    robot_markers_en_celda[(r, c)].append(marker)

            # Añadir marcadores de robots al grid
            for (r, c), markers in robot_markers_en_celda.items():
                if len(markers) == 1:
                    grid_repr[r][c] = markers[0]
                else:
                    grid_repr[r][c] = '+' # Símbolo para múltiples robots

            # Imprimir el tablero
            print("-" * (self.columnas * 2 + 1))
            for r in range(self.filas):
                print("|" + "|".join(grid_repr[r]) + "|")
            print("-" * (self.columnas * 2 + 1))
            print(f"Visitadas: {len(self._celdas_visitadas)}/{self.total_celdas} | "
                    f"Activos: {self._robots_activos}")
            

if __name__ == "__main__":
    # Ejemplo de uso:
    tablero = Tablero(5, 5, 2)
    tablero.render()
    estado, done = tablero.step([1, 1]) # Ambos robots intentan moverse
    estado, done = tablero.step([1, 3]) # Ambos robots intentan moverse

    tablero.render()
    print("Estado:", estado)
    print("Episodio terminado:", done)