import pygame
import sys
import time
import pickle
from collections import defaultdict
import os # Para comprobar si existe el archivo

# --- Importar desde nuestros módulos ---
try:
    from tablero import Tablero, StateType, ACTION_MAP, ACTION_TERMINATE_INDEX
    from q_learning_agent import elegir_acciones, inicializar_q_tables, QTableType
    import config # Para dimensiones, archivo Q-table, etc.
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de que 'tablero.py', 'q_learning_agent.py' y 'config.py' estén en el mismo directorio o accesibles.")
    sys.exit()

# --- Configuración Visual de Pygame ---
CELL_SIZE = 60        # Tamaño de cada celda en píxeles
MARGIN = 10         # Margen alrededor del tablero
INFO_HEIGHT = 50      # Espacio extra abajo para texto de información
FPS = 5             # Frames por segundo (controla la velocidad de la simulación)

# Colores (RGB)
COLOR_FONDO = (40, 40, 40)
COLOR_LINEAS = (90, 90, 90)
COLOR_CELDA_VACIA = (60, 60, 60)
COLOR_CELDA_VISITADA = (70, 90, 70) # Verde oscuro
COLOR_ROBOT_ACTIVO = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)] # Rojo, Azul, Verde, Amarillo, Magenta, Cian
COLOR_ROBOT_INACTIVO = (100, 100, 100) # Gris
COLOR_TEXTO = (220, 220, 220)
COLOR_INFO_BG = (50, 50, 50)

# --- Funciones Auxiliares ---

def cargar_q_tables_visual(num_robots: int, filename: str) -> QTableType | None:
    """
    Carga la lista de Q-tables desde un archivo pickle.
    Convierte los dicts cargados de nuevo a defaultdicts anidados.
    Devuelve None si el archivo no existe o hay un error.
    """
    if not os.path.exists(filename):
        print(f"Advertencia: Archivo de Q-Tables '{filename}' no encontrado.")
        return None
    try:
        with open(filename, 'rb') as f:
            # Asume que se guardó como una lista de dicts normales
            list_of_dicts = pickle.load(f)
            if len(list_of_dicts) != num_robots:
                print(f"Error: El archivo Q-Table contiene datos para {len(list_of_dicts)} robots, se esperaban {num_robots}.")
                return None

            # Convertir de nuevo a defaultdict anidado
            q_tables = inicializar_q_tables(num_robots) # Crea la estructura defaultdict
            for i in range(num_robots):
                loaded_dict = list_of_dicts[i]
                # Recorrer el dict cargado y poblar el defaultdict
                for state_tuple, action_values_dict in loaded_dict.items():
                    # Reconstruir el defaultdict interno para las acciones
                    action_defaultdict = defaultdict(float, action_values_dict)
                    q_tables[i][state_tuple] = action_defaultdict # Asignar al defaultdict externo

            print(f"Q-Tables cargadas correctamente desde '{filename}'.")
            return q_tables
    except Exception as e:
        print(f"Error crítico al cargar o procesar Q-Tables desde '{filename}': {e}")
        return None

def get_robot_color(robot_id: int, activo: bool) -> tuple[int, int, int]:
    """Devuelve el color para un robot según su ID y estado activo."""
    if activo:
        # Cicla a través de los colores definidos si hay más robots que colores
        return COLOR_ROBOT_ACTIVO[robot_id % len(COLOR_ROBOT_ACTIVO)]
    else:
        return COLOR_ROBOT_INACTIVO

def draw_board_state(surface: pygame.Surface, env: Tablero, estado: StateType, font: pygame.font.Font):
    """Dibuja el estado actual del tablero en la superficie de Pygame."""
    surface.fill(COLOR_FONDO) # Limpiar pantalla con color de fondo

    posiciones, visitadas, activos = estado
    filas, columnas = env.filas, env.columnas

    # Dibujar celdas (visitadas o vacías)
    for r in range(filas):
        for c in range(columnas):
            rect = pygame.Rect(
                MARGIN + c * CELL_SIZE,
                MARGIN + r * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
            color_celda = COLOR_CELDA_VACIA
            if (r, c) in visitadas:
                color_celda = COLOR_CELDA_VISITADA
            pygame.draw.rect(surface, color_celda, rect)

    # Dibujar líneas de la cuadrícula
    for r in range(filas + 1):
        y = MARGIN + r * CELL_SIZE
        pygame.draw.line(surface, COLOR_LINEAS, (MARGIN, y), (MARGIN + columnas * CELL_SIZE, y), 1)
    for c in range(columnas + 1):
        x = MARGIN + c * CELL_SIZE
        pygame.draw.line(surface, COLOR_LINEAS, (x, MARGIN), (x, MARGIN + filas * CELL_SIZE), 1)

    # Dibujar robots
    radius = CELL_SIZE // 3 # Radio del círculo del robot
    for i in range(env.num_robots):
        r, c = posiciones[i]
        activo = activos[i]
        color_robot = get_robot_color(i, activo)

        # Centro del círculo del robot
        center_x = MARGIN + c * CELL_SIZE + CELL_SIZE // 2
        center_y = MARGIN + r * CELL_SIZE + CELL_SIZE // 2

        pygame.draw.circle(surface, color_robot, (center_x, center_y), radius)

        # Dibujar ID del robot dentro o cerca del círculo
        id_text = font.render(str(i), True, COLOR_FONDO) # ID en color de fondo para contraste
        text_rect = id_text.get_rect(center=(center_x, center_y))
        surface.blit(id_text, text_rect)

def draw_info_panel(surface: pygame.Surface, screen_width:int, screen_height: int,
                      step: int, max_steps: int, visited: int, total_cells: int,
                      font: pygame.font.Font):
    """Dibuja el panel de información en la parte inferior."""
    panel_rect = pygame.Rect(0, screen_height - INFO_HEIGHT, screen_width, INFO_HEIGHT)
    pygame.draw.rect(surface, COLOR_INFO_BG, panel_rect)

    info_text = f"Paso: {step}/{max_steps} | Visitadas: {visited}/{total_cells}"
    text_surface = font.render(info_text, True, COLOR_TEXTO)
    text_rect = text_surface.get_rect(midleft=(MARGIN, screen_height - INFO_HEIGHT // 2))
    surface.blit(text_surface, text_rect)

# --- Función Principal de Visualización ---
def main():
    pygame.init()

    # Cargar configuración del entorno
    filas = config.FILAS
    columnas = config.COLUMNAS
    num_robots = config.N_ROBOTS

    # Configurar pantalla de Pygame
    screen_width = columnas * CELL_SIZE + 2 * MARGIN
    screen_height = filas * CELL_SIZE + 2 * MARGIN + INFO_HEIGHT
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(f"Simulación Robots ({filas}x{columnas}) - Q-Learning")

    # Cargar fuente para texto
    try:
        font = pygame.font.SysFont(None, CELL_SIZE // 2) # Fuente por defecto del sistema
        info_font = pygame.font.SysFont(None, INFO_HEIGHT // 2)
    except Exception as e:
        print(f"Error cargando fuente: {e}. Usando fuente por defecto.")
        font = pygame.font.Font(None, CELL_SIZE // 2) # Fuente básica de Pygame
        info_font = pygame.font.Font(None, INFO_HEIGHT // 2)

    # Cargar Q-Tables aprendidas
    q_tables = cargar_q_tables_visual(num_robots, config.CARGAR_QTABLES_FILENAME)
    if q_tables is None:
        print("No se pudieron cargar las Q-Tables. La simulación no puede continuar con política aprendida.")
        print("Ejecuta 'train.py' primero para generar el archivo " + config.CARGAR_QTABLES_FILENAME)
        pygame.quit()
        sys.exit()

    # Crear el entorno
    # En train.py y visualize_sim.py
    env = Tablero(
        filas=config.FILAS,
        columnas=config.COLUMNAS,
        num_robots=config.N_ROBOTS,
        posicion_inicial=config.POSICION_INICIAL,
        posicion_carga=config.POSICION_CARGA,      # <<< NUEVO
        bateria_maxima=config.BATERIA_MAXIMA,      # <<< NUEVO
        bateria_inicial=config.BATERIA_INICIAL       # <<< NUEVO
    )

    # Variables de simulación
    estado = env.reset()
    done = False
    paso = 0
    max_steps_eval = config.MAX_STEPS_EVALUACION
    clock = pygame.time.Clock()

    print("\n--- Iniciando Visualización (Política Aprendida, Epsilon=0) ---")
    print("Presiona ESC o cierra la ventana para salir.")

    # Bucle principal de la simulación/visualización
    running = True
    while running and not done and paso < max_steps_eval:
        # Manejo de eventos de Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not running:
            break

        # 1. Elegir acciones usando la política aprendida (epsilon=0)
        acciones = elegir_acciones(q_tables, estado, 0.0, env)

        # 2. Ejecutar un paso en el entorno
        estado_siguiente, done = env.step(acciones)
        estado = estado_siguiente # Actualizar estado
        paso += 1

        # 3. Dibujar el estado actual
        screen.fill(COLOR_FONDO) # Limpiar antes de redibujar
        draw_board_state(screen, env, estado, font)
        draw_info_panel(screen, screen_width, screen_height, paso, max_steps_eval,
                        len(estado[1]), env.total_celdas, info_font)

        # 4. Actualizar la pantalla
        pygame.display.flip()

        # 5. Controlar la velocidad de la simulación
        clock.tick(FPS)

    # --- Fin del bucle ---
    if running: # Si no se salió prematuramente
        print("\n--- Simulación Finalizada ---")
        if done:
             print(f"Objetivo alcanzado o todos los robots terminaron en {paso} pasos.")
        elif paso >= max_steps_eval:
             print(f"Se alcanzó el límite de pasos ({max_steps_eval}).")
        else:
             print("Simulación interrumpida.")

        # Mantener la ventana abierta un poco al final
        end_time = time.time()
        while time.time() - end_time < 3: # Espera 3 segundos
             for event in pygame.event.get():
                 if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                     running=False
                     break
             if not running: break

    pygame.quit()
    sys.exit()

# --- Ejecutar el script ---
if __name__ == "__main__":
    main()