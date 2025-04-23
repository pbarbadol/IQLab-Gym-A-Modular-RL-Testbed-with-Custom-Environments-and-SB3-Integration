# config.py

# --- Configuración del Entorno ---
FILAS = 5                     # Número de filas del tablero
COLUMNAS = 5                  # Número de columnas del tablero
N_ROBOTS = 2                  # Número de robots
# Posición inicial opcional (None para empezar todos en (0,0))
#POSICION_INICIAL = [(0, 0), (FILAS - 1, COLUMNAS - 1)]
POSICION_INICIAL = None

# --- Hiperparámetros de Q-Learning ---
ALPHA = 0.1                   # Tasa de aprendizaje (Learning Rate) - Qué tanto aprende de la nueva info
GAMMA = 0.9                   # Factor de descuento (Discount Factor) - Importancia de recompensas futuras
EPSILON_START = 1.0           # Tasa de exploración inicial (100% aleatorio al principio)
EPSILON_END = 0.01            # Tasa de exploración final (mínimo 1% de exploración)
EPSILON_DECAY = 0.9995        # Factor de decaimiento de epsilon por episodio (más lento = más exploración)

# --- Configuración del Entrenamiento ---
NUM_EPISODIOS = 100000         # Número total de episodios de entrenamiento
MAX_STEPS_PER_EPISODE = 150   # Límite de pasos por episodio (evita bucles infinitos)

# --- Configuración Adicional ---
EPISODIOS_PARA_LOG = 100      # Cada cuántos episodios imprimir progreso
GUARDAR_QTABLES_FILENAME = "q_tables_final.pkl" # Nombre archivo para guardar Q-Tables
CARGAR_QTABLES_FILENAME = "q_tables_final.pkl"  # Nombre archivo para cargar Q-Tables (puede ser el mismo)
GENERAR_GRAFICO = True        # Si se genera el gráfico de recompensas al final
VENTANA_PROMEDIO_MOVIL = 100  # Ventana para suavizar el gráfico de recompensas

# --- Configuración de Evaluación ---
EVALUAR_AL_FINAL = True       # Si se ejecuta un episodio de prueba al final
MAX_STEPS_EVALUACION = MAX_STEPS_PER_EPISODE * 2 # Pasos máximos en la evaluación
RENDER_EVALUACION = True      # Si se muestra el tablero durante la evaluación
PAUSA_RENDER_EVAL = 0.3       # Segundos de pausa entre pasos de render (0 para no pausar)