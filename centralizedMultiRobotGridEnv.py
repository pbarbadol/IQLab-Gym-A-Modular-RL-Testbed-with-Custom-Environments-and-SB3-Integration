import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MultiRobotGridEnv(gym.Env):
    """
    Entorno Gymnasium simplificado para la exploración de una cuadrícula por múltiples robots.
    """
    # --- Constantes ---
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_TERMINATE = 4
    NUM_ACTIONS = 5

    REWARD_NEW_CELL_EXPLORED = 2.0
    PENALTY_PER_STEP = -0.1
    PENALTY_ROBOT_TERMINATES_EARLY = -20.0 # Penalización simplificada
    PENALTY_ROBOT_COLISION = -10.0

    OBS_GRID_MAP = "grid_map"
    OBS_VECTOR_FEATURES = "vector_features"

    CELL_NOT_EXPLORED = 0
    CELL_EXPLORED = 1
    ROBOT_LOCATION_MARKER = 1.0

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self, grid_size=10, num_robots=3, max_steps=200, render_mode=None, dense_reward=True):
        super().__init__()

        self.grid_size = grid_size
        self.num_robots = num_robots
        self.max_steps = max_steps
        self.current_step = 0
        self.dense_reward = dense_reward

        self.action_space = spaces.MultiDiscrete([self.NUM_ACTIONS] * self.num_robots)

        self.observation_space = spaces.Dict({
            self.OBS_GRID_MAP: spaces.Box(
                low=0, high=255, shape=(2, self.grid_size, self.grid_size), dtype=np.uint8
            ),
            self.OBS_VECTOR_FEATURES: spaces.Box(
                low=0, high=255, shape=(self.num_robots * 2 + self.num_robots + 1,), dtype=np.uint8
            )
        })

        self.reward_new_cell = self.REWARD_NEW_CELL_EXPLORED
        self.reward_step_penalty = self.PENALTY_PER_STEP if self.dense_reward else 0.0
        self.reward_all_explored_bonus = float(self.grid_size * self.grid_size * 1.5)
        self.reward_robot_terminates_early_penalty = self.PENALTY_ROBOT_TERMINATES_EARLY if self.dense_reward else 0.0

        self._grid = None
        self._robot_positions = None
        self._robot_active = None

        self.render_mode = render_mode
        self.window_size = 512
        self.window = None
        self.clock = None
        self.font = None
        self.cell_size = self.window_size // self.grid_size

        if self.render_mode in ["human", "rgb_array"]:
            pygame.font.init() # Asumimos que esto funciona
            self.font = pygame.font.Font(None, int(self.cell_size * 0.5))


    def _get_obs(self):
        explored_map = (self._grid * 255).astype(np.float32)
        

        # Canal 1: Posiciones de los robots activos (0 o 255)
        # Primero creamos el mapa con floats (0.0 o ROBOT_LOCATION_MARKER)
        robot_location_map_float = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for i in range(self.num_robots):
            if self._robot_active[i]:
                row, col = self._robot_positions[i]
                # ROBOT_LOCATION_MARKER es 1.0, lo escalaremos después
                robot_location_map_float[row, col] = self.ROBOT_LOCATION_MARKER

        robot_location_map = (robot_location_map_float * 255).astype(np.uint8)

        grid_map_obs = np.stack([explored_map, robot_location_map], axis=0)
        # grid_map_obs ya es uint8 debido a los .astype(np.uint8) anteriores

        robot_pos_normalized = []
        for r_idx in range(self.num_robots):
            r, c = self._robot_positions[r_idx]
            norm_r = r / max(1, self.grid_size - 1)
            norm_c = c / max(1, self.grid_size - 1)
            robot_pos_normalized.extend([norm_r, norm_c])

        robot_active_float = np.array(self._robot_active, dtype=np.float32)
        current_step_normalized = np.array([self.current_step / self.max_steps], dtype=np.float32)

        vector_features_obs = np.concatenate(
            (np.array(robot_pos_normalized, dtype=np.float32),
             robot_active_float,
             current_step_normalized)
        ).astype(self.observation_space[self.OBS_VECTOR_FEATURES].dtype)
        grid_map_obs = grid_map_obs.astype(np.uint8) # Asegurar que el stack final sea uint8
        return {
            self.OBS_GRID_MAP: grid_map_obs,
            self.OBS_VECTOR_FEATURES: vector_features_obs
        }

    # Dentro de tu clase MultiRobotGridEnv:

    def _get_info(self):
        total_cells = self.grid_size * self.grid_size
        explored_cells = int(np.sum(self._grid))
        return {
            "explored_cells": explored_cells,
            "active_robots": int(np.sum(self._robot_active)),
            "robot_positions": [list(pos) for pos in self._robot_positions],
            "current_step": self.current_step,
            "is_success": explored_cells == total_cells # IMPORTANTE para Monitor
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self._robot_positions = []
        self._robot_active = [True] * self.num_robots

        for _ in range(self.num_robots):
            pos = (self.np_random.integers(0, self.grid_size), self.np_random.integers(0, self.grid_size))
            self._robot_positions.append(pos)
            self._grid[pos[0], pos[1]] = self.CELL_EXPLORED

        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        reward = 0.0

        for i in range(self.num_robots):
            if not self._robot_active[i]:
                continue

            robot_action = action[i]

            if robot_action != self.ACTION_TERMINATE:
                reward += self.reward_step_penalty
            else: # ACTION_TERMINATE
                self._robot_active[i] = False
                # Penalización simplificada si termina y no está todo explorado
                if np.sum(self._grid) < (self.grid_size * self.grid_size * 0.9): # Si menos del 90% explorado
                    reward += self.reward_robot_terminates_early_penalty
                continue # El robot termina, no se mueve

            row, col = self._robot_positions[i]
            d_row, d_col = 0, 0
            if robot_action == self.ACTION_UP: d_row = -1
            elif robot_action == self.ACTION_DOWN: d_row = 1
            elif robot_action == self.ACTION_LEFT: d_col = -1
            elif robot_action == self.ACTION_RIGHT: d_col = 1

            new_row, new_col = row + d_row, col + d_col

            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                self._robot_positions[i] = (new_row, new_col)
                if self._grid[new_row, new_col] == self.CELL_NOT_EXPLORED:
                    self._grid[new_row, new_col] = self.CELL_EXPLORED
                    reward += self.reward_new_cell

        terminated = False
        truncated = False
        explored_cells = np.sum(self._grid)
        total_cells = self.grid_size * self.grid_size

        if not any(self._robot_active):
            terminated = True
        
        if explored_cells == total_cells:
            terminated = True
            if not self.dense_reward:
                reward = self.reward_all_explored_bonus
            else:
                reward += self.reward_all_explored_bonus

        if self.current_step >= self.max_steps:
            truncated = True
            if not self.dense_reward and not terminated:
                reward = (explored_cells / total_cells) * self.reward_all_explored_bonus
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Multi-Robot Grid Exploration")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                fill_color = (200, 200, 200) if self._grid[r, c] == self.CELL_EXPLORED else (230, 230, 230)
                pygame.draw.rect(canvas, fill_color, rect)
                pygame.draw.rect(canvas, (0, 0, 0), rect, 1)

        robot_colors = [(255,0,0), (0,255,0), (0,0,255), (255,165,0), (128,0,128), (0,128,128)]
        robot_counts_on_cell = {}

        for i in range(self.num_robots):
            if self._robot_active[i]:
                pos = self._robot_positions[i]
                row, col = pos
                count_at_pos = robot_counts_on_cell.get(pos, 0)
                
                # Desplazamiento simple para robots en la misma celda
                offset_x_factor = (count_at_pos % 2 - 0.5) * 0.3 # Alterna entre -0.15 y +0.15 (para 2 robots)
                offset_y_factor = (count_at_pos // 2 % 2 - 0.5) * 0.3 # Similar para Y si hay más
                if self.num_robots == 1: # Sin offset si solo hay un robot
                    offset_x_factor = 0
                    offset_y_factor = 0

                center_x = (col + 0.5 + offset_x_factor) * self.cell_size
                center_y = (row + 0.5 + offset_y_factor) * self.cell_size
                radius = self.cell_size * 0.25

                pygame.draw.circle(canvas, robot_colors[i % len(robot_colors)], (center_x, center_y), radius)

                if self.font:
                    text_surf = self.font.render(str(i), True, (0,0,0))
                    text_rect = text_surf.get_rect(center=(center_x, center_y))
                    canvas.blit(text_surf, text_rect)
                
                robot_counts_on_cell[pos] = count_at_pos + 1

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else: # rgb_array
            return np.transpose(pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        # Pygame.font.quit() es llamado por pygame.quit() si font fue inicializado.
        # Solo llamar a pygame.quit() si algún módulo fue inicializado.
        if pygame.get_init():
             pygame.quit()
        self.font = None # Asegurar que la referencia a la fuente se limpie

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 64): # Ajusta features_dim según necesites
        super().__init__(observation_space, features_dim)
        
        # Asumimos que la clave para el mapa es MultiRobotGridEnv.OBS_GRID_MAP
        # Necesitas acceder a la forma del sub-espacio 'grid_map'
        grid_map_space = observation_space.spaces[MultiRobotGridEnv.OBS_GRID_MAP] # Acceder al sub-espacio
        n_input_channels = grid_map_space.shape[0] # Debería ser 2
        height = grid_map_space.shape[1]
        width = grid_map_space.shape[2]

        # Define una CNN más pequeña. Ajusta esto según tu grid_size.
        # Para un grid de 8x8, necesitamos ser muy conservadores.
        self.cnn = nn.Sequential(
            # Entrada: (2, 8, 8)
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1), # Salida: (16, 8, 8)
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2), # Si usas esto, salida: (16, 4, 4)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),          # Salida: (32, 8, 8) o (32, 4, 4) si MaxPool
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2), # Si usas esto, salida: (32, 2, 2)
            nn.Flatten(),
        )

        # Calcular el tamaño de la salida aplanada de la CNN dinámicamente
        with torch.no_grad():
            # Crea un tensor de muestra con la forma correcta del espacio de observación del grid_map
            # El primer '1' es para el tamaño del batch
            dummy_input = torch.as_tensor(grid_map_space.sample()[None]).float()
            # Las observaciones de imagen uint8 se normalizan a float [0,1] por SB3 antes de entrar a la CNN
            # así que .float() está bien aquí.
            n_flatten = self.cnn(dummy_input).shape[1]

        # Capa lineal para proyectar a features_dim
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: spaces.Dict) -> torch.Tensor:
        # Extraer la parte 'grid_map' de las observaciones
        # SB3 ya se encarga de pasar el tensor correcto (normalizado si era uint8)
        grid_map_tensor = observations[MultiRobotGridEnv.OBS_GRID_MAP]
        return self.linear(self.cnn(grid_map_tensor))
    

# --- Main para probar (CON MODIFICACIONES PARA LOGGING Y PPO) ---
if __name__ == '__main__':
    # --- La prueba con agente aleatorio se puede mantener o comentar ---
    # print("--- Iniciando prueba con agente aleatorio (versión simplificada) ---")
    # ... (tu código de prueba aleatoria) ...
    # print("Prueba aleatoria finalizada.")

    import os # Necesario para crear directorios
    from stable_baselines3.common.monitor import Monitor # IMPORTANTE

    print("\n--- Iniciando Entrenamiento PPO con Logging Mejorado ---")
    
    # --- Parámetros de Entrenamiento ---
    GRID_SIZE_TRAIN = 8
    NUM_ROBOTS_TRAIN = 3
    # MAX_STEPS_TRAIN = GRID_SIZE_TRAIN * GRID_SIZE_TRAIN # Permitir suficientes pasos para explorar teóricamente
    MAX_STEPS_TRAIN = int(GRID_SIZE_TRAIN * GRID_SIZE_TRAIN * 1.25) # Un poco más de margen
    DENSE_REWARD_TRAIN = True
    # Vamos a poner un valor para una ejecución de prueba más corta, o mantén los 5M si tienes tiempo.
    TOTAL_TIMESTEPS_TRAIN = 5_000_000 # Prueba más corta para verificar logging
    # TOTAL_TIMESTEPS_TRAIN = 5_000_000 # Si quieres intentar un entrenamiento más largo

    # --- Directorios para Logs y Modelo ---
    EXPERIMENT_NAME = f"PPO_MR_{NUM_ROBOTS_TRAIN}R_{GRID_SIZE_TRAIN}x{GRID_SIZE_TRAIN}"
    LOG_DIR_BASE = "./ppo_training_logs/"
    MODEL_DIR_BASE = "./ppo_trained_models/"
    
    TENSORBOARD_LOG_PATH = os.path.join(LOG_DIR_BASE, EXPERIMENT_NAME)
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR_BASE, EXPERIMENT_NAME, "model") # SB3 añade .zip

    os.makedirs(TENSORBOARD_LOG_PATH, exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR_BASE, EXPERIMENT_NAME), exist_ok=True)

    # --- Factory para el Entorno con Monitor ---
    def make_env_monitored(grid_size, num_robots, max_steps, dense_reward, 
                           render_mode=None, rank=0, seed=0, log_dir=None, 
                           include_terminate=True): # Añadido include_terminate
        def _init():
            env = MultiRobotGridEnv(
                grid_size=grid_size,
                num_robots=num_robots,
                max_steps=max_steps,
                dense_reward=dense_reward,
                render_mode=render_mode,
                # include_terminate_action=include_terminate # Asegúrate que tu env lo acepte
            )
            
            monitor_path_for_trial = None
            if log_dir: # Solo crear path si log_dir se proporciona
                monitor_path_for_trial = os.path.join(log_dir, f"monitor_{rank}")
                # No necesitas crear el directorio aquí, Monitor lo hace si filename se da.
            
            # Envolver con Monitor. Pasar info_keywords para que se logueen.
            env = Monitor(env, filename=monitor_path_for_trial, 
                          info_keywords=("explored_cells", "active_robots", "is_success"))
            
            env.reset(seed=seed + rank) # Resetear después de wrappers es una buena práctica
            return env
        return _init

    # --- Verificación del Entorno (opcional pero recomendado) ---
    print("Verificando el entorno individual con Monitor...")
    # Para check_env, no necesitamos log_dir, rank, o seed necesariamente si es solo para verificar la API.
    # Pero la factory los espera, así que los pasamos.
    env_check_instance = make_env_monitored(GRID_SIZE_TRAIN, NUM_ROBOTS_TRAIN, MAX_STEPS_TRAIN, 
                                            DENSE_REWARD_TRAIN, rank=0, seed=42, log_dir=None,
                                            include_terminate=True)() # Llamar para obtener la instancia
    try:
        check_env(env_check_instance, warn=True, skip_render_check=True)
        print("Verificación del entorno con Monitor completada.")
    except Exception as e:
        print(f"Error durante check_env: {e}")
        env_check_instance.close() # Asegúrate de cerrar si la verificación falla y sales
        exit()
    finally:
        env_check_instance.close() # Siempre cierra el entorno de verificación

    # --- Crear Entorno Vectorizado para Entrenamiento ---
    # Para este ejemplo, usaremos un solo entorno (num_cpu = 1)
    # Si tuvieras más CPUs, podrías aumentar esto y usar SubprocVecEnv
    num_cpu_train = 1
    env_train_sb3 = DummyVecEnv([
        make_env_monitored(
            GRID_SIZE_TRAIN, NUM_ROBOTS_TRAIN, MAX_STEPS_TRAIN, DENSE_REWARD_TRAIN,
            rank=i, seed=42, log_dir=TENSORBOARD_LOG_PATH, # Usar el mismo dir para logs de Monitor
            include_terminate=True
        ) for i in range(num_cpu_train)
    ])

    # --- Configuración de la Política y el Extractor ---
    # La dimensionalidad de salida de tu CustomCNN
    # Si tu CNN tiene Flatten() -> Linear(n_flatten, features_dim)
    # features_dim es cnn_output_features_dim.
    cnn_output_features_dim = 128  # Ajusta esto según la salida de tu self.linear en CustomCNN

    policy_kwargs_ppo = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=cnn_output_features_dim),
        # Es MUY recomendable definir net_arch para controlar las capas después de la concatenación
        # de la salida de la CNN y la MLP de las características vectoriales.
        # El tamaño de entrada a estas capas es cnn_output_features_dim + (salida de MLP para vector_features)
        # SB3 usa por defecto una MLP [256] para los vector_features si no hay un extractor específico para ellos.
        # Así que la entrada a net_arch sería cnn_output_features_dim + 256 (aproximadamente)
        # O si la MLP de vector_features es más simple, podría ser cnn_output_features_dim + len(vector_features)
        # Prueba con capas de tamaño intermedio o grande:
        net_arch=dict(pi=[256, 128], vf=[256, 128]) # Ejemplo de arquitectura para política y valor
    )
    
    # --- Hiperparámetros PPO Ajustados ---
    # n_steps: Número de pasos a recolectar por entorno antes de actualizar.
    # Total de datos para una actualización = n_steps * num_cpu_train
    # Un buen valor total suele ser 1024, 2048, o 4096.
    # Si MAX_STEPS_TRAIN es ~80, y queremos ver varios episodios:
    n_steps_ppo = 1024 // num_cpu_train if num_cpu_train > 0 else 1024 
    if n_steps_ppo < MAX_STEPS_TRAIN : # Asegurar al menos un episodio completo si es posible
        n_steps_ppo = MAX_STEPS_TRAIN * 2 # Ej: recolectar datos de 2 episodios completos

    model = PPO(
        "MultiInputPolicy",
        env_train_sb3,
        verbose=1, # Para ver logs de SB3 durante el entrenamiento
        tensorboard_log=TENSORBOARD_LOG_PATH, # Directorio para los logs de TensorBoard
        learning_rate=3e-4,     # Puede necesitar ajuste (e.g., 1e-4)
        n_steps=n_steps_ppo,
        batch_size=64,          # Minibatch size
        n_epochs=10,            # Épocas de optimización por recolección
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # Coeficiente de entropía (0.01 puede ser un buen inicio)
        vf_coef=0.5,            # Coeficiente de la función de valor
        max_grad_norm=0.5,      # Recorte de gradiente
        policy_kwargs=policy_kwargs_ppo
    )

    print(f"Empezando entrenamiento PPO por {TOTAL_TIMESTEPS_TRAIN} timesteps...")
    print(f"Logs de TensorBoard se guardarán en: {TENSORBOARD_LOG_PATH}")
    print(f"Modelo se guardará en: {MODEL_SAVE_PATH}.zip")
    
    try:
        #model.learn(total_timesteps=TOTAL_TIMESTEPS_TRAIN, progress_bar=True)
        print("Entrenamiento completado.")
        #model.save(MODEL_SAVE_PATH)
        print(f"Modelo guardado en {MODEL_SAVE_PATH}.zip")
    except Exception as e_learn:
        print(f"ERROR durante el entrenamiento: {e_learn}")
        import traceback
        traceback.print_exc()
    finally:
        env_train_sb3.close() # Siempre cierra el entorno de entrenamiento

    # --- Cargar y Probar el Agente Entrenado ---
    print(f"\nCargando modelo desde {MODEL_SAVE_PATH}.zip (si el entrenamiento fue exitoso)...")
    try:
        # No es necesario pasar 'env' a PPO.load si los espacios de obs/acción no cambian
        # y si estás usando el mismo tipo de política.
        loaded_model = PPO.load(MODEL_SAVE_PATH) 
        print("Modelo cargado exitosamente.")

        print("\n--- Probando el agente entrenado ---")
        env_eval = MultiRobotGridEnv(
            grid_size=GRID_SIZE_TRAIN, # Usar mismos params que entrenamiento para evaluación justa
            num_robots=NUM_ROBOTS_TRAIN,
            max_steps=MAX_STEPS_TRAIN + int(MAX_STEPS_TRAIN * 0.5), # Un poco más de margen
            render_mode='human', 
            dense_reward=DENSE_REWARD_TRAIN,
            # include_terminate_action=True # Asegúrate que esto coincida con el entrenamiento
        )

        for episode in range(3): # Evaluar por 3 episodios
            obs, info = env_eval.reset()
            terminated, truncated = False, False
            episode_reward, episode_steps = 0, 0
            print(f"\n--- Evaluación Episodio {episode + 1} ---")
            for eval_step_count in range(MAX_STEPS_TRAIN * 2): # Límite de pasos para la evaluación
                action, _ = loaded_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env_eval.step(action)
                episode_reward += reward
                episode_steps += 1
                if terminated or truncated:
                    print(f"Evaluación Episodio {episode + 1} finalizado en {episode_steps} pasos. Recompensa: {episode_reward:.2f}")
                    print(f"  Info final: Celdas Expl: {info['explored_cells']}, Activos: {info['active_robots']}, Éxito: {info.get('is_success', False)}")
                    break
            if not (terminated or truncated): # Si salió por el límite de eval_step_count
                 print(f"Evaluación Episodio {episode + 1} alcanzó el límite de pasos de evaluación ({eval_step_count+1}). Recompensa: {episode_reward:.2f}")
                 print(f"  Info final: Celdas Expl: {info['explored_cells']}, Activos: {info['active_robots']}, Éxito: {info.get('is_success', False)}")
        
        env_eval.close()

    except FileNotFoundError:
        print(f"No se encontró el modelo guardado en '{MODEL_SAVE_PATH}.zip'. Saltando evaluación.")
    except Exception as e_eval:
        print(f"ERROR durante la carga o evaluación del modelo: {e_eval}")
        import traceback
        traceback.print_exc()
        if 'env_eval' in locals() and env_eval: env_eval.close()
        
    print("\nPrueba del agente entrenado finalizada.")