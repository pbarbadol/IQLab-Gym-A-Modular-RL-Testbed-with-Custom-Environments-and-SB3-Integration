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

    def _get_info(self):
        return {
            "explored_cells": int(np.sum(self._grid)),
            "active_robots": int(np.sum(self._robot_active)),
            "robot_positions": [list(pos) for pos in self._robot_positions],
            "current_step": self.current_step,
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
    

# --- Main para probar (sin cambios significativos, debería seguir funcionando) ---
if __name__ == '__main__':
    print("--- Iniciando prueba con agente aleatorio (versión simplificada) ---")
    GRID_SIZE_TEST = 10
    NUM_ROBOTS_TEST = 2
    MAX_STEPS_TEST = 100
    DENSE_REWARD_TEST = True


    # Definir policy_kwargs para usar el extractor personalizado
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=64) # Puedes ajustar este 64 a 32, 128, etc.
                                                        # Es la dimensionalidad de las características extraídas del grid_map
    )

    env_test = MultiRobotGridEnv(
        grid_size=GRID_SIZE_TEST,
        num_robots=NUM_ROBOTS_TEST,
        render_mode='human',
        max_steps=MAX_STEPS_TEST,
        dense_reward=DENSE_REWARD_TEST
    )
    print("Espacio de Acción:", env_test.action_space)
    print("Espacio de Observación:", env_test.observation_space)

    for episode in range(1): # Un episodio para prueba rápida
        obs, info = env_test.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        print(f"\n--- Episodio Aleatorio {episode+1} ---")
        while not terminated and not truncated:
            action = env_test.action_space.sample()
            obs, reward, terminated, truncated, info = env_test.step(action)
            total_reward += reward
            steps += 1
            if steps % 20 == 0 or terminated or truncated:
                print(f"  Paso {steps}: Recompensa: {reward:.2f}, Celdas Expl: {info['explored_cells']}, Robots Act: {info['active_robots']}")

            if terminated or truncated:
                print(f"Episodio {episode+1} {'terminado' if terminated else 'truncado'} después de {steps} pasos.")
                print(f"  Recompensa total: {total_reward:.2f}")
                print(f"  Celdas exploradas: {info['explored_cells']} / {env_test.grid_size**2}")
                break
    env_test.close()
    print("Prueba aleatoria finalizada.")

    print("\n--- Iniciando prueba con Stable Baselines3 PPO (versión simplificada) ---")
    GRID_SIZE_TRAIN = 8
    NUM_ROBOTS_TRAIN = 3
    MAX_STEPS_TRAIN = 80
    DENSE_REWARD_TRAIN = True
    TOTAL_TIMESTEPS_TRAIN = 2_000_000 

    def make_env(grid_size, num_robots, max_steps, dense_reward, render_mode=None):
        def _init():
            env = MultiRobotGridEnv(
                grid_size=grid_size, num_robots=num_robots, max_steps=max_steps,
                dense_reward=dense_reward, render_mode=render_mode
            )
            return env
        return _init

    print("Verificando el entorno individual...")
    env_single_check = make_env(GRID_SIZE_TRAIN, NUM_ROBOTS_TRAIN, MAX_STEPS_TRAIN, DENSE_REWARD_TRAIN)()
    try:
        check_env(env_single_check, warn=True, skip_render_check=True)
        print("Verificación del entorno individual completada.")
    except Exception as e:
        print(f"Error durante check_env: {e}")
        env_single_check.close()
        exit()
    finally:
        env_single_check.close()

    env_train_sb3 = DummyVecEnv([make_env(GRID_SIZE_TRAIN, NUM_ROBOTS_TRAIN, MAX_STEPS_TRAIN, DENSE_REWARD_TRAIN)])
    
    model = PPO(
        "MultiInputPolicy",
        env_train_sb3,
        verbose=0,
        tensorboard_log="./ppo_multirobot_simplified_tensorboard/",
        ent_coef=0.01,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=32,
        policy_kwargs=policy_kwargs  # <--- Propia red convolucional
    )

    print(f"Empezando entrenamiento PPO por {TOTAL_TIMESTEPS_TRAIN} timesteps...")
    #model.learn(total_timesteps=TOTAL_TIMESTEPS_TRAIN, progress_bar=True)
    print("Entrenamiento completado.")

    model_path = "./ppo_multirobot_simplified_model2"
    #model.save(model_path)
    print(f"Modelo guardado en {model_path}.zip")
    print("Cargando modelo guardado...")
    model = PPO.load(model_path, env=env_train_sb3)
    print("\n--- Probando el agente entrenado (versión simplificada) ---")
    env_eval = MultiRobotGridEnv(
        grid_size=GRID_SIZE_TRAIN, num_robots=NUM_ROBOTS_TRAIN,
        max_steps=MAX_STEPS_TRAIN + 20, render_mode='human', dense_reward=DENSE_REWARD_TRAIN
    )

    for episode in range(2): # Dos episodios de evaluación
        obs, info = env_eval.reset()
        terminated, truncated = False, False
        episode_reward, episode_steps = 0, 0
        print(f"\n--- Evaluación Episodio {episode + 1} ---")
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_eval.step(action)
            episode_reward += reward
            episode_steps += 1
            if terminated or truncated:
                print(f"Evaluación Episodio {episode + 1} finalizado en {episode_steps} pasos. Recompensa: {episode_reward:.2f}")
                print(f"  Info final: Celdas Expl: {info['explored_cells']}, Robots Act: {info['active_robots']}")
                break
    
    env_train_sb3.close()
    env_eval.close()
    print("\nPrueba del agente entrenado finalizada.")