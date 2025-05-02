# train_dqn_tablero.py

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Importa tu entorno y el wrapper
from tablero_gymnasium import TableroEnv # Asegúrate que el fichero se llame tablero_env.py
from flattenMultiDiscreteAction import FlattenMultiDiscreteAction

# --- Configuración del Entorno y Entrenamiento ---
FILAS = 4
COLUMNAS = 4
NUM_ROBOTS = 2 # ¡Cuidado si aumentas mucho este número!
MAX_STEPS_PER_EPISODE = 100 # Límite de pasos por episodio (importante para que no se eternice)
TOTAL_TIMESTEPS = 100_000 # Pasos totales de entrenamiento (ajusta según necesidad)
MODEL_SAVE_PATH = "dqn_tablero_model"

# --- Función para crear y envolver el entorno ---
def create_env():
    env = TableroEnv(
        filas=FILAS,
        columnas=COLUMNAS,
        num_robots=NUM_ROBOTS,
        max_steps=MAX_STEPS_PER_EPISODE,
        render_mode=None # Sin renderizado durante entrenamiento (más rápido)
    )
    # ¡Aplicamos el wrapper para aplanar la acción!
    env = FlattenMultiDiscreteAction(env)
    # Opcional: FlattenObservation si MultiInputPolicy da problemas
    # from gymnasium.wrappers import FlattenObservation
    # env = FlattenObservation(env)
    return env

# --- Creación del Entorno Vectorizado (SB3 lo prefiere) ---
# Usaremos DummyVecEnv ya que nuestro entorno es simple y no necesita paralelismo complejo
# env = make_vec_env(create_env, n_envs=1) # make_vec_env puede ser más simple
env = DummyVecEnv([create_env]) # Explicitamente DummyVecEnv

print("\n--- Entorno listo para SB3 ---")
print(f"Observation Space (después de wrappers): {env.observation_space}")
print(f"Action Space (después de wrappers): {env.action_space}")
print(f"Número de acciones discretas: {env.action_space.n}")


# --- Creación del Modelo DQN ---
# Para Dict observation space, necesitamos "MultiInputPolicy"
# Si usaste FlattenObservation, podrías usar "MlpPolicy"
policy_type = "MultiInputPolicy"

# Hiperparámetros (puedes necesitar ajustarlos mucho)
model = DQN(
    policy_type,
    env,
    verbose=1,                  # Muestra info del entrenamiento
    buffer_size=50_000,         # Tamaño del buffer de repetición
    learning_rate=1e-4,         # Tasa de aprendizaje
    batch_size=64,              # Tamaño del minibatch
    learning_starts=1000,       # Pasos antes de empezar a aprender
    exploration_fraction=0.2,   # Fracción de pasos para decaer epsilon
    exploration_final_eps=0.05, # Valor final de epsilon (exploración mínima)
    train_freq=(1, "step"),     # Frecuencia de entrenamiento (cada paso)
    gradient_steps=1,           # Pasos de gradiente por actualización
    target_update_interval=500, # Frecuencia de actualización de la red target
    tensorboard_log="./dqn_tablero_tensorboard/" # Para visualizar con TensorBoard
)

print("\n--- Iniciando Entrenamiento ---")
# Entrenar el agente
# Ajusta total_timesteps según el tiempo/recursos disponibles y la complejidad
model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=10, progress_bar=True)

# Guardar el modelo entrenado
model.save(MODEL_SAVE_PATH)
print(f"\n--- Modelo guardado en {MODEL_SAVE_PATH}.zip ---")

# --- Evaluación del Modelo Entrenado (Opcional) ---
print("\n--- Evaluando el modelo entrenado ---")

# Carga el modelo (aunque ya lo tenemos en memoria, es buena práctica mostrar cómo se carga)
# Asegúrate de pasarle un entorno (incluso uno no vectorizado) al cargar si tiene wrappers
eval_env_func = create_env # Reutiliza la función creadora
loaded_model = DQN.load(MODEL_SAVE_PATH, env=eval_env_func())

# O usa el modelo ya entrenado directamente: loaded_model = model

# Crea un entorno solo para evaluación (puedes activar el renderizado aquí)
eval_env = create_env()
# Para ver la evaluación:
# eval_env.render_mode = "human" # ¡OJO! El wrapper no tiene render_mode, hay que hacerlo en el env base ANTES de envolver
# Para que funcione el render hay que modificar create_env para que acepte render_mode="human"
# o crear una función específica eval_create_env

num_eval_episodes = 5
total_rewards = []

for i in range(num_eval_episodes):
    obs, info = eval_env.reset()
    terminated = False
    truncated = False
    episode_reward = 0
    step = 0
    print(f"\n--- Evaluación Episodio {i+1} ---")
    # eval_env.render() # Render inicial si está activado

    while not terminated and not truncated:
        step += 1
        # deterministic=True para usar la mejor acción aprendida (sin exploración épsilon)
        action, _states = loaded_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += reward
        # if eval_env.render_mode == "human":
        #     eval_env.render() # Renderiza cada paso
        #     import time
        #     time.sleep(0.2) # Pausa para ver

    print(f"Episodio {i+1} terminado. Pasos: {step} Recompensa: {episode_reward:.2f}")
    # print(f"  Celdas visitadas: {len(info.get('visited_cells_set', set()))}/{eval_env.unwrapped.total_celdas}") # Acceder al env original
    total_rewards.append(episode_reward)

print("\n--- Resultados Evaluación ---")
print(f"Recompensa media en {num_eval_episodes} episodios: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")

# Cerrar el entorno (importante si usa renderizado gráfico)
env.close()
eval_env.close()

print("\n--- Proceso Completo ---")