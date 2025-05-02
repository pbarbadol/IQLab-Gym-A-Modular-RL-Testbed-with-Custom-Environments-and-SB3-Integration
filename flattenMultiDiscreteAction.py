# wrapper_flatten_action.py
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActionWrapper
import numpy as np

class FlattenMultiDiscreteAction(ActionWrapper):
    """
    Wrapper para aplanar un espacio de acción MultiDiscrete en uno Discrete.
    Necesario para usar algoritmos como DQN de SB3 con entornos MultiDiscrete.

    ¡Advertencia! El tamaño del espacio Discrete resultante es A^N,
    donde A es el número de acciones atómicas y N es el número de agentes/dimensiones.
    Esto puede volverse inviable rápidamente.
    """
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(env.action_space, spaces.MultiDiscrete), \
            "El espacio de acción original debe ser MultiDiscrete"

        # Calcula el tamaño del nuevo espacio Discrete
        self.action_dims = env.action_space.nvec
        self.flat_action_size = np.prod(self.action_dims).item() # item() para obtener int de numpy

        # Define el nuevo espacio de acción aplanado
        self.action_space = spaces.Discrete(self.flat_action_size)

        print(f"Wrapper FlattenMultiDiscreteAction activado:")
        print(f"  - Espacio original: {env.action_space}")
        print(f"  - Dimensiones: {self.action_dims}")
        print(f"  - Nuevo espacio aplanado: {self.action_space}")
        if self.flat_action_size > 500: # Umbral arbitrario
            print(f"  - ¡ADVERTENCIA! El espacio de acción aplanado ({self.flat_action_size}) es muy grande. DQN puede tener dificultades.")


    def action(self, action):
        """
        Convierte la acción aplanada (int) de vuelta al formato MultiDiscrete (array).
        """
        assert self.action_space.contains(action), f"Acción {action} fuera de los límites de {self.action_space}"
        # np.unravel_index deshace el aplanamiento
        unflattened_action = np.unravel_index(action, self.action_dims)
        # Devolver como un array de numpy compatible con MultiDiscrete
        return np.array(unflattened_action)

