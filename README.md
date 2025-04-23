# Trabajo final de máster: Q-Learning Multi-Robot en un Entorno de Tablero

Autor: **Pablo Barbado Lozano**  
Fecha: **23 de abril de 2025**

---

## Descripción

Este proyecto implementa un entorno personalizado tipo **gridworld** donde múltiples **robots colaboran** para explorar celdas en un tablero. Se aplica **Q-Learning independiente por agente** para que los robots aprendan a moverse, colaborar y terminar la tarea eficientemente.

El entorno, la lógica de recompensa y la política de entrenamiento están totalmente desarrollados en Python, con una representación textual del entorno para depuración y evaluación.

---

## Estructura del Proyecto

proyecto/
├── config.py               # Configuración general y de entrenamiento
├── tablero.py              # Lógica del entorno del tablero y movimientos
├── q_learning_agent.py     # Lógica del agente Q-learning y recompensas
├── train.py                # Script principal de entrenamiento y evaluación
├── q_tables_final.pkl      # (opcional) Q-tables aprendidas guardadas
├── training_rewards.png    # (opcional) Gráfico de recompensas
└── README.md               # Este archivo


---

## ⚙️ Configuración rápida

Antes de ejecutar, asegúrate de tener instaladas las dependencias necesarias:

```bash
pip install matplotlib numpy
```

---

## Cómo ejecutar

Para entrenar a los agentes y evaluar la política final:

```bash
python train.py
```

Esto entrenará durante `NUM_EPISODIOS` episodios y, si está activado en `config.py`, generará:
- Un gráfico de recompensas `training_rewards.png`
- Una evaluación visual con `render()` del tablero

---

## Detalles técnicos

- **Entorno:** `Tablero` con tamaño configurable (`FILAS x COLUMNAS`)
- **Robots:** `N_ROBOTS` se entrenan simultáneamente
- **Aprendizaje:** Q-Learning con política epsilon-greedy por robot
- **Estados:** Posiciones de los robots + celdas visitadas + robots activos
- **Acciones posibles:** arriba, abajo, izquierda, derecha y terminar

---

## Recompensas definidas

- `STEP_PENALTY`: penalización por cada paso (-1)
- `NEW_CELL_REWARD`: recompensa al descubrir nuevas celdas (+20)
- `TERMINATE_EARLY_PENALTY`: castigo si terminan antes de completar (-5)
- `GOAL_COMPLETED_REWARD`: bonificación al completar todo (+20)
- Y más... (ver `q_learning_agent.py`)

---

## Notas

- El diseño busca un **entorno simple pero suficientemente complejo** para demostrar el aprendizaje multi-agente independiente.
- Se respeta el principio de separación: el entorno no impone objetivos; solo define reglas físicas. Las recompensas definen los objetivos.

---

## Ejemplo de entrenamiento

```bash
Ep: 49500/100000 | Pasos: 16 | Done: True | Eps: 0.0100 | Rec Reciente: 471.071 | T: 13.6s / ~0.5m
Ep: 49600/100000 | Pasos: 16 | Done: True | Eps: 0.0100 | Rec Reciente: 460.685 | T: 13.7s / ~0.5m
Ep: 49700/100000 | Pasos: 16 | Done: True | Eps: 0.0100 | Rec Reciente: 470.581 | T: 13.7s / ~0.5m
```

## Ejemplo de resultado
Este ejemplo es con una matriz 5x5, con 2 robots, inicalizados en la misma posición:
```bash

--- Evaluación de la Política Aprendida (Epsilon = 0) ---
-----------
|+| | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
-----------
Visitadas: 1/25 | Activos: [True, True]

Paso 1 | Acciones: [1, 3] | Recompensas: [39.00, 39.00]
-----------
|.|1| | | |
|0| | | | |
| | | | | |
| | | | | |
| | | | | |
-----------
Visitadas: 3/25 | Activos: [True, True]

Paso 2 | Acciones: [3, 3] | Recompensas: [39.00, 39.00]
-----------
|.|.|1| | |
|.|0| | | |
| | | | | |
| | | | | |
| | | | | |
-----------
Visitadas: 5/25 | Activos: [True, True]

Paso 3 | Acciones: [1, 3] | Recompensas: [39.00, 39.00]
-----------
|.|.|.|1| |
|.|.| | | |
| |0| | | |
| | | | | |
| | | | | |
-----------
Visitadas: 7/25 | Activos: [True, True]

Paso 4 | Acciones: [3, 3] | Recompensas: [39.00, 39.00]
-----------
|.|.|.|.|1|
|.|.| | | |
| |.|0| | |
| | | | | |
| | | | | |
-----------
Visitadas: 9/25 | Activos: [True, True]

Paso 5 | Acciones: [1, 1] | Recompensas: [39.00, 39.00]
-----------
|.|.|.|.|.|
|.|.| | |1|
| |.|.| | |
| | |0| | |
| | | | | |
-----------
Visitadas: 11/25 | Activos: [True, True]

Paso 6 | Acciones: [1, 2] | Recompensas: [39.00, 39.00]
-----------
|.|.|.|.|.|
|.|.| |1|.|
| |.|.| | |
| | |.| | |
| | |0| | |
-----------
Visitadas: 13/25 | Activos: [True, True]

Paso 7 | Acciones: [3, 3] | Recompensas: [19.00, 19.00]
-----------
|.|.|.|.|.|
|.|.| |.|1|
| |.|.| | |
| | |.| | |
| | |.|0| |
-----------
Visitadas: 14/25 | Activos: [True, True]

Paso 8 | Acciones: [3, 1] | Recompensas: [39.00, 39.00]
-----------
|.|.|.|.|.|
|.|.| |.|.|
| |.|.| |1|
| | |.| | |
| | |.|.|0|
-----------
Visitadas: 16/25 | Activos: [True, True]

Paso 9 | Acciones: [0, 2] | Recompensas: [39.00, 39.00]
-----------
|.|.|.|.|.|
|.|.| |.|.|
| |.|.|1|.|
| | |.| |0|
| | |.|.|.|
-----------
Visitadas: 18/25 | Activos: [True, True]

Paso 10 | Acciones: [2, 3] | Recompensas: [19.00, 19.00]
-----------
|.|.|.|.|.|
|.|.| |.|.|
| |.|.|.|1|
| | |.|0|.|
| | |.|.|.|
-----------
Visitadas: 19/25 | Activos: [True, True]

Paso 11 | Acciones: [2, 2] | Recompensas: [-1.00, -1.00]
-----------
|.|.|.|.|.|
|.|.| |.|.|
| |.|.|1|.|
| | |0|.|.|
| | |.|.|.|
-----------
Visitadas: 19/25 | Activos: [True, True]

Paso 12 | Acciones: [2, 2] | Recompensas: [19.00, 19.00]
-----------
|.|.|.|.|.|
|.|.| |.|.|
| |.|1|.|.|
| |0|.|.|.|
| | |.|.|.|
-----------
Visitadas: 20/25 | Activos: [True, True]

Paso 13 | Acciones: [1, 0] | Recompensas: [39.00, 39.00]
-----------
|.|.|.|.|.|
|.|.|1|.|.|
| |.|.|.|.|
| |.|.|.|.|
| |0|.|.|.|
-----------
Visitadas: 22/25 | Activos: [True, True]

Paso 14 | Acciones: [2, 1] | Recompensas: [19.00, 19.00]
-----------
|.|.|.|.|.|
|.|.|.|.|.|
| |.|1|.|.|
| |.|.|.|.|
|0|.|.|.|.|
-----------
Visitadas: 23/25 | Activos: [True, True]

Paso 15 | Acciones: [0, 0] | Recompensas: [19.00, 19.00]
-----------
|.|.|.|.|.|
|.|.|1|.|.|
| |.|.|.|.|
|0|.|.|.|.|
|.|.|.|.|.|
-----------
Visitadas: 24/25 | Activos: [True, True]

Paso 16 | Acciones: [0, 3] | Recompensas: [39.00, 39.00]
-----------
|.|.|.|.|.|
|.|.|.|1|.|
|0|.|.|.|.|
|.|.|.|.|.|
|.|.|.|.|.|
-----------
Visitadas: 25/25 | Activos: [True, True]

--- Fin Evaluación ---
Episodio terminado en 16 pasos. Done=True
Recompensa total acumulada en evaluación: [484. 484.] (Promedio: 484.00)
Celdas visitadas: 25/25
```
