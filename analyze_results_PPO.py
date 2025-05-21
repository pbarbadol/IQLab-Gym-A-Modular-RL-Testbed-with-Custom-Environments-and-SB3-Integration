import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN ---
# Directorio donde guardaste los CSV descargados de TensorBoard
# Asegúrate de que esta ruta sea correcta y que los archivos estén aquí.
TENSORBOARD_CSV_DIR = "ppo_training_logs\PPO_MR_3R_8x8\PPO_1" 
# Directorio donde se guardarán los gráficos generados
PLOTS_OUTPUT_DIR = "./ppo_final_plots/" 
os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_CSV_DIR, exist_ok=True) # Crear también por si acaso

# --- FUNCIÓN DE PLOTEO ---
def plot_metric_from_csv(csv_filename, 
                         value_column_name="Value",
                         step_column_name="Step",
                         title="Título del Gráfico", 
                         ylabel="Etiqueta Eje Y", 
                         output_plot_filename="plot.png",
                         color=None,
                         linestyle='-'):
    """
    Lee un CSV descargado de TensorBoard y genera un gráfico.
    """
    csv_filepath = os.path.join(TENSORBOARD_CSV_DIR, csv_filename)
    print(f"Intentando procesar: {csv_filepath}")

    try:
        df = pd.read_csv(csv_filepath)
        if df.empty:
            print(f"  ADVERTENCIA: El archivo CSV '{csv_filename}' está vacío.")
            return
        
        if value_column_name not in df.columns:
            print(f"  ERROR: La columna de valor '{value_column_name}' no se encuentra en '{csv_filename}'. Columnas disponibles: {list(df.columns)}")
            return
        if step_column_name not in df.columns:
            print(f"  ERROR: La columna de paso '{step_column_name}' no se encuentra en '{csv_filename}'. Columnas disponibles: {list(df.columns)}")
            return

        plt.figure(figsize=(10, 5)) # Ajustado el tamaño para que no sea tan alto
        plt.plot(df[step_column_name], df[value_column_name], color=color, linestyle=linestyle, linewidth=1.5)
        
        plt.title(title, fontsize=16)
        plt.xlabel("Timesteps de Entrenamiento", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True) # Formato científico para timesteps
        plt.tight_layout() # Ajusta el layout para que todo quepa bien
        
        output_path = os.path.join(PLOTS_OUTPUT_DIR, output_plot_filename)
        plt.savefig(output_path, dpi=200) # Aumentar un poco el DPI para mejor calidad
        plt.close()
        print(f"  Gráfico guardado en: {output_path}")

    except FileNotFoundError:
        print(f"  ERROR: Archivo CSV no encontrado: {csv_filepath}")
        print(f"  Asegúrate de que el archivo '{csv_filename}' exista en la carpeta '{TENSORBOARD_CSV_DIR}'.")
    except Exception as e:
        print(f"  ERROR procesando el archivo '{csv_filename}': {e}")

# --- GENERACIÓN DE GRÁFICOS ---

print("\nGenerando gráficos para PPO desde CSVs de TensorBoard...")

# 1. Gráfico de Recompensa Promedio por Episodio
plot_metric_from_csv(
    csv_filename="ep_rew_mean.csv",
    title="Evolución de la Recompensa Promedio por Episodio (PPO)",
    ylabel="Recompensa Promedio (`ep_rew_mean`)",
    output_plot_filename="ppo_final_reward_curve.png",
    color="dodgerblue"
)

# 2. Gráfico de Longitud Promedio del Episodio
plot_metric_from_csv(
    csv_filename="ep_len_mean.csv",
    title="Evolución de la Longitud Promedio del Episodio (PPO)",
    ylabel="Longitud Promedio Episodio (`ep_len_mean`)",
    output_plot_filename="ppo_final_eplen_curve.png",
    color="forestgreen"
)

# 3. Gráfico de Tasa de Éxito
plot_metric_from_csv(
    csv_filename="success_rate.csv",
    title="Evolución de la Tasa de Éxito Promedio (PPO)",
    ylabel="Tasa de Éxito Promedio (`success_rate`)",
    output_plot_filename="ppo_final_success_rate_curve.png",
    color="crimson"
)

print("\nProceso de generación de gráficos finalizado.")
print(f"Revisa la carpeta '{PLOTS_OUTPUT_DIR}' para ver los gráficos.")
print(f"Si los gráficos no se generaron, verifica los nombres de los archivos CSV en la carpeta '{TENSORBOARD_CSV_DIR}' y en el script.")