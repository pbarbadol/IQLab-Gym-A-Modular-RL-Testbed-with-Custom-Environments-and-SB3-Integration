# analyze_results.py (Versión Completa y Mejorada)
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np

# --- Configuración ---
BASE_RESULTS_DIR = "experiment_battery_main"
AGGREGATED_PLOTS_DIR = os.path.join(BASE_RESULTS_DIR, "aggregated_plots")
METRICS_FILENAME = "training_metrics.csv" # Nombre del archivo de métricas
GROUP_CONFIG_FILENAME = "experiment_group_config_summary.json" # Nombre del archivo de config del grupo

# --- Funciones Auxiliares de Carga y Agregación (con más depuración) ---

def load_metrics_for_group(group_path: str, metrics_csv_filename: str) -> list[pd.DataFrame]:
    print(f"  Intentando cargar métricas para el grupo en: {group_path}")
    all_trial_metrics = []
    if not os.path.isdir(group_path):
        print(f"    ADVERTENCIA: El directorio del grupo '{group_path}' no existe.")
        return all_trial_metrics

    for item in os.listdir(group_path):
        item_path = os.path.join(group_path, item)
        if os.path.isdir(item_path) and item.startswith("trial_"):
            # print(f"    Encontrado directorio de trial: {item}") # Puede ser muy verboso
            metrics_file = os.path.join(item_path, metrics_csv_filename)
            if os.path.exists(metrics_file):
                # print(f"      Encontrado archivo de métricas: {metrics_file}")
                try:
                    df = pd.read_csv(metrics_file)
                    if df.empty:
                        print(f"        ADVERTENCIA: El archivo de métricas {metrics_file} está VACÍO.")
                    else:
                        # print(f"        Métricas cargadas desde {metrics_file}. Filas: {len(df)}, Columnas: {list(df.columns)}")
                        all_trial_metrics.append(df)
                except pd.errors.EmptyDataError:
                    print(f"        ADVERTENCIA: Error al leer {metrics_file} - Archivo vacío (EmptyDataError).")
                except Exception as e:
                    print(f"        ERROR al cargar {metrics_file}: {e}")
            # else:
                # print(f"      ADVERTENCIA: No se encontró '{metrics_csv_filename}' en {item_path}")
    num_loaded = len(all_trial_metrics)
    print(f"  Cargados {num_loaded} dataframes de métricas para el grupo '{os.path.basename(group_path)}'.")
    return all_trial_metrics

def aggregate_group_metrics(list_of_dfs: list[pd.DataFrame], metric_cols_to_agg: list[str]):
    # print(f"    Intentando agregar {len(list_of_dfs)} DataFrames para las columnas: {metric_cols_to_agg}")
    if not list_of_dfs:
        # print("      No hay DataFrames para agregar. Retornando None.")
        return None, None

    processed_dfs_for_concat = []
    for i, df_trial in enumerate(list_of_dfs):
        if df_trial.empty:
            # print(f"      DataFrame del trial {i} está vacío. Saltando.")
            continue
        if 'episodio' not in df_trial.columns:
            print(f"      ADVERTENCIA (Agregación): Columna 'episodio' no encontrada en DataFrame del trial {i}. Columnas: {list(df_trial.columns)}. Saltando este trial.")
            continue
        
        # Seleccionar solo 'episodio' y las columnas métricas que realmente existen en este df_trial
        actual_metric_cols_in_df = [m_col for m_col in metric_cols_to_agg if m_col in df_trial.columns]
        if not actual_metric_cols_in_df:
            # print(f"      ADVERTENCIA (Agregación): Ninguna de las columnas métricas especificadas se encontró en el trial {i}. Saltando.")
            continue
            
        cols_to_keep_for_this_df = ['episodio'] + actual_metric_cols_in_df
        processed_dfs_for_concat.append(df_trial[cols_to_keep_for_this_df].set_index('episodio'))

    if not processed_dfs_for_concat:
        # print("      No hay DataFrames válidos para concatenar después del preprocesamiento.")
        return None, None

    try:
        # Usar outer join para manejar diferentes longitudes de episodio (rellenará con NaN)
        concatenated_df = pd.concat(processed_dfs_for_concat, axis=1, keys=[f'trial_{i}' for i in range(len(processed_dfs_for_concat))], join='outer')
        
        aggregated_results = {}
        for col_to_agg in metric_cols_to_agg: # Iterar sobre las columnas que QUERÍAMOS agregar
            try:
                # Seleccionar todas las apariciones de esta columna métrica en los diferentes trials
                metric_df_trials = concatenated_df.xs(col_to_agg, level=1, axis=1) # level=1 es el nombre de la métrica original
                if metric_df_trials.empty:
                    # print(f"        Métrica '{col_to_agg}' resultó en un DataFrame vacío después de .xs().")
                    continue
                # mean() y std() sobre axis=1 (columnas de trials) ignorarán NaNs por defecto
                aggregated_results[f'{col_to_agg}_mean'] = metric_df_trials.mean(axis=1)
                aggregated_results[f'{col_to_agg}_std'] = metric_df_trials.std(axis=1)
                # Opcional: min/max si se desea
                # aggregated_results[f'{col_to_agg}_min'] = metric_df_trials.min(axis=1)
                # aggregated_results[f'{col_to_agg}_max'] = metric_df_trials.max(axis=1)
            except KeyError:
                # print(f"        ADVERTENCIA (Agregación): Métrica '{col_to_agg}' no encontrada consistentemente en los trials para agregar (KeyError en .xs).")
                pass # Continuar con otras métricas
            except Exception as e_xs:
                 print(f"        Error procesando métrica '{col_to_agg}' después de .xs(): {e_xs}")
        
        if not aggregated_results:
            # print("      Agregación no produjo resultados (aggregated_results está vacío).")
            return None, concatenated_df

        final_df = pd.DataFrame(aggregated_results)
        if final_df.empty:
            # print("      El DataFrame final agregado (final_df) está vacío.")
            return None, concatenated_df
        
        # print("      Agregación de series temporales completada.")
        return final_df.reset_index(), concatenated_df
    
    except Exception as e:
        print(f"      ERROR MAYOR durante la agregación de métricas: {e}.")
        return None, None

# --- Funciones de Ploteo ---

def plot_comparison(groups_data_to_plot: dict,
                    metric_to_plot: str,
                    ylabel: str,
                    title: str,
                    filename: str,
                    show_std_fill: bool = True,
                    std_alpha: float = 0.15,
                    line_styles: list | None = None,
                    colors: list | None = None):
    """Genera un gráfico comparando una métrica específica entre un subconjunto de grupos."""
    print(f"    Generando gráfico comparativo: {title} -> {filename}")
    plt.figure(figsize=(12, 7))
    
    has_plotted_data = False
    
    if line_styles is None:
        line_styles = ['-', '--', '-.', ':'] * (len(groups_data_to_plot) // 4 + 1)
    if colors is None:
        # Usar el ciclo de colores por defecto de Matplotlib si no se especifica
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = [prop_cycle.by_key()['color'][i % len(prop_cycle.by_key()['color'])] for i in range(len(groups_data_to_plot))]


    for i, (group_name, group_df_ts) in enumerate(groups_data_to_plot.items()):
        if group_df_ts is None or not isinstance(group_df_ts, pd.DataFrame):
            # print(f"      Saltando grupo '{group_name}' para '{title}': no es un DataFrame válido.")
            continue
        if f'{metric_to_plot}_mean' not in group_df_ts.columns:
            # print(f"      Saltando grupo '{group_name}' para '{title}': métrica '{metric_to_plot}_mean' no encontrada.")
            continue
        if 'episodio' not in group_df_ts.columns:
            # print(f"      Saltando grupo '{group_name}' para '{title}': no tiene columna 'episodio'.")
            continue

        episodes = group_df_ts['episodio']
        mean_metric = group_df_ts[f'{metric_to_plot}_mean']
        
        current_color = colors[i % len(colors)]
        current_linestyle = line_styles[i % len(line_styles)]

        line, = plt.plot(episodes, mean_metric, label=f"{group_name}", 
                         color=current_color, linestyle=current_linestyle, linewidth=1.5)
        has_plotted_data = True

        if show_std_fill:
            std_metric = group_df_ts.get(f'{metric_to_plot}_std')
            if std_metric is not None and not std_metric.isnull().all():
                plt.fill_between(episodes, mean_metric - std_metric, mean_metric + std_metric,
                                 alpha=std_alpha, color=line.get_color())

    if not has_plotted_data:
        print(f"      ADVERTENCIA: No se plotearon datos para '{title}'. El gráfico estará vacío.")
    
    plt.title(title, fontsize=16)
    plt.xlabel('Episodio', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='best', fontsize='small', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(AGGREGATED_PLOTS_DIR, filename), dpi=150) # Mejorar resolución
        # print(f"      Gráfico comparativo guardado: {filename}")
    except Exception as e_save:
        print(f"      ERROR guardando gráfico '{filename}': {e_save}")
    plt.close()

def plot_final_metric_bars(final_metrics_data: dict,
                           metric_key: str,
                           title: str,
                           ylabel: str,
                           filename: str,
                           sort_bars: bool = False,
                           higher_is_better: bool = True):
    """Genera un gráfico de barras para métricas finales."""
    print(f"    Generando gráfico de barras: {title} -> {filename}")
    group_names_for_bar = []
    metric_values_for_bar = []

    for group_name, metrics_series in final_metrics_data.items():
        if metrics_series is not None and metric_key in metrics_series:
            group_names_for_bar.append(group_name)
            metric_values_for_bar.append(metrics_series[metric_key])
        # else:
            # print(f"      Métrica '{metric_key}' no encontrada para el grupo '{group_name}' en datos finales.")

    if not group_names_for_bar:
        print(f"      No hay datos para el gráfico de barras '{title}'.")
        return

    # Ordenar si se solicita
    if sort_bars:
        zipped_pairs = zip(metric_values_for_bar, group_names_for_bar)
        sorted_pairs = sorted(zipped_pairs, reverse=higher_is_better)
        metric_values_for_bar = [pair[0] for pair in sorted_pairs]
        group_names_for_bar = [pair[1] for pair in sorted_pairs]
        
    plt.figure(figsize=(max(10, len(group_names_for_bar) * 0.9), 7)) # Ajustar tamaño
    bars = plt.bar(group_names_for_bar, metric_values_for_bar, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(group_names_for_bar))))
    
    plt.title(title, fontsize=16)
    plt.xlabel('Grupo de Experimento', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize='small')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(metric_values_for_bar)*0.01 if metric_values_for_bar else 0.01),
                 f'{yval:.3f}', ha='center', va='bottom', fontsize='x-small', bbox = dict(facecolor = 'white', alpha = .5,edgecolor='white',pad=0))


    try:
        plt.savefig(os.path.join(AGGREGATED_PLOTS_DIR, filename), dpi=150)
        # print(f"      Gráfico de barras guardado: {filename}")
    except Exception as e_save_bar:
        print(f"      ERROR guardando gráfico de barras '{filename}': {e_save_bar}")
    plt.close()

def generate_summary_table(all_groups_final_metrics: dict, output_filename="final_performance_summary.csv"):
    """Genera una tabla CSV resumiendo las métricas finales."""
    print(f"    Generando tabla resumen: {output_filename}")
    summary_list = []
    for group_name, metrics_series in all_groups_final_metrics.items():
        if metrics_series is not None and not metrics_series.empty:
            data = metrics_series.copy()
            data['experiment_group'] = group_name
            summary_list.append(data)
    
    if not summary_list:
        print("      No hay datos para generar la tabla resumen.")
        return

    summary_df = pd.DataFrame(summary_list)
    if 'experiment_group' in summary_df.columns:
        cols = ['experiment_group'] + [col for col in summary_df.columns if col != 'experiment_group']
        summary_df = summary_df[cols]
    else:
        print("      ADVERTENCIA: 'experiment_group' no está en las columnas del DataFrame resumen.")

    filepath = os.path.join(BASE_RESULTS_DIR, output_filename)
    try:
        summary_df.to_csv(filepath, index=False, float_format='%.4f')
        print(f"      Tabla resumen guardada en: {filepath}")
    except Exception as e_save_csv:
        print(f"      ERROR guardando tabla resumen '{filepath}': {e_save_csv}")

# --- Script Principal ---
if __name__ == "__main__":
    print(f"--- Iniciando Análisis de Resultados en '{BASE_RESULTS_DIR}' ---")
    os.makedirs(AGGREGATED_PLOTS_DIR, exist_ok=True)

    experiment_groups_found = [d for d in os.listdir(BASE_RESULTS_DIR) if os.path.isdir(os.path.join(BASE_RESULTS_DIR, d)) and d != os.path.basename(AGGREGATED_PLOTS_DIR)]
    print(f"Grupos de experimentos encontrados: {experiment_groups_found if experiment_groups_found else 'Ninguno'}")

    if not experiment_groups_found:
        print("No se encontraron directorios de grupos de experimentos para analizar. Saliendo.")
        exit()
        
    all_aggregated_ts_metrics = {} # Para series temporales
    all_final_perf_metrics = {}    # Para valores finales (usados en barras y tabla)

    metrics_to_analyze_and_plot = ['recompensa_promedio', 'pasos_por_episodio', 'porcentaje_visitadas', 'tasa_exito_100ep', 'td_error_promedio']
    print(f"Métricas a analizar: {metrics_to_analyze_and_plot}")

    for group_name_iter in experiment_groups_found:
        print(f"\nProcesando grupo: {group_name_iter}")
        current_group_path = os.path.join(BASE_RESULTS_DIR, group_name_iter)
        
        group_config_path = os.path.join(current_group_path, GROUP_CONFIG_FILENAME)
        if os.path.exists(group_config_path):
            with open(group_config_path, 'r') as f_cfg:
                # group_config_data = json.load(f_cfg) # Cargar si se necesita para anotar
                pass
        
        trial_dfs_list = load_metrics_for_group(current_group_path, METRICS_FILENAME)
        
        if not trial_dfs_list:
            print(f"  No se cargaron métricas para el grupo '{group_name_iter}'. Saltando este grupo.")
            continue

        # Agregación de series temporales
        agg_ts_df, _ = aggregate_group_metrics(trial_dfs_list, metrics_to_analyze_and_plot)
        if agg_ts_df is not None and not agg_ts_df.empty:
             all_aggregated_ts_metrics[group_name_iter] = agg_ts_df
        # else:
            # print(f"    Fallo al agregar métricas de series temporales para '{group_name_iter}' o el resultado fue vacío.")

        # Agregación para métricas de rendimiento final
        current_group_final_metrics_list = []
        for i_df, df_trial_iter in enumerate(trial_dfs_list):
            if not df_trial_iter.empty:
                cols_for_final_mean = [m_col for m_col in metrics_to_analyze_and_plot if m_col in df_trial_iter.columns]
                if not cols_for_final_mean: continue

                num_rows = len(df_trial_iter)
                # Usar una ventana más pequeña si hay pocos episodios, pero al menos 1
                window_for_final_mean = min(max(1, num_rows // 10), 100) if num_rows > 10 else num_rows 
                
                if num_rows > 0 :
                    current_group_final_metrics_list.append(df_trial_iter[cols_for_final_mean].iloc[-window_for_final_mean:].mean())
        
        if current_group_final_metrics_list:
            final_perf_df_for_this_group = pd.DataFrame(current_group_final_metrics_list)
            if not final_perf_df_for_this_group.empty:
                all_final_perf_metrics[group_name_iter] = final_perf_df_for_this_group.mean()
        # else:
            # print(f"    No se pudieron calcular métricas de rendimiento final para '{group_name_iter}'.")

    # --- Generación de Gráficos y Tabla ---
    print("\n--- Generando Gráficos y Tabla Resumen ---")

    # Definir subconjuntos de grupos para gráficos enfocados
    baseline_group_name = next((name for name in all_aggregated_ts_metrics if "Baseline" in name), None)
    
    robot_scal_groups = {k: v for k, v in all_aggregated_ts_metrics.items() if "Scalability_Robots" in k or k == baseline_group_name}
    grid_scal_groups  = {k: v for k, v in all_aggregated_ts_metrics.items() if "Scalability_Grid" in k or k == baseline_group_name}
    init_pos_groups   = {k: v for k, v in all_aggregated_ts_metrics.items() if "InitialPos" in k or k == baseline_group_name}

    # Colores y estilos (puedes personalizarlos más)
    # colors_rb = plt.cm.tab10.colors # Ejemplo de un mapa de colores
    # linestyles_rb = ['-', '--', '-.', ':']

    # 1. Gráficos de Curvas Enfocados
    if robot_scal_groups:
        plot_comparison(robot_scal_groups, 'porcentaje_visitadas', '% Celdas Visitadas', 'Cobertura vs. Nº Robots', 'comp_visit_n_robots.png')
        plot_comparison(robot_scal_groups, 'recompensa_promedio', 'Recompensa Promedio', 'Recompensa vs. Nº Robots', 'comp_recomp_n_robots.png')
    if grid_scal_groups:
        plot_comparison(grid_scal_groups, 'porcentaje_visitadas', '% Celdas Visitadas', 'Cobertura vs. Tamaño Grid', 'comp_visit_grid_size.png', std_alpha=0.1)
    if init_pos_groups:
        plot_comparison(init_pos_groups, 'recompensa_promedio', 'Recompensa Promedio', 'Recompensa vs. Pos. Inicial', 'comp_recomp_init_pos.png', show_std_fill=False)

    # Gráfico general de una métrica clave, quizás solo medias para claridad
    if all_aggregated_ts_metrics:
         plot_comparison(all_aggregated_ts_metrics, 'porcentaje_visitadas', '% Celdas Visitadas',
                         'Visión General Cobertura (Medias y ±Std)', 'overview_visitadas_all_groups.png',
                         std_alpha=0.1) # Un alfa bajo para el general

    # 2. Gráficos de Barras para Métricas Finales
    if all_final_perf_metrics:
        plot_final_metric_bars(all_final_perf_metrics, 'porcentaje_visitadas', 'Cobertura Final Promedio', '% Celdas Visitadas', 'bar_final_visitadas.png', sort_bars=True, higher_is_better=True)
        plot_final_metric_bars(all_final_perf_metrics, 'recompensa_promedio', 'Recompensa Final Promedio', 'Recompensa Promedio', 'bar_final_recompensa.png', sort_bars=True, higher_is_better=True)
        plot_final_metric_bars(all_final_perf_metrics, 'tasa_exito_100ep', 'Tasa de Éxito Final Promedio', 'Tasa de Éxito (%)', 'bar_final_exito.png', sort_bars=True, higher_is_better=True)
        plot_final_metric_bars(all_final_perf_metrics, 'pasos_por_episodio', 'Pasos Medios Finales por Episodio', 'Pasos Promedio', 'bar_final_pasos.png', sort_bars=True, higher_is_better=False) # Menos pasos es mejor

    # 3. Tabla Resumen
    if all_final_perf_metrics:
        generate_summary_table(all_final_perf_metrics, "final_performance_summary.csv")

    print("\n--- ANÁLISIS DE RESULTADOS COMPLETADO ---")