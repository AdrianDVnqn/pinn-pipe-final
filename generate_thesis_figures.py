import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

COLORS = {
    "pinn":             "#2563EB",   # azul
    "mass_balance_npw": "#16A34A",   # verde
    "pressure_gradient":"#D97706",   # naranja
    "lstm":             "#DC2626",   # rojo
}

MARKERS = {
    "pinn":             "o",
    "mass_balance_npw": "s",
    "pressure_gradient":"^",
    "lstm":             "D",
}

METHOD_LABELS = {
    "pinn":             "PINN (propuesta)",
    "mass_balance_npw": "Balance de masa + NPW",
    "pressure_gradient":"Gradiente de presión",
    "lstm":             "LSTM puro",
}

NOISE_ORDER = ["trivial","facil","moderado",
               "dificil","muy_dificil"]

NOISE_LABELS = {
    "trivial":     "Trivial\n(500 Pa)",
    "facil":       "Fácil\n(2k Pa)",
    "moderado":    "Moderado\n(8k Pa)",
    "dificil":     "Difícil\n(25k Pa)",
    "muy_dificil": "Muy difícil\n(50k Pa)",
}

os.makedirs("figs", exist_ok=True)

def plot_main_comparison(df, agg):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparación de métodos de detección y localización de fugas en oleoductos", fontsize=16, y=0.98)
    
    # [0,0] Error x_leak n_sensors=3
    ax = axs[0,0]
    methods = ["pinn", "mass_balance_npw", "pressure_gradient", "lstm"]
    x_pos = np.arange(len(NOISE_ORDER))
    
    max_err = 0
    for m in methods:
        sub = agg[(agg['method'] == m) & (agg['n_sensors'] == 3)]
        sub = sub.set_index('noise_level').reindex(NOISE_ORDER)
        means = sub['x_error_mean_km'].values
        stds = sub['x_error_std_km'].values
        
        ax.errorbar(x_pos, means, yerr=stds, marker=MARKERS[m], color=COLORS[m], 
                    label=METHOD_LABELS[m], capsize=5, linewidth=2, markersize=8)
        
        valid_means = [v for v in means if pd.notna(v)]
        if valid_means:
            max_err = max(max_err, max(valid_means))
            
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Umbral aceptable (0.5 km)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_ORDER])
    ax.set_ylabel("Error de localización $x_{leak}$ (km)")
    ax.set_title("Precisión de localización (3 sensores de presión)")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [0,1] Detection rate n_sensors=3
    ax = axs[0,1]
    for m in methods:
        sub = agg[(agg['method'] == m) & (agg['n_sensors'] == 3)]
        sub = sub.set_index('noise_level').reindex(NOISE_ORDER)
        rates = sub['detection_rate'].values
        ax.plot(x_pos, rates, marker=MARKERS[m], color=COLORS[m], 
                label=METHOD_LABELS[m], linewidth=2, markersize=8)

    ax.axhline(80, color='gray', linestyle='--', alpha=0.7, label='Umbral aceptable (80%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_ORDER])
    ax.set_ylabel("Tasa de detección (%)")
    ax.set_title("Tasa de detección de fugas (3 sensores de presión)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # [1,0] Error x_leak n_sensors=2
    ax = axs[1,0]
    for m in methods:
        sub = agg[(agg['method'] == m) & (agg['n_sensors'] == 2)]
        sub = sub.set_index('noise_level').reindex(NOISE_ORDER)
        means = sub['x_error_mean_km'].values
        stds = sub['x_error_std_km'].values
        
        ax.errorbar(x_pos, means, yerr=stds, marker=MARKERS[m], color=COLORS[m], 
                    label=METHOD_LABELS[m], capsize=5, linewidth=2, markersize=8)
            
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_ORDER])
    ax.set_ylabel("Error de localización $x_{leak}$ (km)")
    ax.set_title("2 sensores de presión")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # [1,1] Degradation factor
    ax = axs[1,1]
    deg_data = []
    for m in methods:
        sub = agg[(agg['method'] == m) & (agg['n_sensors'] == 3)]
        if not sub.empty:
            deg = sub['degradation_factor'].iloc[0]
            if pd.notna(deg):
                deg_data.append((m, deg))
                
    deg_data.sort(key=lambda x: x[1]) # sort best to worst (lower factor is better)
    
    y_pos = np.arange(len(deg_data))
    bars = ax.barh(y_pos, [d[1] for d in deg_data], color=[COLORS[d[0]] for d in deg_data], height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([METHOD_LABELS[d[0]] for d in deg_data])
    ax.set_xlabel("Factor de degradación (x_error muy_dificil / trivial)")
    ax.set_title("Robustez ante ruido (3 sensores)")
    
    for i, bar in enumerate(bars):
        val = bar.get_width()
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f"{val:.1f}x", 
                va='center', fontweight='bold')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.text(0.5, 0.01, "Barras de error: ±1σ sobre 12 escenarios", ha='center', fontsize=10, style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig("figs/thesis_main_comparison.png", dpi=300)
    plt.close(fig)
    return "figs/thesis_main_comparison.png"


def plot_error_heatmaps(df):
    methods = ["pinn", "mass_balance_npw", "pressure_gradient", "lstm"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Error de localización (km) por tamaño de fuga y nivel de ruido (3 sensores)", fontsize=16)
    
    for idx, m in enumerate(methods):
        ax = axs[idx // 2, idx % 2]
        sub = df[(df['method'] == m) & (df['n_sensors'] == 3) & (df['has_leak'] == True)]
        
        # map leak_size to categorical order
        size_order = ['small', 'medium', 'large']
        pivot = sub.pivot_table(values='x_leak_error_km', index='leak_size', columns='noise_level', aggfunc='mean')
        
        # Ensure all columns and rows exist
        for n in NOISE_ORDER:
            if n not in pivot.columns:
                pivot[n] = np.nan
        for s in size_order:
            if s not in pivot.index:
                pivot.loc[s] = np.nan
                
        pivot = pivot.reindex(index=size_order, columns=NOISE_ORDER)
        
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax, 
                    cbar_kws={'label': 'Error (km)'}, vmin=0, vmax=5.0)
        
        ax.set_title(METHOD_LABELS[m])
        ax.set_ylabel("Tamaño de fuga")
        ax.set_xlabel("Nivel de ruido")
        ax.set_xticklabels([NOISE_LABELS[n].replace('\n',' ') for n in NOISE_ORDER], rotation=45, ha='right')
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("figs/thesis_error_heatmap.png", dpi=300)
    plt.close(fig)
    return "figs/thesis_error_heatmap.png"


def plot_by_leak_size(df, agg):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Impacto del nivel de ruido según tamaño de la fuga (3 sensores)", fontsize=16)
    
    methods = ["pinn", "mass_balance_npw", "pressure_gradient", "lstm"]
    sizes = [('small', 'Pequeña (0.005 m³/s)'), ('medium', 'Mediana (0.015 m³/s)'), ('large', 'Grande (0.030 m³/s)')]
    x_pos = np.arange(len(NOISE_ORDER))
    
    for idx, (sz, title) in enumerate(sizes):
        ax = axs[idx]
        for m in methods:
            sub = df[(df['method'] == m) & (df['n_sensors'] == 3) & (df['leak_size'] == sz) & (df['has_leak'] == True)]
            means = []
            for n in NOISE_ORDER:
                val = sub[sub['noise_level'] == n]['x_leak_error_km'].mean()
                means.append(val)
                
            ax.plot(x_pos, means, marker=MARKERS[m], color=COLORS[m], 
                    label=METHOD_LABELS[m], linewidth=2, markersize=8)
            
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_ORDER], rotation=45)
        ax.set_ylabel("Error de localización (km)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()
            
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig("figs/thesis_by_leak_size.png", dpi=300)
    plt.close(fig)
    return "figs/thesis_by_leak_size.png"


def plot_sensor_impact(agg):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    methods = ["pinn", "mass_balance_npw", "pressure_gradient", "lstm"]
    
    # Left: x_error for 2 vs 3 sensors (moderado)
    ax = axs[0]
    bar_width = 0.35
    x_pos = np.arange(len(methods))
    
    err_3s = []
    err_2s = []
    
    for m in methods:
        sub_3s = agg[(agg['method'] == m) & (agg['n_sensors'] == 3) & (agg['noise_level'] == 'moderado')]
        sub_2s = agg[(agg['method'] == m) & (agg['n_sensors'] == 2) & (agg['noise_level'] == 'moderado')]
        
        val_3s = sub_3s['x_error_mean_km'].values[0] if not sub_3s.empty else np.nan
        val_2s = sub_2s['x_error_mean_km'].values[0] if not sub_2s.empty else np.nan
        
        err_3s.append(val_3s)
        err_2s.append(val_2s)
        
    ax.bar(x_pos - bar_width/2, err_3s, bar_width, label='3 Sensores', color='#3B82F6')
    ax.bar(x_pos + bar_width/2, err_2s, bar_width, label='2 Sensores', color='#93C5FD')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha='right')
    ax.set_ylabel("Error de localización (km)")
    ax.set_title("Comparación 3 vs 2 Sensores (Ruido Moderado)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right: Error increase
    ax = axs[1]
    increases = []
    for m, v2, v3 in zip(methods, err_2s, err_3s):
        if pd.notna(v2) and pd.notna(v3) and v3 > 0:
            inc = (v2 - v3) / v3 * 100.0
            increases.append((m, inc))
        else:
            increases.append((m, np.nan))
            
    # Sort for plot
    valid_incs = [(m, v) for m, v in increases if pd.notna(v)]
    valid_incs.sort(key=lambda x: x[1])
    
    y_pos = np.arange(len(valid_incs))
    bars = ax.barh(y_pos, [v[1] for v in valid_incs], color=[COLORS[v[0]] for v in valid_incs], height=0.6)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([METHOD_LABELS[v[0]] for v in valid_incs])
    ax.set_xlabel("Incremento del error (%)")
    ax.set_title("Degradación al perder un sensor (3 → 2)")
    
    for bar in bars:
        val = bar.get_width()
        ax.text(val + 5, bar.get_y() + bar.get_height()/2, f"+{val:.1f}%", 
                va='center', fontweight='bold')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig("figs/thesis_sensor_impact.png", dpi=300)
    plt.close(fig)
    return "figs/thesis_sensor_impact.png"


def plot_summary_table(agg):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    methods = ["pinn", "mass_balance_npw", "pressure_gradient", "lstm"]
    cols = ["Método", "Det.% Trivial", "Det.% Muy Difícil", "x_err Trivial (km)", "x_err Muy Difícil (km)", "Degradación", "Inf. time (s)"]
    
    cell_text = []
    cell_colors = []
    
    for m in methods:
        sub = agg[(agg['method'] == m) & (agg['n_sensors'] == 3)]
        if sub.empty:
            continue
            
        row = []
        colors = []
        
        row.append(METHOD_LABELS[m])
        colors.append('white') # method name
        
        # Det % trivial
        val_dt = sub[sub['noise_level'] == 'trivial']['detection_rate'].values
        val_dt = val_dt[0] if len(val_dt) > 0 else np.nan
        row.append(f"{val_dt:.0f}%" if pd.notna(val_dt) else "N/A")
        colors.append(get_color_det(val_dt))
        
        # Det % muy dificil
        val_dm = sub[sub['noise_level'] == 'muy_dificil']['detection_rate'].values
        val_dm = val_dm[0] if len(val_dm) > 0 else np.nan
        row.append(f"{val_dm:.0f}%" if pd.notna(val_dm) else "N/A")
        colors.append(get_color_det(val_dm))
        
        # x_err trivial
        val_xt = sub[sub['noise_level'] == 'trivial']['x_error_mean_km'].values
        val_xt = val_xt[0] if len(val_xt) > 0 else np.nan
        row.append(f"{val_xt:.2f}" if pd.notna(val_xt) else "N/A")
        colors.append(get_color_err(val_xt))
        
        # x_err muy_dificil
        val_xm = sub[sub['noise_level'] == 'muy_dificil']['x_error_mean_km'].values
        val_xm = val_xm[0] if len(val_xm) > 0 else np.nan
        row.append(f"{val_xm:.2f}" if pd.notna(val_xm) else "N/A")
        colors.append(get_color_err(val_xm))
        
        # Degradacion
        val_deg = sub['degradation_factor'].iloc[0]
        row.append(f"{val_deg:.1f}x" if pd.notna(val_deg) else "N/A")
        colors.append(get_color_deg(val_deg))
        
        # Inference time
        val_inf = sub['inference_time_s'].mean()
        row.append(f"{val_inf:.2f}" if pd.notna(val_inf) else "N/A")
        colors.append('white')
        
        cell_text.append(row)
        cell_colors.append(colors)
        
    table = ax.table(cellText=cell_text, colLabels=cols, cellColours=cell_colors, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Bold headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E5E7EB')
            
    fig.savefig("figs/thesis_summary_table.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    return "figs/thesis_summary_table.png"

def get_color_det(val):
    if pd.isna(val): return 'white'
    if val >= 95: return '#86EFAC'
    if val >= 80: return '#FEF08A'
    return '#FCA5A5'

def get_color_err(val):
    if pd.isna(val): return 'white'
    if val <= 0.5: return '#86EFAC'
    if val <= 1.5: return '#FEF08A'
    return '#FCA5A5'
    
def get_color_deg(val):
    if pd.isna(val): return 'white'
    if val <= 2.0: return '#86EFAC'
    if val <= 5.0: return '#FEF08A'
    return '#FCA5A5'


if __name__ == '__main__':
    if not os.path.exists("results/master_results.csv"):
        print("Falta results/master_results.csv")
        print("Correr primero: python run_factorial_experiment.py")
        sys.exit(1)

    df  = pd.read_csv("results/master_results.csv")
    agg = pd.read_csv("results/aggregate_metrics.csv")

    print("Generando figuras para la tesis...")

    plot_main_comparison(df, agg)
    plot_error_heatmaps(df)
    plot_by_leak_size(df, agg)
    plot_sensor_impact(agg)
    plot_summary_table(agg)

    print("Figuras guardadas en figs/:")
    for f in ["thesis_main_comparison.png",
              "thesis_error_heatmap.png",
              "thesis_by_leak_size.png",
              "thesis_sensor_impact.png",
              "thesis_summary_table.png"]:
        print(f"  ✓ figs/{f}")
