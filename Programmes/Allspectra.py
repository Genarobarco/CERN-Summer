#%%

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
from glob import glob
from Functions import extraer_presion, Sep_rut, Excel_value, RP, integral
import pandas as pd

err_SC_standard = 0.1e-7 #A
err_SV = 50 #v
err_lambda = 5 #nm
err_pressure = 10e-3 #bar

element_mix = 'N2' 
Mix_Integrals = {}
list_concentraciones = [0, 0.1, 0.5, 1, 5, 10, 20, 50, 100]
integration_limits = [300, 450]
candela_integral = [500, 750]
yield_error = .15

Concentracion_CF4 = 5
Presion_CF4 = 5


excel_path = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Whole_Data.xlsx"
excel_path_old = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Old_Data.xlsx"
base_path= rf'C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\{element_mix}'

data_sets = []
pressures_sets=[]
currents_sets=[]
voltages_sets=[]
titles = []
data_integral=[]

for Concentracion_N2 in list_concentraciones:

    if '.' in str(Concentracion_N2):
        Concentration_mix = str(Concentracion_N2).replace('.','-')

    else:
        Concentration_mix = Concentracion_N2


    path = f'{base_path}\{Concentration_mix}'
    pattern = os.path.join(path, '*_bar', '40kV40mA', '0V')

    rutas = glob(pattern)
    rutas_ordenadas = sorted(rutas, key=extraer_presion)

    pressures = []
    currents = []
    voltages = []
    data = []

    title = f'{Concentracion_N2}%'

    print(' ')
    print(' Concentration: ', Concentracion_N2)

    for i in rutas_ordenadas:

        Element, Concentracion, Presion, Volt_Amp, Ar_Concentration = Sep_rut(i)

        if type(Presion)==int: #ONLY TAKING INT PRESSURES
            filters = {
                'Element A': 'Ar',
                'Concentration A':  Ar_Concentration,
                'Element B': Element,
                'Concentration B': Concentracion,
                'Pressure (bar)': Presion
            }

            Saturation_current = Excel_value(filters, 'SC')
            Saturation_current_3kV = Excel_value(filters, 'C3kV')
            Saturation_volt = Excel_value(filters, 'SV')

            if Excel_value(filters, 'C3kV')==0:
                print('Current at 3kV do not exist. Using standard error of', err_SC_standard)
                err_SC = err_SC_standard

            else:
                err_SC = abs(Saturation_current_3kV - Saturation_current)
                print('Current', Excel_value(filters, 'C3kV'))
                print('Current Error: ', err_SC)

            pattern_data = os.path.join(i, 'Analized_Data', '*_AllData.txt')
            archivos = glob(pattern_data)
            
            if archivos:
                archivo = archivos[0]
                df = pd.read_csv(archivo, sep='\t', engine='python')
            else:
                print(f"No se encontró archivo en {i}")

            N_e = Saturation_current / (-1.602176634e-19)
            err_NumeroElectrones = err_SC/ (-1.602176634e-19)

            pressures.append(Presion)
            currents.append(Saturation_current)
            voltages.append(Saturation_volt)
            data.append(df)

    
    data_sets.append(data)
    pressures_sets.append(pressures)
    currents_sets.append(currents)
    voltages_sets.append(voltages)
    titles.append(title)

    print('Lista de presiones: ', pressures)
    integrales = []
    err_integral = []

    for ValPressureIndex in range(len(pressures)):
        sum, err = integral(data[ValPressureIndex], 
                       'Lambda', err_lambda,
                       'Phe', 'Err_Phe',
                       *integration_limits)
        integrales.append(sum)
        err_integral.append(sum*yield_error)

    df = pd.DataFrame({
            'Pressures': pressures,
            'integrals': integrales, 
            'Err_int': err_integral
            })

    data_integral.append(df)
    print('------------------------------------------------------------')

Ruta_Candela = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\CF4\5\5_bar\40kV40mA\After_WindowChange"

Candela_FD = RP(Ruta_Candela)
df_candela = Candela_FD['calibratedResults']

filters_candela = {
    'Element A': 'Ar',
    'Concentration A':  100 - Concentracion_CF4,
    'Element B': 'CF4',
    'Concentration B': Concentracion_CF4,
    'Pressure (bar)': Presion_CF4
}

Current_candela = Excel_value(filters_candela, 'SC')

NumElectronsCandela = Current_candela / (-1.602176634e-19)
err_NumE_candela = err_SC/ (-1.602176634e-19)

Candela_phe = df_candela['Counts'] / NumElectronsCandela
err_Candela_phe = np.sqrt((df_candela['Err_Counts']/NumElectronsCandela)**2+
                  (df_candela['Counts']*err_NumE_candela/(NumElectronsCandela)**2)**2)

df_reference = pd.DataFrame({
          'Lambda': df_candela['Lambda'],
          'Counts': df_candela['Counts'],
          'Err_Counts': np.sqrt(df_candela['Counts'].clip(lower=0)),
          'Counts_norm': df_candela['Counts']/max(df_candela['Counts']),
          'Err_Counts_norm': np.sqrt(df_candela['Counts'].clip(lower=0))/max(df_candela['Counts']),
          'Phe':Candela_phe,
          'Err_Phe': err_Candela_phe
          })

sum_candela, err_candela = integral(df_reference, 
                       'Lambda', err_lambda,
                       'Phe', 'Err_Phe',
                       *candela_integral)

err_candela = sum_candela*yield_error

print(f"Se cargaron {len(data_sets)} conjuntos de datos.")

all_pressures = set()
for df in data_integral:
    all_pressures.update(df['Pressures'].tolist())
all_pressures = sorted(all_pressures)

# Paso 2: Inicializar diccionario con listas vacías
integrales_por_presion = {p: [] for p in all_pressures}

# Paso 3: Rellenar diccionario con una entrada por concentración
for i in range(len(list_concentraciones)):
    df = data_integral[i]
    concentracion = list_concentraciones[i]

    integrales_en_df = {
        row['Pressures']: row['integrals']
        for _, row in df.iterrows()
    }

    for p in all_pressures:
        valor = integrales_en_df.get(p, None)
        integrales_por_presion[p].append(valor)


#%%

# Compartir ejes entre los subplots (eje x por columnas, eje y entre primeras 2 columnas)
fig, axs = plt.subplots(3, 3, figsize=(16, 9), sharex='col', sharey='row')

colormap = cm.turbo

# Calcular normalización global (presiones de espectros)
all_pressures = [p for subset in pressures_sets[:7] for p in subset]
norm = Normalize(vmin=min(all_pressures), vmax=max(all_pressures))


for idx in range(9):
    row = idx // 3
    col = idx % 3
    ax = axs[row, col]

    if idx < 9:
        # Espectros
        data_i = data_sets[idx]
        pressures = pressures_sets[idx]
        title = titles[idx]

        norm = Normalize(vmin=min(pressures), vmax=max(pressures))

        for i in range(len(pressures)):
            color = colormap(norm(pressures[i]))
            ax.plot(data_i[i]['Lambda'], data_i[i]['Phe'],
                    color=color,
                    linewidth=0.5)
            
            ax.fill_between(data_i[i]['Lambda'],
                            data_i[i]['Phe'] - data_i[i]['Err_Phe'],
                            data_i[i]['Phe'] + data_i[i]['Err_Phe'],
                            color=color,
                            alpha=0.5)

        ax.fill_between(df_candela['Lambda'].clip(lower=0),
                        Candela_phe - err_Candela_phe,
                        Candela_phe + err_Candela_phe,
                        label=f'Ar/CF4 {100-Concentracion_CF4}/{Concentracion_CF4} {Presion_CF4}bar {Current_candela}A',
                        color='magenta',
                        alpha=0.5)

        ax.set_title(title, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=10, labelbottom=True)
        ax.grid()
        ax.legend(fontsize=8, title_fontsize=8)
        ax.set_ylim(-0.1, 70)
        # ax.set_xlim(integration_limits[0], integration_limits[1])
        
        if row == 2:
            ax.set_xlabel(r'$\lambda$ (nm)', fontsize=10)
        if col == 0:
            ax.set_ylabel(r'Photons / electrons (A.U)', fontsize=10)

    # elif idx == 8:

    #     for idx_p, (p, y_vals) in enumerate(integrales_por_presion.items()):
    #         # Mismos colores que en tu estilo
    #         color = colormap(idx_p / len(integrales_por_presion))

    #         ax.errorbar(list_concentraciones, y_vals/sum_candela,
    #             fmt='o-', color=color,
    #             label=f'{p} bar')


# ax.set_xscale('log')
# ax.set_xlabel('Concentration of N₂ (%)', fontsize=10)
# ax.set_ylabel(r'$\gamma$/e$^-$ $\times \lambda$', fontsize=10)
# ax.tick_params(axis='both', which='major', labelsize=9)
# ax.grid(True)


fig.subplots_adjust(right=0.89, left=0.04, hspace=0.28, wspace=0.1)
cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label='Pressure (bar)')

plt.savefig('AllSpectra_uv.jpg', format='jpg',
            bbox_inches='tight', dpi=300)


#%%

# Crear figura con 1 fila y 2 columnas
fig, (ax_sc, ax_sv) = plt.subplots(1, 2, figsize=(16, 9), sharex=True)

# Colores por mezcla
colors = plt.cm.turbo(np.linspace(0, 1, len(currents_sets)))

# === Plot SC vs Pressure ===
for i, (pressures, currents, title) in enumerate(zip(pressures_sets, currents_sets, titles)):
    ax_sc.errorbar(pressures, currents,
                   yerr = err_SC, xerr = err_pressure, 
                   fmt = 'o-', color=colors[i], label=title)

ax_sc.set_xlabel('Pressure (bar)', fontsize=14)
ax_sc.set_ylabel('Saturation Current (A)', fontsize=14)
ax_sc.set_title('SC vs Pressure', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax_sc.grid(True)
ax_sc.legend(fontsize=14)

# === Plot SV vs Pressure ===
for i, (pressures, voltages, title) in enumerate(zip(pressures_sets, voltages_sets, titles)):
    ax_sv.errorbar(pressures, voltages,
                   yerr = err_SV, xerr = err_pressure,
                   fmt='s-', color=colors[i], label=title)

ax_sv.set_xlabel('Pressure (bar)', fontsize=14)
ax_sv.set_ylabel('Saturation Voltage (V)', fontsize=14)
ax_sv.set_title('SV vs Pressure', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax_sv.grid(True)
ax_sv.legend(fontsize=9)

#%%


# Crear figura separada para el plot de integrales
fig2, ax2 = plt.subplots(figsize=(16, 9))

# Escala logarítmica en el eje X
ax2.set_xscale('log')
ax2.set_yscale('log')

# Una curva por presión
for idx_p, (p, y_vals) in enumerate(integrales_por_presion.items()):
    color = colormap(idx_p / len(integrales_por_presion))

    yerr_vals = []
    for i in range(len(list_concentraciones)):
        df = data_integral[i]
        matched = df[df['Pressures'] == p]
        if not matched.empty:
            yerr_vals.append(matched['integrals'].values[0]*yield_error)
        else:
            yerr_vals.append(None)

    ax2.errorbar(list_concentraciones, y_vals / sum_candela,
                 yerr = yerr_vals/sum_candela,
                 fmt='o:', color=color,
                 label=f'{p} bar',
                 markersize = 12)

# Etiquetas y estilo
ax2.set_title('Integrals vs N₂ concentration', fontsize=20)
ax2.set_xlabel('Concentration of N₂ (%)', fontsize=20)
ax2.set_ylabel(r'Relative Yield', fontsize=20)
ax2.set_ylim(0.04)

ax2.tick_params(axis='both', which='major', labelsize=22)
ax2.grid(True, which='both')
ax2.legend(fontsize=20, title='Pressure', title_fontsize=20)

# Ajustar y mostrar
plt.tight_layout()
plt.savefig('SC_SV_vs_Pressure.jpg', dpi=300, bbox_inches='tight')
plt.show(block=False)
plt.pause(0.1)
input("Presiona ENTER para cerrar la figura...")
plt.close('all')