#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_excel(r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Current_ArN2_9010_5bar.xlsx")


plt.figure(figsize=(16,9))
plt.errorbar(data['Voltaje'], -data['Corriente'], color='crimson',
             fmt = 'o:')
plt.vlines(x=900, ymin=1.5e-7, ymax=4.25e-7, colors= 'navy')
plt.xlabel('Voltage (V)', fontsize=15)
plt.ylabel('Current (A)', fontsize=15)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=18)

plt.savefig(f'Current_9010.jpg', format='jpg', bbox_inches='tight')
plt.show()