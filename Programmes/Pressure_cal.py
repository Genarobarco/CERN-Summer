#%%

import os
import sys
import shutil
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import factorial

def lineal(x, a, b):
    return a*x+b

ruta = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\Calibracion_Presion.xlsx"

data = pd.read_excel(ruta, header=None, 
                     names = ['Pressure', 'Voltage'])

print(data)

pop, cov = curve_fit(lineal, data['Pressure'], data['Voltage'])

dom = np.linspace(0, 5000, 10000)

plt.scatter(data['Pressure'], data['Voltage'])
plt.plot(dom, lineal(dom, *pop), color = 'crimson')

presiones = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
voltajes = []

for i in presiones:

    voltajes.append(lineal(i, *pop))


columna1 = presiones
columna2 = voltajes

datos = np.column_stack((columna1, columna2))

# Usamos formatter con repr para máxima precisión
np.savetxt("VoltajesyPresiones.txt", datos, delimiter='\t', fmt='%.4f')