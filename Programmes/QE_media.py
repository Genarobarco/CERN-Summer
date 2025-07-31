#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

plt.close('all')

path_R6 = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\QE_Cameras\QE_Retiga_R6.csv"
path_OQ = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Spectra_2025_Pablo_Raul_Genaro\QE_Cameras\QE_OQ.csv"
#We take secondary scintillation of Ar/CF4 at 99-1 as standard light emission
path_spectrum = r"C:\Users\genar\Documents\CERN Summer 2025\Carpeta para CERNbox\Sara\SaraArCF499-1-0_5bar\Secondary\FLMS035211__0__20-27-16-527.txt"

df_R6 = pd.read_csv(path_R6, header=None, delimiter=';', decimal=',')
df_OQ = pd.read_csv(path_OQ, header=None, delimiter=';', decimal=',')
df_spectrum = pd.read_csv(path_spectrum, sep='\t', 
                          names = ['wavelength', 'intensity'])

print(df_spectrum)

#Ploteamos la eficiencia cuántica de la Retiga R6
wavelength_R6 = df_R6[0]
QE_R6 = df_R6[1]


plt.figure()
plt.plot(wavelength_R6, QE_R6, '.')
plt.xlabel('wavelength (nm)'); plt.ylabel('QE')
plt.grid()
plt.title('Quantum Efficiency of Retiga R6 camera')


#Ploteamos la eficiencia cuántica de la Orca Quest 2
wavelength_OQ = df_OQ[0]
QE_OQ = df_OQ[1]


plt.figure()
plt.plot(wavelength_OQ, QE_OQ, '.')
plt.xlabel('wavelength (nm)'); plt.ylabel('QE')
plt.grid()
plt.title('Quantum Efficiency of Orca Quest 2 camera')

#%%

#Ploteamos el espectro de Ar/CF4
wavelength_spectrum = df_spectrum['wavelength']
intensity_spectrum = df_spectrum['intensity']

integral = trapezoid(intensity_spectrum, wavelength_spectrum)

#Normalizamos la integral del espectro a 1
intensity_norm = intensity_spectrum /  integral

plt.figure()
plt.plot(wavelength_spectrum, intensity_norm )
plt.grid()
plt.title('Standard ligth emission')


#Hacemos un interpolador  para ver la pdf del espectro de luz

spectrum_interp = interp1d(wavelength_spectrum, intensity_norm, bounds_error=False, fill_value='extrapolate')
wavelength_plot = np.linspace(wavelength_spectrum.min(), wavelength_spectrum.max(), 5000)

plt.plot(wavelength_plot, spectrum_interp(wavelength_plot) )
plt.xlabel('wavelength (nm)'); plt.ylabel('Normalised intensity')


QE_R6_media = sum(QE_R6 * spectrum_interp(wavelength_R6) )
QE_OQ_media = sum(QE_OQ * spectrum_interp(wavelength_OQ) )

print(f'Mean quantum efficiency for Retiga R6: {QE_R6_media / len(wavelength_R6)}')
print(f'Mean quantum efficiency for Orca Quest 2: {QE_OQ_media / len(wavelength_OQ)}')



#Quitamos el pico del agua
wavelength_water_peak = np.concatenate([wavelength_spectrum[253 - 10 : 253], wavelength_spectrum[323 : 323 + 10]])
intensity_water_peak = np.concatenate([intensity_spectrum[253 - 10 : 253], intensity_spectrum[323 : 323 + 10]])

def recta(x, a, b):
    return a * x + b 

popt, pcov = curve_fit(recta, wavelength_water_peak, intensity_water_peak, p0 = (-0.5, 0.05))

intensity_corrected = np.copy(intensity_spectrum)
intensity_corrected[253 : 323] = recta(wavelength_spectrum[253 : 323], *popt)

integral = trapezoid(intensity_corrected, wavelength_spectrum)

intensity_corrected_norm = intensity_corrected / integral


plt.figure()
plt.plot(wavelength_spectrum, intensity_corrected_norm)
plt.xlabel('wavelength (nm)'); plt.ylabel('Normalised intensity')
plt.grid()
plt.title('Light emission without water')


spectrum_interp_corrected = interp1d(wavelength_spectrum, intensity_corrected_norm, bounds_error=False, fill_value='extrapolate')

plt.plot(wavelength_plot, spectrum_interp_corrected(wavelength_plot) )


QE_R6_media_corrected = sum(QE_R6 * spectrum_interp_corrected(wavelength_R6) )
QE_OQ_media_corrected = sum(QE_OQ * spectrum_interp_corrected(wavelength_OQ) )


print(f'Mean quantum efficiency for Retiga R6 without water peak: {QE_R6_media_corrected / len(wavelength_R6)}')
print(f'Mean quantum efficiency for Orca Quest 2 without water peak: {QE_OQ_media_corrected / len(wavelength_OQ)}')