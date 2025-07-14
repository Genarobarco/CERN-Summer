#%%
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
from glob import glob
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.special import factorial

excel_path = r'C:\Users\genar\VSC code\CERN-Summer\Whole_Data.xlsx'
df = pd.read_excel(excel_path)

# New values
new_data = {
    'Element A': 'Ar',
    'Concentration A': 100,
    'Element B': 'Ar',
    'Concentration B': 0,
    'Pressure (bar)': 5,
    'Volt-Amp': '40kV40mA',
    'SV': 900,
    'SC': -3.6e-7,
    'Voltage list': '-',
    'Current list': '-',
    'Folder Path': '-',
}

# Filter to check if a row with the same Element B, Concentration B, and Pressure exists
mask = (
    (df['Element B'] == new_data['Element B']) &
    (df['Concentration B'] == new_data['Concentration B']) &
    (df['Pressure (bar)'] == new_data['Pressure (bar)'])
)

if df[mask].empty:
    # No match found, append new row
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    print("New row added.")
else:
    # Match found, update that row
    for key, value in new_data.items():
        df.loc[mask, key] = value
    print("Row updated.")

# Save back to Excel
df.to_excel(excel_path, index=False)