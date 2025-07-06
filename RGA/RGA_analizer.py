#%%

import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

filename = 'HeatingNight_WaterSearch'

Ruta = rf'C:\Users\genar\VSC code\CERN-Summer\RGA\{filename}.txt'

f_hour=[00,00,00]
s_hour=[18,00,00]

number_ticks=10

def RGA_spec(Ruta, f_hour, s_hour, number_ticks, 
             save=False, save_name='RGa_nombre'):
    
    names = ['Time','Nitrogene', 'Oxygen', 'Water', 'Argon', 'CF4', 'None']
    data = pd.read_csv(Ruta, sep=',', skiprows=27, 
                       header=None, engine='python', 
                       names=names)

    # Convert Time (assumed in seconds) to timedelta objects
    time_seconds = data['Time']
    time_labels = [str(timedelta(seconds=int(t))) for t in time_seconds]

    Element = ['Nitrogene', 'Oxygen', 'Water', 'Argon', 'CF4']
    color_element= ['red', 'blue', 'green', 'lightblue', 'black']
    total_pressure = data['Argon'] + data['CF4']

    # Hour Limits in format hh:mm:ss

    f_value=3600*f_hour[0]+60*f_hour[1]+f_hour[2]
    s_value=3600*s_hour[0]+60*s_hour[1]+s_hour[2]

    # Plot

    plt.figure(figsize=(12, 8))

    indice=0
    for i in Element: 
        plt.plot(time_seconds, data[i]*1.33322, 
                color=color_element[indice],label=i, linewidth=2.5)
        indice+=1

    plt.title(filename, fontsize=20)
    plt.legend(fontsize=15)
    plt.xlabel('Time (hh:mm:ss)', fontsize=15)
    plt.ylabel('Pressure (mBar)', fontsize=15)
    plt.tick_params(axis='both', labelsize=15)
    plt.semilogy()
    plt.grid()

    # Format the x-ticks as HH:MM:SS
    plt.xticks(time_seconds[::len(time_seconds)//number_ticks],  # reduce number of ticks
            time_labels[::len(time_labels)//number_ticks],    # format those ticks
            rotation=45)

    plt.tight_layout()

    plt.xlim(f_value,s_value)

    if save:
        plt.savefig(f'{save_name}.jpg', format='jpg', dpi=300, bbox_inches='tight')
        plt.show()
    
    else:
        plt.show()

RGA_spec(Ruta, f_hour, s_hour, number_ticks, 
         save=True, save_name=filename)