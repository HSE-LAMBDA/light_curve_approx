import numpy as np
import pandas as pd

import matplotlib
matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt
from matplotlib import rc

rc('font',**{'family':'serif'})
rc('text', usetex=True)
rc('text.latex', preamble='''
        \\usepackage[utf8]{inputenc}
        \\usepackage{amssymb}
        \\usepackage{amsfonts}
        \\usepackage[russian]{babel}''')

N_OBS = 500
N_PASSBANDS = 6

passband2name = {0: 'u', 1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'y'}
passband2lam  = {0: np.log10(3751.36), 1: np.log10(4741.64), 2: np.log10(6173.23), 
                 3: np.log10(7501.62), 4: np.log10(8679.19), 5: np.log10(9711.53)}

def get_object(data, object_id):
    anobject = data[data.object_id == object_id]
    return anobject

def get_passband(anobject, passband):
    light_curve = anobject[anobject.passband == passband]
    return light_curve

def create_approx_object(anobject, n=N_OBS):
    mjd = anobject['mjd'].values
    df_s = []
    for passband in range(N_PASSBANDS):
        df = pd.DataFrame()
        df['mjd'] = np.linspace(mjd.min(), mjd.max(), n)
        df['object_id'] = anobject.object_id
        df['passband'] = passband
        df['log_lam'] = passband2lam[passband]
        df['flux'] = 0
        df['flux_err'] = 0
        df['detected_bool'] = 1
        df_s.append(df)
    new_object = pd.concat(df_s, axis=0)
    return new_object

def compile_obj(t, flux, flux_err, passband):
    obj = pd.DataFrame()
    obj['mjd']      = t
    obj['flux']     = flux
    obj['flux_err'] = flux_err
    obj['passband'] = passband
    return obj

def plot_light_curves(anobject, suf="", title=""):
    anobject = anobject.sort_values('mjd')
    fig = plt.figure(figsize=(9, 5), tight_layout = {'pad': 0})
    for passband in range(6):
        light_curve = get_passband(anobject, passband)
        plt.plot(light_curve['mjd'].values, light_curve['flux'].values, '-o', 
                    label=passband2name[passband], linewidth=2)
#    plt.xlabel("Modified Julian date", fontsize=24)
    plt.xlabel('Модифицированная Юлианская дата', fontsize=24)
    plt.xticks(fontsize=22)
#    plt.ylabel("Flux", fontsize=24)
    plt.ylabel('Поток излучения', fontsize=24)
    plt.yticks(fontsize=22)
    plt.legend(loc='best', ncol=2, fontsize=24, columnspacing=1.0)
    plt.title(title, fontsize=28)
    plt.grid(True)
    plt.show()
#    fig.savefig("../pictures/light_curve_{}_{}.pdf".format(anobject.object_id.to_numpy()[0], suf), 
#                bbox_inches='tight', pad_inches=0.0)