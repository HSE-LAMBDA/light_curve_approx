import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils


def hist(bins, name_metric='MAPE', *object_column):
    plt.figure(figsize=(9, 4))
    for i in object_column:
        cmap = 'bgrcrykw'[np.random.randint(0,8)]
        hist_object = np.array(i[0])
        plt.hist(hist_object, density=True, alpha=0.5, color=cmap, bins=bins, label=i[1])
        plt.axvline(np.mean(hist_object), color=cmap, alpha=1)
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.grid(b=1)
    plt.legend()
    plt.title("Histogram %s" % name_metric)
    plt.show()


def hist_log10(bins, name_metric='MAPE', *object_column):
    plt.figure(figsize=(9, 4))
    for i in object_column:
        cmap = 'bgrcrykw'[np.random.randint(0,8)]
        hist_object = np.log10(i[0])
        plt.hist(hist_object, color=cmap, alpha=0.5, bins=bins, label=i[1])
        plt.axvline(np.mean(hist_object), color=cmap, alpha=1)
    plt.xlabel('Data')
    plt.grid(b=1)
    plt.legend()
    plt.title("Histogram log10( %s )" % name_metric)
    
