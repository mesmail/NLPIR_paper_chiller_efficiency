import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'figure.dpi': 110, 'savefig.dpi': 200,
                     'axes.titlesize': 9, 'axes.titleweight': 'bold', 'legend.frameon': False})
C_BASE, C_PRE, C_ACC, C_GREY = '#1f5a8a', '#d9731a', '#2a9d8f', '#7a7a7a'
PEAK = (14.0, 22.0)
