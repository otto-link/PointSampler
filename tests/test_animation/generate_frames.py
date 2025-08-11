import glob
import matplotlib.pyplot as plt
import numpy as np

def setup_and_save_plot(fname):
    plt.gca().set_aspect('equal')    
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.axis('off')

    plt.savefig(fname,
                dpi=80,
                bbox_inches='tight',
                pad_inches=0,
                transparent=False)
    return None

if __name__ == '__main__':
    plt.style.use('dark_background')

    # poisson
    flist = sorted(glob.glob('anim_poisson_*.csv'))
    xmin = 0
    xmax = 2
    ymin = 0
    ymax = 1

    for fname in flist:
        print(fname)

        data = np.genfromtxt(fname, delimiter=',', skip_header=True)

        plt.figure()
        plt.plot(data[:, 0], data[:, 1], 'w.', mfc='w', ms=1)
        setup_and_save_plot(fname + '.jpg')
        plt.close()
        
    # relaxation
    flist = sorted(glob.glob('anim_relaxation_ktree_*.csv'))
    xmin = 0
    xmax = 2
    ymin = 0
    ymax = 1

    for fname in flist:
        print(fname)

        data = np.genfromtxt(fname, delimiter=',', skip_header=True)

        plt.figure()
        plt.plot(data[:, 0], data[:, 1], 'w.', mfc='w', ms=1)
        setup_and_save_plot(fname + '.jpg')
        plt.close()

    # distance
    flist = sorted(glob.glob('anim_distance_filter_*.csv'))
    xmin = 0
    xmax = 2
    ymin = 0
    ymax = 1

    for fname in flist:
        print(fname)

        data = np.genfromtxt(fname, delimiter=',', skip_header=True)

        plt.figure()
        plt.plot(data[:, 0], data[:, 1], 'w.', mfc='w', ms=1)
        setup_and_save_plot(fname + '.jpg')
        plt.close()

        
    # plt.show()
