import glob
import matplotlib.pyplot as plt
import numpy as np

def setup_and_save_plot(fname):
    # border
    if False:
        plt.plot([xmin, xmax, xmax, xmin, xmin],
                 [ymax, ymax, ymin, ymin, ymax],
                 'w-',
                 lw=0.5)

    plt.gca().set_aspect('equal')    
    plt.xlim([xmin - gap, xmax + gap])
    plt.ylim([ymin - gap, ymax + gap])
    plt.axis('off')

    plt.savefig(fname,
                dpi=200,
                bbox_inches='tight',
                pad_inches=0,
                transparent=False)
    return None

if __name__ == '__main__':
    plt.style.use('dark_background')

    flist = sorted(glob.glob('out_*.csv'))

    xmin = -1
    xmax = 1
    ymin = -2
    ymax = 2
    gap = 0.01

    if True:
        for fname in flist:
            print(fname)

            data = np.genfromtxt(fname, delimiter=',', skip_header=True)

            plt.figure()
            plt.plot(data[:, 0], data[:, 1], 'w.', mfc='w', ms=1)
            # plt.title(fname)
            setup_and_save_plot(fname + '.jpg')

    # --- metrics

    if True:
        fname = 'metrics_first_neighbor_distance'
        data = np.genfromtxt('metrics_first_neighbor_distance.csv', delimiter=',', skip_header=True)
        dist_sq = np.genfromtxt('metrics_first_neighbor_distance_dist_sq.csv', delimiter=',', skip_header=True)
    
    
        plt.figure()
        plt.plot(data[:, 0], data[:, 1], 'w.', mfc='w', ms=1)

        theta = np.linspace(0, 2 * np.pi, 16)

        for k in range(len(data[:, 0])):
            x = data[k, 0]
            y = data[k, 1]
            r = 0.5 * np.sqrt(dist_sq[k])
            plt.plot(x + r * np.cos(theta), y + r * np.sin(theta), 'w-', lw=0.5)
    
        setup_and_save_plot(fname + '.jpg')


        #
        fname = 'metrics_distance_to_boundary'
        data = np.genfromtxt('metrics_distance_to_boundary.csv', delimiter=',', skip_header=True)
        dist = np.genfromtxt('metrics_distance_to_boundary_dist.csv', delimiter=',', skip_header=True)
        dist = dist / np.max(dist)
        
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], s=dist * 30, c=dist, cmap='gray')
    
        setup_and_save_plot(fname + '.jpg')
        
        #
        fname = 'metrics_nearest_neighbors_indices'
        data = np.genfromtxt('metrics_nearest_neighbors_indices.csv', delimiter=',', skip_header=True)
        idx = np.genfromtxt('metrics_nearest_neighbors_indices_idx.csv', delimiter=',', skip_header=True)
    
        plt.figure()
        plt.plot(data[:, 0], data[:, 1], 'w.', mfc='w', ms=1)

        for k in range(len(data[:, 0])):
            x = data[k, 0]
            y = data[k, 1]
            alpha = np.random.rand()
            
            for p in idx[k, :]:
                p = int(p)
                xn = data[p, 0]
                yn = data[p, 1]
                plt.plot([x, xn], [y, yn], 'w-', lw=0.5, alpha=alpha)
    
        setup_and_save_plot(fname + '.jpg')
        
        #
        fname = 'metrics_dbscan_clustering'
        data = np.genfromtxt('metrics_dbscan_clustering.csv', delimiter=',', skip_header=True)
        labels = np.genfromtxt('metrics_dbscan_clustering_labels.csv', delimiter=',', skip_header=True)
    
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], s=2, c=labels, cmap='bwr')
    
        setup_and_save_plot(fname + '.jpg')
        
        #
        fname = 'metrics_percolation_clustering'
        data = np.genfromtxt('metrics_percolation_clustering.csv', delimiter=',', skip_header=True)
        labels = np.genfromtxt('metrics_percolation_clustering_labels.csv', delimiter=',', skip_header=True)
    
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], s=2, c=labels, cmap='bwr')
    
        setup_and_save_plot(fname + '.jpg')
        
        #
        fname = 'metrics_kmeans_clustering'
        data = np.genfromtxt('metrics_kmeans_clustering.csv', delimiter=',', skip_header=True)
        centroids = np.genfromtxt('metrics_kmeans_clustering_centroids.csv', delimiter=',', skip_header=True)
        labels = np.genfromtxt('metrics_kmeans_clustering_labels.csv', delimiter=',', skip_header=True)
    
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], s=2, c=labels, cmap='bwr')
    
        setup_and_save_plot(fname + '.jpg')

        #
        fname = 'metrics_radial_distribution'
        data = np.genfromtxt('metrics_radial_distribution.csv', delimiter=',', skip_header=True)
        r = np.genfromtxt('metrics_radial_distribution_r.csv', delimiter=',', skip_header=True)
        g = np.genfromtxt('metrics_radial_distribution_pdf.csv', delimiter=',', skip_header=True)

        plt.figure();
        plt.plot(r, g, 'w-')
        
        #
        fname = 'random_walk_filaments_dst'
        data = np.genfromtxt('out_random_walk_filaments.csv', delimiter=',', skip_header=True)
        d = np.genfromtxt('metrics_random_walk_filaments_dst.csv', delimiter=',', skip_header=True)

        plt.figure();
        plt.scatter(data[:, 0], data[:, 1], s=0.5, c=d, cmap='gray_r')
    
        setup_and_save_plot(fname + '.jpg')

        #
        fname = 'metrics_distance_to_boundary'
        data = np.genfromtxt('metrics_distance_to_boundary.csv', delimiter=',', skip_header=True)
        dist = np.genfromtxt('metrics_distance_to_boundary_dist.csv', delimiter=',', skip_header=True)
        dist = dist / np.max(dist)
        
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], s=dist * 30, c=dist, cmap='gray')
    
        setup_and_save_plot(fname + '.jpg')
        
        #
        fname = 'metrics_local_density_knn'
        data = np.genfromtxt('metrics_local_density_knn.csv', delimiter=',', skip_header=True)
        d = np.genfromtxt('metrics_local_density_knn_d.csv', delimiter=',', skip_header=True)
        
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], s=1, c=d, cmap='nipy_spectral')
    
        setup_and_save_plot(fname + '.jpg')
        
        
    plt.show()
