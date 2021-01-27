import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # markers and colormap
    # np.uniqe returns sorted unique elements as list
    # with 2 labels we should get 2 so the colors will be red and blue
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # gets the min and max value of the first feature
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # gets the min and max value of the second feature
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # np.meshgrid returns coordinate matrices from coordinate vectors.
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # np.ravel returns a 1D array
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    print(xx1.shape)
    z = z.reshape(xx1.shape)

    # draws contour lines, z = height values over which the contour line is drawn
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                    c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')
