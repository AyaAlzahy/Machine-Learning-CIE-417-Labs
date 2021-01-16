from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt


def synthesize_data():
    '''
    This function synthesizes balanced data with 2 features that belong to 1 of 2 classes
    '''
    #synthesize balanced 2-class data
    #For more details about make_classification: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
    X, y = make_classification(200, 2, 2, 0, weights=[.5, .5], random_state=15)
    return X,y

def test_and_plot_logistic_regression(ax, clf, X, y):
    #Create a test data grid
    xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    #Predict the probabilities of test data
    probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    plot_decision_boundary(ax, xx,yy, probs, X, y)

def plot_decision_boundary(ax, xx,yy, probs, X, y):
    ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    ax.scatter(X[100:,0], X[100:, 1], c=y[100:], s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)
    ax.set(aspect="equal",
       xlim=(-5, 5), ylim=(-5, 5),
       xlabel="$X_1$", ylabel="$X_2$")
