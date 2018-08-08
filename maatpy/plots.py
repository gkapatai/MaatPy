import itertools
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.utils import check_random_state
from sklearn.metrics import cohen_kappa_score as kappa_scorer

__all__ = ['plot_confusion_matrix', 'plot_decision_function', 'plot_learning_curve', 'plot_resampling']


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=[.5, .75, 1.0], random_state=None):
    """
    Generate a simple plot of the test and training learning curve.

    Source:
    http://scikit-learn.org/stable/modules/generated/sklearn.learning_curve.learning_curve.html#sklearn.learning_curve

    :param estimator: object type that implements the "fit" and "predict" methods
           An object of that type which is cloned for each validation.
    :param title: string
           Title for the chart.
    :param X: X: {array-like, sparse matrix}, shape (n_samples, n_features)
           Matrix containing the data which have to be sampled.
    :param y: array-like, shape (n_samples,)
           Corresponding label for each sample in X.
    :param ylim: tuple, shape (ymin, ymax), optional
           Defines minimum and maximum yvalues plotted.
    :param cv: int, cross-validation generator or an iterable, optional
           Determines the cross-validation splitting strategy.
           Possible inputs for cv are:
            - None, to use the default 3-fold cross-validation,
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.
    :param n_jobs: integer, optional
           Number of jobs to run in parallel (default 1).
    :param train_sizes: array-like, shape (n_ticks,), dtype float or int
           Relative or absolute numbers of training examples that will be used to generate the learning curve.
           If the dtype is float, it is regarded as a fraction of the maximum size of the training set (that is
           determined by the selected validation method), i.e. it has to be within (0, 1]. Otherwise it is
           interpreted as absolute sizes of the training sets.
    :param random_state: int, RandomState instance or None, optional (default=None)
           If int, random_state is the seed used by the random number generator; If RandomState instance,
           random_state is the random number generator; If None, the random number generator is the RandomState
           instance used by 'np.random'.
    :return:
    """

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    random_state = check_random_state(random_state)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=kappa_scorer, random_state=random_state)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_decision_function(X, y, clf, ax):
    """
    Plot the decision function of a classifier given some data

    Source:
    http://contrib.scikit-learn.org/imbalanced-learn/stable/auto_examples/under-sampling/\
    plot_comparison_under_sampling.html

    :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
           Matrix containing the data which have to be sampled.
    :param y: array-like, shape (n_samples,)
           Corresponding label for each sample in X.
    :param clf: classifier or pipeline object that implements the predict function
    :param ax: matplotlib.pyplot.axis object
    :return:
    """
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')


def plot_resampling(X, y, sampling, ax):
    """
    Plot the sample space after resampling to illustrate the characteristic of an algorithm.

    Source:
    http://contrib.scikit-learn.org/imbalanced-learn/stable/auto_examples/under-sampling/\
    plot_comparison_under_sampling.html

    :param X:{array-like, sparse matrix}, shape (n_samples, n_features)
           Matrix containing the data which have to be sampled.
    :param y: array-like, shape (n_samples,)
           Corresponding label for each sample in X.
    :param sampling: sampler object or pipeline object that implements the fit_sample function
    :param ax: matplotlib.pyplot.axis object
    :return:
    """
    X_res, y_res = sampling.fit_sample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8)
    # make nice plotting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    return Counter(y_res)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.

    Source:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    :param cm: sklearn.metrics.confusion_matrix object
           A confusion matrix object created using the sklearn.metrics.confusion_matrix on your data
           { confusion_matrix(y_test, y_pred) }
    :param classes: array, shape = [n_classes], optional
           List of labels to index the matrix. This may be used to reorder or select a subset of labels.
           If none is given, those that appear at least once in y_true or y_pred are used in sorted order.
    :param normalize: bool, optional (default=False)
           Whether or not to normalise the confusion matrix
    :param title: str, optional (default='Confusion matrix')
           Title for the plot.
    :param cmap: matplotlib colormap, optional (default=plt.cm.Blues)
    :return:
    """
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')