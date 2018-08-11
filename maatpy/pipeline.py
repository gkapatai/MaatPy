from imblearn.pipeline import Pipeline
from sklearn.pipeline import _name_estimators


class Pipeline(Pipeline):

    def __init__(self, steps, memory=None):
        """

        :param steps: list
               List of (name, transform) tuples (implementing fit/transform/fit_sample) that are chained,
               in the order in which they are chained, with the last object an estimator.
        :param memory: Instance of joblib.Memory or string, optional (default=None)
               Used to cache the fitted transformers of the pipeline. By default, no caching is performed. If a string
               is given, it is the path to the caching directory. Enabling caching triggers a clone of the transformers
               before fitting. Therefore, the transformer instance given to the pipeline cannot be inspected directly.
               Use the attribute "named_steps" or "steps" to inspect estimators within the pipeline. Caching the
               transformers is advantageous when fitting is time consuming.
        """
        super(Pipeline, self).__init__(steps=steps, memory=memory)


def make_pipeline(*steps, **kwargs):
    """
    Construct a Pipeline from the given estimators.
    This is a shorthand for the Pipeline constructor; it does not require, and does not permit, naming the estimators.
    Instead, their names will be set to the lowercase of their types automatically.

    :param steps: list
           List of (name, transform) tuples (implementing fit/transform/fit_sample) that are chained,
           in the order in which they are chained, with the last object an estimator.
    :param kwargs:
    :return: p: Pipeline
    """
    memory = kwargs.pop('memory', None)
    if kwargs:
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    return Pipeline(_name_estimators(steps))
