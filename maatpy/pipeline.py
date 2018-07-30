from imblearn.pipeline import *


class Pipeline(Pipeline):
    """
    Pipeline of transforms and resamples with a final estimator.

    This is just a dummy class that creates an object of imblearn.pipeline.Pipeline class
    """

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

