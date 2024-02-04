import bayesmark.random_search as rs
from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class OurOptimizer(AbstractOptimizer):
    # Unclear what is best package to list for primary_import here.
    primary_import = "bayesmark"

    def __init__(self, api_config, random=np_util.random):
        """Build wrapper class to use random search function in benchmark.

        Settings for `suggest_dict` can be passed using kwargs.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        self.random = random
        self.meta_model = GaussianProcessRegressor(kernel=Matern(nu=2.5))
        self.api_config = api_config
        self.preprocessor_dict = self.build_preprocessor(api_config)

    def process(self, x):
        features = []
        for feature in self.api_config:
            feature_values = x[feature]
            features.append(self.preprocessor_dict[feature](feature_values))
        return features

    def build_preprocessor(self, api_config):
        processing_dict = {}
        for feature in api_config:
            if api_config[feature]['type'] == 'bool':
                processing_dict[feature] = lambda x: int(x)

            elif api_config[feature]['type'] == 'cat':
                processing_dict[feature] = self.categorical_processing(api_config[feature]['values'])

            elif api_config[feature]['type'] == 'int':
                if 'range' in api_config[feature]:
                    min_val, max_val = api_config[feature]['range']
                    processing_dict[feature] =  self.min_max_processing(min_val, max_val)

            elif api_config[feature]['type'] == 'real':
                if 'range' in api_config[feature]:
                    min_val, max_val = api_config[feature]['range']
                    if api_config[feature]['space'] == 'log':
                        min_val = np.log10(min_val)
                        max_val = np.log10(max_val)
                        processing_dict[feature] =  lambda x: self.min_max_processing(min_val, max_val)(np.log10(x))
                    else:
                        processing_dict[feature] =  self.min_max_processing(min_val, max_val)

                else:
                    processing_dict[feature] = lambda x: x
                
        return processing_dict

    def categorical_processing(self, cats):
        def process(x):
            return cats.index(x)
        return process
    
    def min_max_processing(self, min_val, max_val):
        def process(x):
            return (x - min_val) / (max_val - min_val)
        return process

    def suggest(self, n_suggestions=1):
        """Get suggestion.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        N = 30
        x_guess = rs.suggest_dict([], [], self.api_config, n_suggestions=N*n_suggestions,)# random=self.random)
        x_guess_p = [self.process(x) for x in x_guess]
        y_guess = self.meta_model.predict(x_guess_p)
        best_idx = np.argsort(y_guess)[:n_suggestions]
        x_guess = [x_guess[i] for i in best_idx]
        return x_guess


    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        X_p = [self.process(x) for x in X]
        self.meta_model.fit(X_p, y)


if __name__ == "__main__":
    experiment_main(OurOptimizer)
