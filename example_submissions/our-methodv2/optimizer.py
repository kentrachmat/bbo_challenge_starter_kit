import bayesmark.random_search as rs
from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

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
        self.X = np.empty((0, len(api_config)))  # Storing input points
        self.y = np.empty((0, 1)) 

        self.preprocessor_dict = self.build_preprocessor(api_config)

    def process(self, x):
        """Function to process one point in the hyperparameter space.

        Args:
            x (dict): Dictionary with values of hyperparameters.

        Returns:
            List: List of processed values.
        """        
        features = []
        for feature in self.api_config:
            feature_values = x[feature]
            features.append(self.preprocessor_dict[feature](feature_values))
        return features

    def build_preprocessor(self, api_config):
        """This function builds a dictionary of functions to process the hyperparameters.

        Args:
            api_config (dict): Dictionary with the configuration of the hyperparameters.

        Returns:
            dict: Dictionary with the functions to process the hyperparameters.
        """        
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
                        processing_dict[feature] =  self.log_min_max_processing(min_val, max_val)
                    else:
                        processing_dict[feature] = self.min_max_processing(min_val, max_val)

                else:
                    processing_dict[feature] = lambda x: x
                
        return processing_dict

    def categorical_processing(self, cats):
        """This function returns a function to process categorical variables.

        Args:
            cats (list): List of possible values for the categorical variable.
        """        
        def process(x):
            return cats.index(x)
        return process
    
    def min_max_processing(self, min_val, max_val):
        """This function returns a function to process real variables."""
        def process(x):
            return (x - min_val) / (max_val - min_val)
        return process
    
    def log_min_max_processing(self, min_val, max_val):
        """This function returns a function to process real variables in log scale."""
        def process(x):
            return (np.log10(x) - np.log10(min_val)) / (np.log10(max_val) - np.log10(min_val))
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
        if len(self.X) > 0 and len(self.y) > 0:
            # Sample new points by maximizing the expected improvement
            next_guess = self.max_sampling(n_suggestions)
        else:
            # Use a random initialization strategy for the first suggestion
            next_guess = rs.suggest_dict([], [], self.api_config, n_suggestions=n_suggestions,)
        return next_guess

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
        # Process the input points and store them
        X_p = np.array([self.process(x) for x in X])
        # Store the input points and the corresponding values
        self.X = np.vstack((self.X, X_p))
        self.y = np.append(self.y, y)

        if self.X.shape[0] > 0 and self.y.shape[0] > 0:
            # Fit a Gaussian Process model to the data
            self.meta_model.fit(self.X, self.y)
            
    def max_sampling(self, n_suggestions):
        N = 10 
        M = 1000 # Number of random points to sample

        # Sample M random points
        candidates = rs.suggest_dict([], [], self.api_config, n_suggestions=max(M, N*n_suggestions))

        # Process the input points
        X = np.array([self.process(c) for c in candidates])

        # Compute the expected improvement for each point
        ei_list = self.expected_improvement(X)

        # Select the n_suggestions points with the highest expected improvement
        best = np.argsort(ei_list)[-n_suggestions:]
        return [candidates[i] for i in best]

    def expected_improvement(self, X):
        """Expected improvement according to the formula in the paper.

        Args:
            X (array): Array with the values

        Returns:
            array: Array with the expected improvement for each point.
        """        
        mu, sigma = self.meta_model.predict(X, return_std=True)
        mu_sample = self.meta_model.predict(self.X)

        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            gamma = (mu_sample_opt - mu)/sigma
            ei = sigma*(gamma*norm.cdf(gamma) + norm.pdf(gamma))
            ei[sigma == 0.0] = 0.0

        return ei
    

if __name__ == "__main__":
    experiment_main(OurOptimizer)
