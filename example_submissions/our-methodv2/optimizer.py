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
        self.lb, self.ub = self.extract_bounds(api_config)

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
        if len(self.X) > 0 and len(self.y) > 0:
            # Sample new points using MCMC after at least one observation
            sampled_points = self.mcmc_sample(n_suggestions)
            next_guess = [self.convert_to_dict(point) for point in sampled_points]
        else:
            # Use a random initialization strategy for the first suggestion
            next_guess = self.random_initialization(n_suggestions)
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
        X_p = np.array([self.process(x) for x in X])
        self.X = np.vstack((self.X, X_p))
        self.y = np.append(self.y, y)

        if self.X.shape[0] > 0 and self.y.shape[0] > 0:
            self.meta_model.fit(self.X, self.y)
            
    def mcmc_sample(self, n_suggestions):
        # Start from a random point
        current_point = np.random.uniform(self.lb, self.ub)
        current_ei = self.expected_improvement(current_point.reshape(1, -1))

        samples = [current_point]
        for _ in range(n_suggestions - 1):
            # Propose a new point
            new_point = np.random.uniform(self.lb, self.ub)
            new_ei = self.expected_improvement(new_point.reshape(1, -1))

            # Accept new point with probability min(1, new_ei/current_ei)
            if new_ei > current_ei or np.random.uniform(0, 1) < new_ei / current_ei:
                current_point, current_ei = new_point, new_ei

            samples.append(current_point)

        return np.array(samples)

    def expected_improvement(self, X, xi=0.01):
        mu, sigma = self.meta_model.predict(X, return_std=True)
        mu_sample = self.meta_model.predict(self.X)

        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei
    
    def random_initialization(self, n_suggestions):
        # Randomly sample points within the bounds
        random_points = np.random.uniform(self.lb, self.ub, (n_suggestions, len(self.api_config)))
        return [self.convert_to_dict(point) for point in random_points]

    def convert_to_dict(self, point):
        # Convert a point from array format to dictionary format
        return {feature: point[i] for i, feature in enumerate(self.api_config)}
    
    def extract_bounds(self, api_config):
        lb = []
        ub = []
        for feature, config in api_config.items():
            if config['type'] in ['int', 'real']:
                f_lb, f_ub = config['range']
                lb.append(f_lb)
                ub.append(f_ub)
            elif config['type'] == 'bool':
                lb.append(0)
                ub.append(1)
            elif config['type'] == 'cat':
                lb.append(0)
                ub.append(len(config['values']) - 1)
            else:
                raise ValueError(f"Unsupported type: {config['type']}")
        return np.array(lb), np.array(ub)

if __name__ == "__main__":
    experiment_main(OurOptimizer)
