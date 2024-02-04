from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

class NewOptimizerName(AbstractOptimizer):
    primary_import = "sklearn"

    def __init__(self, api_config):
        AbstractOptimizer.__init__(self, api_config)
        self.gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-6)
        self.bounds = np.array([v["range"] for v in api_config.values()], dtype=float)
        self.X = []
        self.y = []

    def suggest(self, n_suggestions=1):
        if not self.X:
            # Randomly select initial point if no data is available
            return [self.random_suggestion() for _ in range(n_suggestions)]

        self.gp.fit(self.X, self.y)
        suggestions = []
        for _ in range(n_suggestions):
            suggestion = self.optimize_acquisition()
            suggestions.append(suggestion)
        return suggestions

    def observe(self, X, y):
        self.X.extend(X)
        self.y.extend(y)

    def random_suggestion(self):
        return {k: np.random.uniform(v[0], v[1]) for k, v in zip(self.api_config.keys(), self.bounds)}

    def acquisition_function(self, x):
        x = np.atleast_2d(x)
        mu, sigma = self.gp.predict(x, return_std=True)
        best_y = max(self.y)
        with np.errstate(divide='warn'):
            improvement = mu - best_y - 0.01
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return -ei  # Negative because we minimize

    def optimize_acquisition(self):
        res = minimize(lambda x: self.acquisition_function(x), 
                       x0=np.random.uniform(self.bounds[:, 0], self.bounds[:, 1]), 
                       bounds=self.bounds)
        return {k: v for k, v in zip(self.api_config.keys(), res.x)}

if __name__ == "__main__":
    experiment_main(NewOptimizerName)
