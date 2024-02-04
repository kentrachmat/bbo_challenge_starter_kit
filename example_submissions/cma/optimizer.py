import cma
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main

class CMAESOptimizer(AbstractOptimizer):
    primary_import = "cma"

    def __init__(self, api_config):
        AbstractOptimizer.__init__(self, api_config)
        
        # Extract bounds for each parameter and set up CMA-ES options
        self.bounds = [(-5, 5)] * len(api_config)  # Example bounds, adjust as needed
        self.options = {'bounds': self.bounds}
        self.es = None

    def suggest(self, n_suggestions=1):
        if self.es is None:
            # Initialize CMA-ES
            self.es = cma.CMAEvolutionStrategy(len(self.bounds) * [0], 0.5, self.options)
        
        # Ask for suggestions
        next_guesses = self.es.ask(n_suggestions)
        return next_guesses

    def observe(self, X, y):
        if self.es is not None:
            # Update CMA-ES with observed results
            self.es.tell(X, y)

if __name__ == "__main__":
    experiment_main(CMAESOptimizer)
