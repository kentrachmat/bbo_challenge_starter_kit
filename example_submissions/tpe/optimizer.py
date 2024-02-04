from hyperopt import hp, tpe, Trials
from hyperopt.fmin import generate_trials_to_calculate
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.space import JointSpace

class TPEOptimizer(AbstractOptimizer):
    primary_import = "hyperopt"

    def __init__(self, api_config):
        AbstractOptimizer.__init__(self, api_config)

        # Convert api_config to space for Hyperopt
        self.space = self.convert_api_config_to_hyperopt_space(api_config)
        self.trials = Trials()
        self.tpe_suggest = tpe.suggest

    def convert_api_config_to_hyperopt_space(self, api_config):
        """Converts the api_config to a Hyperopt compatible space.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables.

        Returns
        -------
        space : dict
            Hyperopt compatible space.
        """
        space = {}
        for k, v in api_config.items():
            if v['type'] == 'real':
                space[k] = hp.uniform(k, v['space'][0], v['space'][1])
            elif v['type'] == 'int':
                space[k] = hp.quniform(k, v['space'][0], v['space'][1], 1)
            elif v['type'] == 'cat':
                space[k] = hp.choice(k, v['values'])
        return space

    def suggest(self, n_suggestions=1):
        new_ids = [len(self.trials.trials) + i for i in range(n_suggestions)]
        random_state = None  # Can be replaced with a seed for reproducibility

        suggestions = []
        for _ in range(n_suggestions):
            suggestion = self.tpe_suggest(new_ids, self.trials, self.space, random_state=random_state)
            trials_to_calculate = generate_trials_to_calculate(suggestion)
            self.trials.insert_trial_docs(trials_to_calculate)
            self.trials.refresh()
            suggestions.append({k: v[0] for k, v in suggestion[0]['misc']['vals'].items() if len(v) > 0})

        return suggestions

    def observe(self, X, y):
        for x, loss in zip(X, y):
            self.trials.attachments[x['tid']]['loss'] = loss

if __name__ == "__main__":
    experiment_main(TPEOptimizer)