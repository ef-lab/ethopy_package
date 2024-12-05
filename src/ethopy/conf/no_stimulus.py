# Blank experiment
from ethopy.core.behavior import Behavior
from ethopy.core.stimulus import Stimulus
from ethopy.experiments.passive import Experiment

# define session parameters
session_params = {
    'setup_conf_idx': 0,
}

exp = Experiment()
exp.setup(logger, Behavior, session_params)

conditions = []
conditions += exp.make_conditions(stim_class=Stimulus(), conditions={
    'trial_selection': 'fixed',
    'difficulty': 0
})

# run experiment
exp.push_conditions(conditions)
exp.start()
