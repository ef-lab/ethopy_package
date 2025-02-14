# Orientation discrimination experiment
from ethopy.behaviors.multi_port import MultiPort
from ethopy.experiments.match_port import Experiment
from ethopy.stimuli.grating import Grating

# define session parameters
session_params = {
    "trial_selection": "staircase",
    "max_reward": 3000,
    "min_reward": 30,
    "bias_window": 5,
    "staircase_window": 10,
    "stair_up": 0.7,
    "stair_down": 0.6,
    "setup_conf_idx": 0,
}

exp = Experiment()
exp.setup(logger, MultiPort, session_params)

# define stimulus conditions
key = {
    "contrast": 100,
    "spatial_freq": 0.05,  # cycles/deg
    "square": 0,  # squarewave or Guassian
    "temporal_freq": 0,  # cycles/sec
    "flatness_correction": 1,  # adjustment of spatiotemporal frequencies based on animal distance
    "duration": 5000,
    "difficulty": 1,
    "trial_duration": 5000,
    "intertrial_duration": 0,
    "reward_amount": 8,
}

repeat_n = 1
conditions = []

ports = {1: 0,
         2: 90}

block = exp.Block(difficulty=1, next_up=1, next_down=1, trial_selection='staircase', metric='dprime', stair_up=1, stair_down=0.5)

for port in ports:
    conditions += exp.make_conditions(stim_class=Grating(), conditions={**block.dict(),
                                                                              **key,
                                                                              'theta'        : ports[port],
                                                                              'reward_port'  : port,
                                                                              'response_port': port})

# run experiments
exp.push_conditions(conditions)
exp.start()