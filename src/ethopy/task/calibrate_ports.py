from ethopy.experiments.calibrate import Experiment

# define calibration parameters
session_params = {
    'duration'        : [20, 30, 40, 150],
    'ports'           : [1, 2],
    'pulsenum'        : [60, 30, 20, 10],
    'pulse_interval'  : [40, 40, 40, 40],
    'save'            : True,
    'setup_conf_idx'  : 0,
}

# run experiment
exp = Experiment()
exp.setup(logger, session_params)
exp.run()

