from ethopy.behaviors.multi_port import MultiPort
from ethopy.core.stimulus import Stimulus
from ethopy.experiments.free_water import Experiment

# define session parameters
session_params = {
    'start_time'            : '00:00:00',
    'stop_time'             : '23:50:00',
    'max_reward'            : 5000,
    'bias_window'           : 5,
    'noresponse_intertrial' : False,
    'setup_conf_idx'        : 0,
}

exp = Experiment()
exp.setup(logger, MultiPort, session_params)
block = exp.Block(trial_selection='biased')

conditions = exp.make_conditions(stim_class=Stimulus(), 
                                 conditions={**block.__dict__,                            
                                             'timeout_duration'   : 0,
                                             'intertrial_duration': 500,
                                             'init_duration'      : 0,
                                             'delay_duration'     : 0,
                                             'reward_amount'      : 10,
                                             'reward_port': -1,
                                             'response_port': -1,
                                             })

# run experiments
exp.push_conditions(conditions)
exp.start()


