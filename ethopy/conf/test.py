# Object experiment task 15046
from scipy import interpolate

from Behaviors.MultiPort import *
from Experiments.MatchPort import *
from Stimuli.Panda import *


def interpolate_movement(x):
    if len(x) <= 3:
        return x
    bspline_representation = interpolate.splrep(np.linspace(0, len(x), len(x)), x)
    return interpolate.splev(np.linspace(0, len(x), 100), bspline_representation)

interp = lambda x: interpolate_movement(x)

# define session parameters
session_params = {
    'max_reward'            : 1200,
    'min_reward'            : 700,
    'setup_conf_idx'        : 0,
    'hydrate_delay'         : 45,
}

exp = Experiment()
exp.setup(logger, MultiPort, session_params)
conditions = []

# define environment conditions
env_key = {
    'abort_duration'        : 500,
    'punish_duration'       : 10000,
    'init_ready'            : 300,
    'trial_ready'           : 0,
    'intertrial_duration'   : 500,
    'trial_duration'        : 9000,
    'reward_duration'       : 5000}

print(env_key)

panda_obj = Panda()
panda_obj.fill_colors.set({'background': (0, 0, 0),
                           'start': (0.2, 0.2, 0.2),
                           'reward': (0.6, 0.6, 0.6),
                           'punish': (0, 0, 0)})
    
# #target
# resp_obj = [1, 1]
# rew_prob = [1, 2]
# x_pos = [-0.3, 0.3]
# rot_f = lambda: interp((np.random.rand(20)-.5) *100)
# rots = rot_f()
# block = exp.Block(difficulty=0, next_up=1, next_down=0, staircase_window=20, trial_selection='staircase', stair_up=0.75, stair_down=0.55, incremental_punishment=True)
# for idx, obj_comb in enumerate(resp_obj):
#    conditions += exp.make_conditions(stim_class=panda_obj, conditions={**env_key, **block.dict(),                                               
#            'obj_id'        : resp_obj[idx],
#            'obj_dur'       : 9000,
#            'obj_pos_x'     : x_pos[idx],
#            'obj_pos_y'     : 0.02,
#            'obj_mag'       : 0.5,
#            'obj_rot'       : (rots, rots),
#            'obj_tilt'      : (0, 0),
#            'reward_port'       : rew_prob[idx],
#            'response_port'     : rew_prob[idx],
#             'reward_amount'     : 6})

#distractor+target rots in the center
resp_obj = [(2, 1), (1, 2)]
rew_prob = [1, 2]
x_pos = [(-0.3, 0.3), (-0.3, 0.3)]
rot_f = lambda: interp((np.random.rand(20)-.5) *100)
rots = rot_f()
block = exp.Block(difficulty=1, next_up=1, next_down=1, trial_selection='staircase', stair_up=0.75, stair_down=0.55, incremental_punishment=True)
for idx, obj_comb in enumerate(resp_obj):
    conditions += exp.make_conditions(stim_class=panda_obj, conditions={**env_key,**block.dict(),                                                   
            'obj_id'        : resp_obj[idx],
            'obj_dur'       : 9000,
            'obj_pos_x'     : x_pos[idx],
            'obj_pos_y'     : 0.02,
            'obj_mag'       : 0.5,
            'obj_rot'       : (rots, rots),
            'obj_tilt'      : (0, 0),
            'reward_port'       : rew_prob[idx],
            'response_port'     : rew_prob[idx],
            'reward_amount'     : 6})


# run experiments
exp.push_conditions(conditions)
exp.start()

