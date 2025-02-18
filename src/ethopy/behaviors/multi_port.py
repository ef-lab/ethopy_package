import datajoint as dj
import numpy as np

from ethopy.core.behavior import Behavior
from ethopy.core.logger import behavior


@behavior.schema
class MultiPort(Behavior, dj.Manual):
    definition = """
    # This class handles the behavior variables for RP
    ->behavior.BehCondition
    """

    class Response(dj.Part):
        definition = """
        # Lick response condition
        -> MultiPort
        response_port              : tinyint          # response port id
        """

    class Reward(dj.Part):
        definition = """
        # reward port conditions
        -> MultiPort
        ---
        reward_port               : tinyint          # reward port id
        reward_amount=0           : float            # reward amount
        reward_type               : varchar(16)      # reward type
        """

    def __init__(self):
        super().__init__()
        self.cond_tables = ["MultiPort", "MultiPort.Response", "MultiPort.Reward"]
        self.required_fields = ["response_port", "reward_port", "reward_amount"]
        self.default_key = {"reward_type": "water"}

    def is_ready(self, duration, since=False):
        position, ready_time, tmst = self.interface.in_position()
        if duration == 0:
            return True
        elif position == 0 or position.ready == 0:
            return False
        elif not since:
            return ready_time > duration  # in position for specified duration
        elif tmst >= since:
            # has been in position for specified duration since timepoint
            return ready_time > duration
        else:
            # has been in position for specified duration since timepoint
            return (ready_time + tmst - since) > duration

    def is_correct(self):
        """Check if the response port is correct.

        if current response port is -1, then any response port is correct
        otherwise if the response port is equal to the current response port/ports,
        then it is correct

        Returns:
            bool: True if correct, False otherwise

        """
        return self.curr_cond['response_port'] == -1 or \
            np.any(np.equal(self.response.port, self.curr_cond['response_port']))

    def is_off_proximity(self):
        return self.interface.off_proximity()

    def reward(self, tmst=0):
        """Give reward at latest licked port.

        After the animal has made a correct response, give the reward at the
        first port that animal has licked and is definded as reward.

        Args:
            tmst (int, optional): Time in milliseconds. Defaults to 0.

        Returns:
            bool: True if rewarded, False otherwise

        """
        # if response and reward ports are the same no need of tmst
        if self.response.reward:
            tmst = 0
        # check that the last licked port is also a reward port
        licked_port = self.is_licking(since=tmst, reward=True)
        if licked_port == self.curr_cond["reward_port"]:
            self.interface.give_liquid(licked_port)
            self.log_reward(self.reward_amount[self.licked_port])
            self.update_history(self.response.port, self.reward_amount[self.licked_port])
            return True
        return False

    def exit(self):
        super().exit()
        self.interface.cleanup()

    def punish(self):
        port = self.response.port if self.response.port > 0 else np.nan
        self.update_history(port, punish=True)
