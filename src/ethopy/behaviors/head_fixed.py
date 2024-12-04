import datajoint as dj

from ethopy.core.behavior import Behavior, BehCondition
from ethopy.core.logger import behavior


@behavior.schema
class HeadFixed(Behavior, dj.Manual):
    definition = """
    # This class handles the behavior variables for RP
    ->BehCondition
    """

    def setup(self, exp):
        self.logging = True
        super(HeadFixed, self).setup(exp)

    def exit(self):
        super().exit()
        self.interface.cleanup()
