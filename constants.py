"""
constants.py shall contain the hardcoded values that our model might need to 
run a simulation.
"""

from scipy import stats


FEMALE = 0
MALE = 1

STATES = namedtuple('States', ['S', 'E', 'I_mild', 'I_wild', 'R'])
COVID_states = STATES(range(5))