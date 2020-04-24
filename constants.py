"""
constants.py shall contain the hardcoded values that our model might need to 
run a simulation.
"""

from enum import Enum
from scipy import stats


FEMALE = 0
MALE = 1

STATES = Enum('States', 'S E I_mild I_wild R')

# Here goes all the transition functions, what is a probability of an agent
# visiting and migrating to another node?
def agent_visit_probability(agent):
    """
    Probability of an agent visiting some node to a higher hierarchy
    """