"""
constants.py shall contain the hardcoded values that our model might need to 
run a simulation.
"""

from enum import Enum

from scipy import stats


FEMALE = 0
MALE = 1

STATES = Enum('States', 'S E I_mild I_wild R'])