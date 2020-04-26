"""
constants.py shall contain the hardcoded values that our model might need to 
run a simulation.
"""

from enum import Enum
from scipy import stats
import pandas as pd
import numpy as np

# gender is a spectrum lol
FEMALE = 0
MALE = 1

STATES = Enum('States', 'S E I_mild I_wild R')
CONTACT_MATRIX = pd.read_csv('data/bd_mu_all_loc.csv').to_numpy()
AGE_GROUPS = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29),
             (30, 34), (35, 39), (40, 44), (45, 49), (50, 54), (55, 59),
             (60, 64), (65, 69), (70, 74), (75, np.inf)]

T_INC = 2.9         # mean incubation period
T_INF_MILD = 14     # mean infectious period (mild)
T_INF_WILD = 5      # mean infectious period (wild)
PROB_MILD = 0.88    # chances of catching a mild infection

GENDER_FACTOR = 7.  # how likely it is for people of same gender
                    # to pass on the disease in the event of a
                    # successful contact (compared to contact between
                    # people of different genders)

# Here goes all the transition functions, what is a probability of an agent
# visiting and migrating to another node?

def age_to_age_group(age: int) -> int:
    """
    convert age to age group id
    """
    for i, age_group in enumerate(AGE_GROUPS):
        if age_group[0] <= age <= age_group[1]:
            return i


def agent_visit_probability(agent):
    """
    Probability of an agent visiting some node to a higher hierarchy
    """
