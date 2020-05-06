"""
constants.py shall contain the hardcoded values that our model might need to 
run a simulation.
"""

from enum import Enum
from scipy import stats
import pandas as pd
import numpy as np

# gender is a spectrum lol
FEMALE = True
MALE = False

States = Enum('States', 'S E I_mild I_wild R')
CONTACT_MATRIX = pd.read_csv('data_files/bd_mu_all_loc.csv').set_index('Age group').to_numpy()
AGE_GROUPS = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29),
             (30, 34), (35, 39), (40, 44), (45, 49), (50, 54), (55, 59),
             (60, 64), (65, 69), (70, 74), (75, np.inf)]
AGE_GROUP_SIZE = 5

STATE_PENALTY = {
    States.S: 1.,
    States.E: 1.,
    States.I_mild: 0.5,
    States.I_wild: 0.05,
    States.R: 1.,
}

# Source: https://www.researchgate.net/figure/Average-Daily-Travel-Time-per-Person-by-Age-Group_fig3_245561594
AGE_TRAVEL_RATIO = np.concatenate((
    np.array([0.3] * 5),
    np.repeat(np.array([0.65, 0.82, 0.9, 1., 0.95, 0.85, 0.75, 0.6, 0.3]), 10)
))
GENDER_TRAVEL_RATIO = 0.5

DEPTH_OF_TREE = 4 # How many possible meeting places.

T_INC = 2.9         # mean incubation period
T_INF_MILD = 14     # mean infectious period (mild)
T_INF_WILD = 5      # mean infectious period (wild)
PROB_MILD = 0.88    # chances of catching a mild infection

GENDER_FACTOR = 7.  # how likely it is for people of same gender
                    # to pass on the disease in the event of a
                    # successful contact (compared to contact between
                    # people of different genders)

# TRIP_PROBABILITY_BY_DISTANCE = {1 : 1/7,        # This dictionary contains the probability
#                                 2 : 1/14,       # of an agent making a trip from a given
#                                 3 : 1/30,       # environment E at distance i from the lowest
#                                 4 : 1/90,       # level env to the super env of E
#                                 5 : 1/180}


TRIP_PROBABILITY_BY_DISTANCE = {3: 0.5,
                                2: 1/7,
                                1: 1/14,
                                0: 1/30}

# Here goes all the transition functions, what is a probability of an agent
# visiting and migrating to another node?

def age_to_age_group(age: int) -> int:
    """
    convert age to age group id
    """
    for i, age_group in enumerate(AGE_GROUPS):
        if age_group[0] <= age <= age_group[1]:
            return i


def age_to_age_group_np(age: np.array) -> int:
    """
    convert age to age group id
    """
    return (age//AGE_GROUP_SIZE).clip(0, len(AGE_GROUPS) - 1)


def agent_visit_probability(agent):
    """
    Probability of an agent visiting some node to a higher hierarchy
    """
    pass
