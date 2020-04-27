import numpy as np
import pandas as pd
from model import Person
from typing import List, Dict
from constants import AGE_GROUPS, age_to_age_group


def simulate(agents: List(Person), contact_matrix: np.array) -> Dict[Person, List(Person)]:
    """
    simulate non-spatial contact among a group of agents.

    agent: list of persons in simulation
    contact_matrix: rate of contact between different age groups

    returns: a dictionary of each person and list of other persons he/she
             came in contact with.
    """
    contacts = dict()
    for (i, agent1) in enumerate(agents):
        for (j, agent2) in enumerate(agents[i+1:]):
            id1 = age_to_age_group(agent1.age)
            id2 = age_to_age_group(agent2.age)
            
            if contacts.get(agent1) is None:
                contacts[agent1] = []            
            if contacts.get(agent2) is None:
                contacts[agent2] = []

            prob1 = contact_matrix[id1][id2]/np.sum(contact_matrix[id1])
            prob2 = contact_matrix[id2][id1]/np.sum(contact_matrix[id2])
            
            if np.random.uniform(0, 1) <= prob1: # make contact with probability prob
                contacts[agent1].append(agent2)
            if np.random.uniform(0, 1) <= prob2: # make contact with probability prob
                contacts[agent2].append(agent1)
    return contacts

