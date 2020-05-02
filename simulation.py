import numpy as np
import pandas as pd
from model import Person
from typing import List, Dict
from constants import (AGE_GROUPS, age_to_age_group, age_to_age_group_np, 
                       TRIP_PROBABILITY_BY_DISTANCE, DEPTH_OF_TREE, States)


def simulate(agents, node_level, contact_matrix: np.array):
    """
    simulate non-spatial contact among a group of agents.

    agent: list of persons in simulation
    contact_matrix: rate of contact between different age groups

    returns: a dictionary of each person and list of other persons he/she
             came in contact with.
    """
    touch = 0
    print('Agents: ', len(agents))
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
                touch += 1
            if np.random.uniform(0, 1) <= prob2: # make contact with probability prob
                contacts[agent2].append(agent1)
                touch += 1
    print('Contacts: ',touch)
    return contacts


def simulate_np(agents, node_level, contact_matrix: np.array):
    """
    simulate non-spatial contact among a group of agents.

    agent: list of persons in simulation
    contact_matrix: rate of contact between different age groups

    returns: a dictionary of each person and list of other persons he/she
             came in contact with.
    """
    if len(agents) == 0:
        return {}

    ages_and_states = np.array([(agent.age, agent.state) for agent in agents])
    ages, states = ages_and_states[:, 0], ages_and_states[:, 1]
    age_groups = age_to_age_group_np(ages).astype(np.uint8)

    trip_probability = TRIP_PROBABILITY_BY_DISTANCE[node_level]
    total_of_age_group = np.array([(age_groups == i).sum() for i in range(len(AGE_GROUPS))])
    # We only care about agents who has state S.
    # This is the probability that j touches i.
    prob_i_j = contact_matrix / (1e-9 + DEPTH_OF_TREE * trip_probability * np.expand_dims(total_of_age_group, axis=1))
    prob_i_j = np.nan_to_num(prob_i_j).clip(0, 1) # TODO (mahi): speedhack, fix
    # Now, expand out the contact probability matrix
    uninfected = (states == States.S)
    can_infect = (states != States.S) & (states != States.R)

    expanded_prob_matrix = prob_i_j[:, age_groups[uninfected]]
    can_infect_count = np.array([(age_groups[can_infect] == i).sum() for i in range(len(AGE_GROUPS))])

    # Now, random contact sampling
    contact_summary = np.random.binomial(can_infect_count[:, np.newaxis], expanded_prob_matrix)

    counter = 0 # Since we only care about contacts of non-sick people
    result = {}
    for agent in agents:
        if agent.state == States.S:
            result[agent] = contact_summary[:, counter].astype(np.uint16)
            counter += 1

    return result


