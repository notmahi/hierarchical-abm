"""
rules.py defines all the rules for the agents movement. In short, to change the
behavior of the agent, we should only change the functions here.

Every rule should be made in a functional way so that they can switched easily
later, and so that we can add more parameters to agent behavior easily.
"""

import numpy as np
from typing import List
from constants import (STATES, 
                       T_INC, T_INF_MILD, T_INF_WILD, T_INF_WILD, 
                       PROB_MILD, 
                       age_to_age_group, 
                       CONTACT_MATRIX, 
                       GENDER_FACTOR, 
                       TRIP_PROBABILITY_BY_DISTANCE)
from enum import Enum

class AgentRules:
    @staticmethod
    def get_trip_probability(agent, source, destination):
        """
        Outputs the probability fo a given agent making a trip from a source node to a destination node.
        agent       - The agent who will make (or not make the trip)
        source      - The source node from where the agent will make (or not make) the trip
        destination - The destination to where the agent will make (or not make) the trip
        """
        assert source.superenvs == destination
        if agent.model == source:
            return 1

        current_node = agent.model
        distance_from_lowest_level = 0
        
        while current_node != source:
            current_node = current_node.superenvs
            distance_from_lowest_level += 1

        return TRIP_PROBABILITY_BY_DISTANCE[distance_from_lowest_level]

    @staticmethod
    def nodes_to_visit(agent):
        current_node = agent.model

        visited_nodes = []
        while current_node is not None:
            next_node = current_node.superenvs
            if np.random.random_sample() <= AgentRules.get_trip_probability(agent, current_node, next_node):
                visited_nodes.append(next_node)
            current_node = next_node

        return visited_nodes

    @staticmethod
    def family_to_migrate_to(agent):
        pass

    @staticmethod
    def should_agent_migrate(agent):
        pass


class DiseaseRules:
    @staticmethod
    def new_disease_state(agent, contacts) -> Enum:
        """
        compute the new disease state for an agent given the list of
        other agents he/she has came in contact with in the past day

        the chances of getting exposed to the virus depends on agent's
        health, age, gender and many other factors

        for simplicity we consider only agent's age and gender
        """
        if agent.state == STATES.S:
            # for each infected person agent meets
            # chances of getting infected depends on contact rate for the person
            risk = 0
            id1 = age_to_age_group(agent.age)
            for person in contacts:
                id2 = age_to_age_group(person.age)
                # normalized contact rate
                contact_rate = CONTACT_MATRIX[id1][id2] # contact rate
                # if agent and person are of same gender then they have higher chance of getting infected
                if agent.gender == person.gender:
                    contact_rate *= GENDER_FACTOR
                risk += contact_rate
            # now decide if the agent got infected with a single coin flip
            if np.random.uniform(0, 1) <= 1-np.exp(-risk):
                return STATES.E
            else:
                return STATES.S

        if agent.state == STATES.E:
            # if a person is already exposed, he/she will either stay exposed
            # or transition to infected regardless of additional contacts
            prob_trans = 1-np.exp(-1/T_INC)
            if np.random.uniform(0, 1) <= prob_trans: # change state
                if np.random.uniform(0, 1) <= PROB_MILD:
                    return STATES.I_mild
                else:
                    return STATES.I_wild
            else:
                return STATES.E

        if agent.state == STATES.I_mild:
            # if a person is infected already, he/she can either stay infected
            # or get removed
            prob_trans = 1-np.exp(-1/T_INF_MILD)
            if np.random.uniform(0, 1) <= prob_trans:
                return STATES.R
            else:
                return STATES.I_mild
        if agent.state == STATES.I_wild:
            prob_trans = 1-np.exp(-1/T_INF_WILD)
            if np.random.uniform(0, 1) <= prob_trans:
                return STATES.R
            else:
                return STATES.I_wild
        if agent.state == STATES.R:
            # if a person is removed, he/she is immune to
            # the disease
            return STATES.R
