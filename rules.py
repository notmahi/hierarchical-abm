"""
rules.py defines all the rules for the agents movement. In short, to change the
behavior of the agent, we should only change the functions here.

Every rule should be made in a functional way so that they can switched easily
later, and so that we can add more parameters to agent behavior easily.
"""

import numpy as np


class AgentRules:
    @staticmethod
    def nodes_to_visit(agent):
        pass

    @staticmethod
    def family_to_migrate_to(agent):
        pass

    @staticmethod
    def should_agent_migrate(agent):
        pass


class DiseaseRules:
    @staticmethod
    def new_disease_state(agent, num_contacts):
        pass

