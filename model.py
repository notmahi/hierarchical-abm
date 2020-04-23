"""
model.py contains the generic units of our agent based model, namely the people
units and the social units.

** IMPORTANT **
For our code to work as expected, the model has to be created top down, which
means the root node (or the highest level of region where we are simulating) 
must be defined first, then its subregions, and so on, until we reach the person
level.
"""

import uuid
import numpy as np

from mesa import Agent, Model

from rules import AgentRules
from constants import *

class Person(Agent):
    """
    The simplest iteration of a person. Basically, this agent shall have all the
    characteristics of the singular person defined, based on which our model 
    will run the simulations.

    For now, we consider the following properties of the person:

    gender (constants):         (MALE, FEMALE)
    age (int):                  Years
    earnings (int):             1000 units of currency, yearly
    location (TBD):             Probably lat/long, for visualization
    model (EnvironmentModel):   The hierarchical model that this agent is 
                                directly a part of.
    state (constants.state):    Which state of the disease is this person on 
                                right now?
    UID (str):                  Each agent is given a single UID to make 
                                identification and location easy.
    is_urban (boolean):         Signify whether the agent is in a rural or urban
                                setting
    """
    def __init__(self, gender, age, earning, location, model, state = None,
                 is_urban = False):
        self.gender = gender
        self.age = age
        self.earning = earning
        self.location = location
        # For individual agents this should be their family
        self.model = model
        # Every agent starts off in the susceptible state, unless defined
        self.state = state or STATES.S
        self.is_urban = is_urban
        # To simplify computation, this is the hierarchy tree this model is a
        # part of
        self.hierarchy_tree = [self.model]
        env_now = self.model
        while model_now.superenv is not None:
            # Build out the whole hierarchy so we can tell where this particular
            # agent is located
            self.hierarchy_tree.append(model_now.superenv)
            model_now = model_now.superenv

        self.uid = uuid.uuid4()

    def step(self):
        """
        In our model, the agent can do one of two things in a day.
        1. The agent can decide to visit (temporarily) some node in the tree.
        2. The agent can also decide to move to some other node.
        """
        self.visit_tree()
        self.possibly_migrate()

    def visit_tree(self):
        """
        This function will need two methods.
        To define a random distribution over the tree, we can think of walking 
        up the tree, and then walking down.
        So, given the agent's disposition, we can think how far up the tree the
        agent is going, and from there how far down the tree the agent will move
        """
        nodes_to_visit = AgentRules.nodes_to_visit(self)
        for node in nodes_to_visit:
            node.register_contact(register_contact)
        
    def possibly_migrate():
        """
        This function considers each agent, and with some low probability, helps
        the agent to decide which household this agent will migrate to.
        """
        if AgentRules.should_agent_migrate(self):
            self.model = AgentRules.family_to_migrate_to(self)
            model_now = self.model
            self.hierarchy_tree = [self.model]
            while model_now.superenv is not None:
                # Build out the whole hierarchy so we can tell where this
                # particular agent is located
                self.hierarchy_tree.append(model_now.superenv)
                model_now = model_now.superenv
        



class EnvironmentModel(Model):
    """
    This is simply an abstract class which will encode all the hierarchies in
    the model. In a simplified way, we can think of an environment model as a 
    unit that holds people or other sub-environments. For example, a union will
    contain villages, a village can contain families, while the families will 
    contain people. The only way the environments will change is how the inter-
    action rules for them will differ.

    As a general rule, this class is an internal representation that shall not
    be directly used as a model. Instead, the subclasses that are defined below
    should be used.

    For now, we will use the following properties in an environment:

    subenvs (EnvironmentModels):    Other environments that are lower in the 
                                    hierarchy from this model
    superenv (EnvironmentModels):   The singular model this env is a subenv of
    population (int):               Total number of people who are a part of 
                                    this environment
    area (float):                   Area of this environment
    population_density (float):     Simply population/area, for the spatial sim
    GISmap (GIS object):            For visualization, the GIS object for this 
                                    environment. We will decide on particulars
                                    later.
    visits (List):                  Environments are representative of physical
                                    spaces, and thus we keep track of who 
                                    visited this space in a day and simulate it
                                    based on spatial ABMs.
    """
    def __init__(self, subenvs, superenvs, population, area, GISmap):
        self.subenvs = subenvs
        self.superenvs = superenvs
        self.population = population
        self.area = area
        self.GISmap = None # Will have to decide how to handle later

        self.pop_density = self.population / self.area
        self.visits = set()

    def step(self):
        """ 
        Environment steps happen in two stages: first, the subenvironments all
        make their steps. Once they are done, this environment makes its step.
        """
        for subenv in self.subenvs:
            subenv.step()
        self.own_step()
        self.clean_up_contacts()

    @classmethod
    def from_data(cls, hierarchy_node, tree_data, parent):
        """
        Create an EnvironmentModel from hierarchical data recursively.
        Warning: When using this method, the subenvs field is empty. That 
        must be filled in manually later.
        """
        statistical_args = cls.parse_data(tree_data[hierarchy_node.node_hash])
        # Warning: Must fill in subenvs seperately!
        return cls([], parent, *statistical_args)

    @staticmethod
    def parse_data(hierarchy_data_row):
        """
        Given a row of statistical data in a row, this will extract the 
        necessary information as arguments and return that.
        """
        raise NotImplementedError('You must define subenvironment steps.')

    def register_contact(self, agent):
        """
        To keep track of all the people who are visiting this environment 
        through the day, we use this function that lets the agent attach its
        uid to this particular environment
        """
        self.visits.add(agent.uid)

    @property
    def leaving_probability(self):
        """
        We want leaving probability to be dynamic, possibly, which is why we keep
        it flexible.
        """
        raise NotImplementedError('You must define subenvironment steps.')

    def clean_up_contacts(self):
        """
        Helper function to (probabilisitically) clean up the visits from the day
        """
        # Keep someone already in the env with probability 1-leaving_probability
        self.visits = set([x for x in self.visits if np.random.random_sample() > self.leaving_probability])

    def own_step(self):
        """
        We take this step once the steps at a subenv level has been completed.
        We can think of this event as "what happened in the district" on a day
        or other predefined timestep.
        """
        raise NotImplementedError('You must define subenvironment steps.')


class FamilyEnv(EnvironmentModel):
    """
    Modeling a family based on our hierarchical agent based model setup. 
    Families or households are the lowest rung on our hierarchy, which are 
    composed of person agents.
    We used an average household area of 50 sq meters
    """
    def __init__(self, people, superenv, area=5.0e-5):
        self.population = len(people)
        self.people = people
        self.superenv = superenv
        self.area = area
        super().__init__(people, superenv, population, area, None)

    @property
    def leaving_probability(self):
        # You don't leave your family
        return 0.0

    def own_step(self):
        pass

    @staticmethod
    def create_family(total_size, total_income, gender_dist):
    """
    Helper method to create a family
    """
        pass

