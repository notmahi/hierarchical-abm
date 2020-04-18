"""
model.py contains the generic units of our agent based model, namely the people
units and the social units.
"""

from mesa import Agent, Model
from constants import *

class Person(Agent):
    """
    The simplest iteration of a person. Basically, this agent shall have all the
    characteristics of the singular person defined, based on which our model 
    will run the simulations.

    For now, we consider the following properties of the person:

    gender (constants):         (MALE, FEMALE)
    age (int):                  Years
    earnings (int):             Units of currency, yearly
    location (TBD):             Probably lat/long, for visualization
    model (EnvironmentModel):   The hierarchical model that this agent is 
                                directly a part of.
    state (constants.state):    Which state of the disease is this person on 
                                right now?
    """
    def __init__(self, gender, age, earning, location, model, state=None):
        self.gender = gender
        self.age = age
        self.earning = earning
        self.location = location
        # For individual agents this should be their family
        self.model = model
        # Every agent starts off in the susceptible state, unless defined
        self.state = state or COVID_state.S
        # To simplify computation, this is the hierarchy tree this model is a
        # part of
        self.hierarchy_tree = 

    def step(self):
        """ What does this agent do in one day? """
        pass


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
    """
    def __init__(subenvs, superenvs, population, area, GISmap):
        self.subenvs = subenvs
        self.superenvs = superenvs
        self.population = population
        self.area = area
        self.GISmap = None # Will have to decide how to handle later

        self.pop_density = self.population / self.area

    def step(self):
        """ 
        Environment steps happen in two stages: first, the subenvironments all
        make their steps. Once they are done, this environment makes its step.
        """
        for subenv in self.subenvs:
            subenv.step()
        self.own_step()

    def own_step(self):
        raise NotImplementedError('You must define subenvironment steps.')



