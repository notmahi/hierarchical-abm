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
from rules import AgentRules, DiseaseRules
from constants import *
from utils.generate_population import PopulationEngine

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
    contacts (List):            Track of the contacts an agent had in a day
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
        self.model = model

        self.uid = uuid.uuid4()

        # Keeping track of the contacts that 
        self.contacts = []

    def set_model(self, model):
        """
        Helper function for setting a person's model (aka a family) and 
        computing the hierarchy tree.
        """
        if self.model is None:
            self.model = model
        # To simplify computation, this is the hierarchy tree this model is a
        # part of
        self.hierarchy_tree = [self.model]
        model_now = self.model
        while model_now.superenv is not None:
            # Build out the whole hierarchy so we can tell where this particular
            # agent is located
            self.hierarchy_tree.append(model_now.superenv)
            model_now = model_now.superenv

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
        if nodes_to_visit:
            for node in nodes_to_visit:
                node.register_contact(register_contact)
        
    def possibly_migrate(self):
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
        
    def register_contacts(self, contacts):
        """
        Given a list of contacts a person had in a day in some environment, 
        update the agents' contact list.
        """
        self.contacts.extend(contacts)

    def process_contacts_and_update_disease_state(self):
        """
        Given the final list of contacts the agent had in a day, process them
        to update the agent's disease state.
        """
        self.state = DiseaseRules.new_disease_state(self, self.contacts)
        # empty out contacts at the end of the day.
        self.contacts = []
        return self.state


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
    def __init__(self, subenvs, superenvs, population, area, GISmap=None):
        self.subenvs = subenvs
        self.superenvs = superenvs
        self.population = population
        self.area = area
        self.GISmap = None # Will have to decide how to handle later

        self.pop_density = self.population / self.area
        self.visits = set()

        # Node-level information relevant when this model is part of a hierarchy
        self.node_level = None
        self.node_hash = None
        self.is_main_node = False

    def step(self, contact_simulation):
        """ 
        Environment steps happen in two stages: first, the subenvironments all
        make their steps. Once they are done, this environment makes its step.
        """
        # Removed for flattened execution.
        # for subenv in self.subenvs:
        #     subenv.step()
        self.own_step(contact_simulation)
        self.clean_up_contacts()

    @classmethod
    def from_data(cls, hierarchy_node, tree_data, parent, hierarchy=None):
        """
        Create an EnvironmentModel from hierarchical data recursively.
        Warning: When using this method, the subenvs field is empty. That 
        must be filled in manually later.
        """
        subenvs, statistical_args = cls.parse_data(tree_data.loc[hierarchy_node.node_hash])
        # Warning: for intermediate levels, subenvs should be empty.
        # Thus, they must be filled in seperately!
        env = cls(subenvs, parent, *statistical_args)
        env.node_hash = hierarchy_node.node_hash
        env.node_level = hierarchy_node.node_level
        env.is_main_node = hierarchy_node.is_main_node
        return env

    @staticmethod
    def parse_data(hierarchy_data_row, parent=None):
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
        self.visits.add(agent)

    @property
    def leaving_probability(self):
        """
        We want leaving probability to be dynamic, possibly, which is why we keep
        it flexible.
        """
        # TODO (mahi): revisit this to make it more realistic.
        # raise NotImplementedError('You must define subenvironment steps.')
        return 1.0

    def clean_up_contacts(self):
        """
        Helper function to (probabilisitically) clean up the visits from the day
        """
        # Keep someone already in the env with probability 1-leaving_probability
        self.visits = set([x for x in self.visits if np.random.random_sample() > self.leaving_probability])

    def own_step(self, contact_simulation):
        """
        We take this step once the steps at a subenv level has been completed.
        We can think of this event as "what happened in the district" on a day
        or other predefined timestep.

        Params:
        contact_simulation (Callable: List(Person) -> Dict[Person, List(Person)])
        """
        # raise NotImplementedError('You must define subenvironment steps.')
        # Step 1: simulate a round of contacts.
        people_contacts = contact_simulation(list(self.visits))
        # Step 2: register that round of contacts with the people.
        for (person, contacts) in people_contacts.items():
            person.register_contacts(contacts)


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
        super().__init__(people, superenv, self.population, area, None)

    @property
    def leaving_probability(self):
        # You don't leave your family
        return 0.0

    def own_step(self):
        pass

    def register_hierarchy(self, hierarchy):
        """
        Helper method to register all the people with the hierarchy.
        """
        for person in self.people:
            hierarchy.register_person(person)


class LowestLevelEnv(EnvironmentModel):
    """
    This is an abstract class to denote the lowest level available for models 
    where the statistics is available, from which we run our population 
    generation procedure. In the class hierarchy, this class should always be
    a parent to FamilyEnv.
    """
    @classmethod
    def from_data(cls, hierarchy_node, tree_data, parent, hierarchy):
        """
        Overriding from_data here because the families need to get created too.
        """
        subenvs, statistical_args = cls.parse_data(tree_data.loc[hierarchy_node.node_hash])
        # Warning: for intermediate levels, subenvs should be empty.
        # Thus, they must be filled in seperately!
        env = cls(subenvs, parent, *statistical_args)
        env.node_hash = hierarchy_node.node_hash
        env.node_level = hierarchy_node.node_level
        for subenv in subenvs:
            subenv.superenv = env
            subenv.register_hierarchy(hierarchy) # Register the population
        return env


    @staticmethod
    def parse_data(hierarchy_data_row):
        """
        Parse the statistical data, generate the basic information about this 
        env and pass along.
        """
        population = hierarchy_data_row['pop_total']
        area = hierarchy_data_row['area']
        # Now, we must get use the population generation methods to fill in and
        # create households and agents, populate the agents in the FamilyEnv
        # and return the results.

        population_engine = PopulationEngine(hierarchy_data_row, seed=12)
        # TODO (mahi): The following will break if the data schema changes, so
        # make sure the schema is the same in real data, or make changes there.
        people, households = population_engine.get_people_and_households()
        
        families = []
        for household in households.itertuples():
            # Generate all the people first
            family_members = []
            for people_id in household.members:
                person = people.loc[people_id]
                member = Person(gender=MALE if person.sex == 'm' else FEMALE,
                                age=person.age,
                                earning=household.income,
                                location=None, # TODO: How should we change?
                                model=None, #filled in later
                                )
                family_members.append(member)
            family = FamilyEnv(family_members, superenv=None)
            for member in family_members:
                member.set_model(family)
            families.append(family)

        return families, (population, area)


class IntermediateLevelEnv(EnvironmentModel):
    """
    Abstract class to denote an intermediate level of environment. Basically, on
    this level the population generation process is passed down to subenvs of 
    this class.
    """
    @staticmethod
    def parse_data(hierarchy_data_row):
        """
        Parse the statistical data, generate the basic information about this 
        env and pass along. We don't fill up subenv because it will be filled
        in in the HierarchicalDataTree construction.
        """
        population = hierarchy_data_row['pop_total']
        area = hierarchy_data_row['area']
        return [], (population, area)
