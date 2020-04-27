"""
hierarchy.py builds a basic hierarchy where the simulation takes place.

Given these sources of information, Hierachy will automatically build a
hierarchical ABM model. The API design has been inspired by PyTorch's sequential
model building.
"""

from collections import deque
from functools import partial

from data import HierarchicalDataTree
from model import EnvironmentModel
from simulation import simulate

class HierarchicalModel(EnvironmentModel):
    """
    A hierarchy is an ABM environmental model that encompasses our hierarchical
    structure. This Model takes care of building this class from scratch.

    hierarchy_data (HierarchicalDataTree): This source designates where the 
                                           hierarchical data is coming from.
    hierarchy_models (list of EnvironmentModel): Tells us to use different env 
                                        models at different levels of hierarchy.
    """
    def __init__(self, hierarchy_data, hierarchy_models, contact_matrix):
        self.hierarchy_models = hierarchy_models
        self.hierarchy_data = hierarchy_data
        self.contact_matrix = contact_matrix
        self.contact_simulation = partial(simulate, contact_matrix=contact_matrix)
        # These are the large structures to keep track of all envs and people
        self._people = {}
        self._envs = {}
        
        self._verify()
        self.final_model = self._build()


    def _verify(self):
        assert isinstance(self.hierarchy_data, HierarchicalDataTree)
        for model in self.hierarchy_models:
            assert issubclass(model, EnvironmentModel), f'{model} isn\' an envi\
                ronmental model'

    def _build(self):
        """
        We recursively build a tree by first building a environment, and then 
        building its subenvironments recursively.
        """
        root_node = self.hierarchy_data.tree_root
        # process queue has the general format: node_now, model_now, parent
        process_queue = deque([(root_node, 
                                self.hierarchy_models[root_node.node_level], 
                                None)])
        
        final_model = None
        while len(process_queue) > 0:
            node_now, model_now, parent = process_queue.pop()
            env = model_now.from_data(node_now, self.hierarchy_data.tree_data, 
                                      parent, self)
            # attach this env to the hierarchy
            self._envs[env.node_hash] = env
            if parent is not None:
                parent.subenvs.append(env)
            else:
                final_model = env
            for subnode in node_now.sub_nodes:
                process_queue.append((subnode, 
                                      self.hierarchy_models[subnode.node_level], 
                                      env))
                                    
        return final_model

    def step(self):
        """
        The step method necessary for the hierarchical model. Simply, this would
        consist of a step on the root or "country level" model.

        TODO (Mahi): Consider if we should aggregate the statistics here.
        """
        # For parallelization, we do not run the model in a tree-like manner
        # self.final_model.step()
        # Instead, we run the steps in loops

        # First, we take steps for every person, which is populating the 
        # ABM hierarchy tree based on what they visited that day.
        # TODO (tmoon): parallelize
        for (uuid, person) in self._people.items():
            person.step()

        # Then, once the tree's contacts are populated, run their individual 
        # steps.
        # TODO (tmoon): parallelize
        for (node_hash, env) in self._envs.items():
            env.step(self.contact_simulation)

        # Once the environments have processed the contacts, update the 
        # individual's disease progression.
        # TODO (tmoon): parallelize
        for (uuid, person) in self._people.items():
            person.process_contacts_and_update_disease_state()


    def register_person(self, person):
        """
        Function to register the population of the hierarchy
        """
        self._people[person.uid] = person

    def get_summary_statistics(self):
        """
        Get summary statistic of district-wise disease progression.
        """
        pass

    def get_full_statistics(self):
        """
        Get a full report of the country population (each person is a column)
        including:
        1. Final disease state of the person.
        2. Date of state changes.
        3. People that this person spread to.
        """
        pass

    def seed(self, seed_params):
        pass