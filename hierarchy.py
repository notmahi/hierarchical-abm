"""
hierarchy.py builds a basic hierarchy where the simulation takes place.

Given these sources of information, Hierachy will automatically build a
hierarchical ABM model. The API design has been inspired by PyTorch's sequential
model building.
"""

from collections import deque

from data import HierarchicalDataTree
from model import EnvironmentModel

class HierachicalModel(EnvironmentModel):
    """
    A hierarchy is an ABM environmental model that encompasses our hierarchical
    structure. This Model takes care of building this class from scratch.

    hierarchy_data (HierarchicalDataTree): This source designates where the 
                                           hierarchical data is coming from.
    hierarchy_models (list of EnvironmentModel): Tells us to use different env 
                                        models at different levels of hierarchy.
    """
    def __init__(self, hierarchy_data, hierarchy_models):
        self.hierarchy_models = hierarchy_models
        self.hierarchy_data = hierarchy_data
        self._verify()
        self.final_model = self._build()

    def _verify(self):
        assert isinstance(self.hierarchy_data, HierarchicalDataTree)
        for model in self.hierarchy_models:
            assert isinstance(model, EnvironmentModel)

    def _build(self):
        """
        We recursively build a tree by first building a environment, and then 
        building its subenvironments recursively.
        """
        root_node = self.hierarchy_data.tree_root
        # process queue has the general format: node_now, model_now, parent
        process_queue = deque([(root_node, 
                                hierarchy_models[root_node.node_level], 
                                None)])
        
        final_model = None
        while len(process_queue) > 0:
            node_now, model_now, parent = process_queue.pop()
            env = model_now.from_data(node_now, self.hierarchy_data, parent)
            if parent is not None:
                parent.subenvs.append(env)
            else:
                final_model = env
            for subnode in node_now.sub_nodes:
                process_queue.append((subnode, 
                                      hierarchy_models[subnode.node_level], 
                                      env))
                                    
        return final_model

    def step(self):
        """
        The step method necessary for the hierarchical model. Simply, this would
        consist of a step on the root or "country level" model.

        TODO (Mahi): Consider if we should aggregate the statistics here.
        """
        self.final_model.step()

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