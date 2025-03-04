"""
data.py parses the data and creates a hierarchical ABM based on that. The
code consumes data in two parts, first the tree structure has to be given as a 
python nested dictionary, and secondly, each node in that nested dictionary
needs to have the information in a row in a dataframe.

Sample usage:

Level hierarchy in Bangladesh

level_hierarchy = { 
    'zilla': 0
    'upozilla': 1
    'union': 2
    'mahalla': 3
    'village': 4
}

graph = {
    "node_level": 0,
    "node_hash" : "01",
    "sub_nodes": [{
        "node_level": 1,
        "node_hash" : "0101",
        "sub_nodes": []
    },
    {
        "node_level": 1,
        "node_hash" : "0102",
        "sub_nodes": []
    }]
}
"""

import json
import pandas as pd


class HierarchicalDataNode:
    """
    Signify each node in the hierarchy, from a data perspective.
    """
    def __init__(self, node_hash, node_level, sub_nodes):
        self.node_hash = node_hash
        self.node_level = node_level
        self.sub_nodes = sub_nodes
        self.is_main_node = False

    def set_parent(self, parent):
        self.node_parent = parent

    def set_main_node(self):
        self.is_main_node = True

    @classmethod
    def from_dict(cls, node_dict, node_parent=None):
        subnodes = [HierarchicalDataNode.from_dict(subnode_dict) for subnode_dict in node_dict['sub_nodes']]
        node_now = cls(node_dict['node_hash'], node_dict['node_level'], subnodes)
        main_node_hash = node_dict['main_sub_node_hash']
        for node in subnodes:
            if node.node_hash == main_node_hash:
                node.set_main_node()
            node.set_parent(node_now)
        return node_now


class HierarchicalDataTree:
    """
    Signify the tree in the hierarchy, from a data perspective, including metadata.
    """
    def __init__(self, tree_dict, level_hierarchy, tree_data):
        self.tree_data = tree_data
        self.level_hierarchy = level_hierarchy
        self.tree_root = HierarchicalDataNode.from_dict(tree_dict)

