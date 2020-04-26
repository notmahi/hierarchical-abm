"""
run_model.py is the main executable file for the model. Basically, this file is
responsible for parsing the raw data given to the model using data.py,
importing the necessary environments and agents from bd_envs and model.py, and
combining all of them with hierarchy.py, and then running through the steps and
aggregating the data at the necessary level.
"""

import argparse

from hierarchy import HierachicalModel
from data import HierarchicalDataTree
from bd_envs import *

"""
TODO (mahi): This is placeholder code since we do not have the data yet. But we
are trying to create approximately what the information may look like.
"""

parser = argparse.ArgumentParser(description='Run an agent-based model for the \
    COVID-19 infected data.')

necessary = parser.add_argument_group('Mandatory', 'Necessary data for every \
    run of the simulation.')
necessary.add_argument('--hierarchy_tree', type=str, help='Relative path for the\
    hierarchy network. (*.json)')
necessary.add_argument('--tree_nodes', type=str, help='Relative path for the table\
    containing information about the tree nodes. (*.csv)')
necessary.add_argument('--contact_matrix', type=str, help='Relative path for load\
    ing the contact matrix file.')

io = parser.add_argument_group('Input/output', 'Input/output options')
io.add_argument('--out_dir', type=str, help='Output location.')
io.add_argument('--resume_from', type=str, help='Saved state data to load from')

optional = parser.add_argument_group('Optional', 'Arguments with some default \
    in place.')
optional.add_argument('--age_pyramid', type=str, default='data\bangladesh_populat\
    ion_pyramid_2017.csv', help='Relative path for age pyramid.')
optional.add_argument('--marriage_data', type=str, default='data\marriage_data_bd\
    .csv', help='Age-wise marriage data for population generation.')
parser.parse_args()

# Step 1: Parse the data into two forms: a nested dict for the tree structure
#         and a dataframe for the node-level statistics
# TODO (mahi): Load these from provided data files.
level_hierarchy = None
tree_data = None
tree_dict = None

# Step 2: Use the parsed data to create a HierarchicalModel which is able to 
#         run the simulation

data_holder = HierarchicalDataTree(tree_dict=tree_dict,
                                   level_hierarchy=level_hierarchy, 
                                   tree_data=tree_data)

# Has to be the same order as level_hierarcht
organizational_levels = (
    DivisionEnv,
    ZillaEnv, 
    UpazillaEnv,
    UnionEnv,
    MahallaEnv,
    VillageEnv
)

model = HierarchicalModel(data_holder, organizational_levels)

# Step 3: Seed the simulation, initialize the disease state in some individuals.
# TODO: (figure out model seeding parameters)
seed_params = None
model.seed(seed_params)

# Step 4: Run the simulation for T days where T is the given number of days
#         and generating summary statistics every day.
for time in range(T):
    model.step()
    summary_stats = model.get_summary_statistics()
    # TODO (mahi): save the summary stats

# Step 5: At the end of the simulation, generate a full statistics about the
#         state of the people at time T.
full_stats = model.get_full_statistics()
# TODO (mahi): save the full statistics.