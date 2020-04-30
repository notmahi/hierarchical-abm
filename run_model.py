"""
run_model.py is the main executable file for the model. Basically, this file is
responsible for parsing the raw data given to the model using data.py,
importing the necessary environments and agents from bd_envs and model.py, and
combining all of them with hierarchy.py, and then running through the steps and
aggregating the data at the necessary level.
"""

import argparse
import json
import os
import time

import pandas as pd

from hierarchy import HierarchicalModel
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
necessary.add_argument('--steps', type=int, help='Number of steps to run.')
necessary.add_argument('--hierarchy_tree', type=str, help='Relative path for \
    the hierarchy network. (*.json)')
necessary.add_argument('--nodes_data', type=str, help='Relative path for the \
    table containing information about the tree nodes. (*.csv)')
necessary.add_argument('--contact_matrix', type=str, help='Relative path for \
    loading the contact matrix file.', default='data_files/bd_mu_all_loc.csv')

io = parser.add_argument_group('Input/output', 'Input/output options')
io.add_argument('--out_dir', type=str, help='Output location.')
io.add_argument('--resume_from', type=str, help='Saved state data to load from')

optional = parser.add_argument_group('Optional', 'Arguments with some default \
    in place.')
optional.add_argument('--age_pyramid', type=str, default='data_files/bangladesh\
    _population_pyramid_2017.csv', help='Relative path for age pyramid.')
optional.add_argument('--marriage_data', type=str, default='data_files/marriage\
    _data_bd.csv', help='Age-wise marriage data for population generation.')

args = parser.parse_args()

# Step 1: Parse the data into two forms: a nested dict for the tree structure
#         and a dataframe for the node-level statistics
# TODO (mahi): Load these from provided data files.
begin = time.perf_counter()
level_hierarchy = None
tree_data = pd.read_csv(args.nodes_data).set_index('node_hash')
with open(args.hierarchy_tree, 'r') as f:
    tree_dict = json.load(f)

contact_matrix = pd.read_csv(args.contact_matrix).set_index('Age group').to_numpy()
loading_data_files = time.perf_counter()
print(f'Loaded data files, {loading_data_files - begin} seconds.')

# Step 2: Use the parsed data to create a HierarchicalModel which is able to 
#         run the simulation

data_holder = HierarchicalDataTree(tree_dict=tree_dict,
                                   level_hierarchy=level_hierarchy, 
                                   tree_data=tree_data)

# Has to be the same order as level_hierarcht
organizational_levels = (
    # DivisionEnv,
    ZillaEnv, 
    UpazillaEnv,
    UnionEnv,
    MahallaEnv,
    VillageEnv
)

model = HierarchicalModel(data_holder, organizational_levels, contact_matrix)
loading_tree = time.perf_counter()
print(f'Loaded tree, {loading_tree - loading_data_files} seconds.')

# Step 3: Seed the simulation, initialize the disease state in some individuals.
# TODO: (figure out model seeding parameters)
seed_params = 1.71e-5
total_exposed = model.seed(seed_params)
print(f'Total exposed in seed: {total_exposed}')
seeding_time = time.perf_counter()
print(f'Seeded tree, {seeding_time - loading_tree} seconds.')

# Step 4: Run the simulation for T days where T is the given number of days
#         and generating summary statistics every day.
T = args.steps
for t in range(T):
    loop_begin_time = time.perf_counter()
    model.step()
    summary_stats = model.get_summary_statistics()
    # For now, just print the summary stats
    print(summary_stats)
    summary_fname = os.path.join(args.out_dir, f'summary_{t}.csv')
    summary_stats.to_csv(summary_fname)
    loop_end_time = time.perf_counter()
    print(f'Ran {t} loops, time: {loop_end_time - loop_begin_time}s')
    # TODO (mahi): save the summary stats

# Step 5: At the end of the simulation, generate a full statistics about the
#         state of the people at time T.

stats_begin_time = time.perf_counter()
full_stats = model.get_full_statistics()

full_stats_fname = os.path.join(args.out_dir, f'full_stats_{t}.csv')
full_stats.to_csv(full_stats_fname)

stats_end_time = time.perf_counter()

print(f'Statistics collection time: {stats_end_time - stats_begin_time}s')
# TODO (mahi): save the full statistics.