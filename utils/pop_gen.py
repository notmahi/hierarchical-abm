"""
pop_gen.py generates a sample population for a given area.

The sample statistics that we have for a area is:
1. ratio of sex,
2. ratio of age groups,
3. category of income.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats


RANDOM_STATE = 123
np_random = np.random.RandomState(seed=RANDOM_STATE)

def generate_people(N, population_ranges, age_densities, sex_ratios_by_range):
    """
    Generate a population dataframe of ages and sexes.
    Within an age range, we assume uniform distribution of ages, which may not
    always be true.

    Parameters:
    population_ranges (list of tuples): A list of tuples with (low, high) ages
    age_densities (list of floats): Each float designates the percentage of 
    people in that age range.
    sex_ratios_by_range (list of tuples): per age range, ratio of (m, f) sex
    """
    dfs = []
    for age_range, density, sex_ratio in zip(population_ranges, age_densities, sex_ratios_by_range):
        count = int(N * density + 0.5)
        males = int(count * sex_ratio[0] + 0.5)
        females = count - males
        base_age = np.array([age_range[0]] * count)
        extra_age = np.array([age_range[1] - age_range[0] + 1] * count)
        print(extra_age)
        extra_age = (extra_age * np.arange(count)) // count
        ages = base_age + extra_age
        sexes = np.array(['f'] * females + ['m'] * males)
        np_random.shuffle(sexes)
        dfs.append(pd.DataFrame({'age':ages,'sex':sexes}))
    df = pd.concat(dfs)
    # Now, randomize their location
    df = df.sample(frac=1,random_state=RANDOM_STATE).reset_index(drop=True)
    return df
 

def assign_households(people, household_stats):
    """
    
    """
    pass


if __name__ == '__main__':
    N = 50
    population_ranges = [(0, 5), (6, 10)]
    age_densities = [0.2, 0.8]
    sex_ratios_by_range = [(0.5, 0.5), (0.1, 0.9)]
    (generate_people(N, population_ranges, age_densities, sex_ratios_by_range))