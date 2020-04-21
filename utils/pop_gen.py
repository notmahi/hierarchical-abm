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
        extra_age = (extra_age * np.arange(count)) // count
        ages = base_age + extra_age
        sexes = np.array(['f'] * females + ['m'] * males)
        np_random.shuffle(sexes)
        dfs.append(pd.DataFrame({'age':ages,'sex':sexes}))
    df = pd.concat(dfs)
    # Now, randomize their location
    df = df.sample(frac=1,random_state=RANDOM_STATE).reset_index(drop=True)
    return df
 

def gen_couples(people, marriage_stats):
    """
    Given a population dataframe, generate pairings among them following some 
    particular heuristics.

    people (DataFrame): A dataframe of people's age and sex
    marriage_stats (dict-like): Basically, given age, returns people's 
    probability of being married.
    """
    people['partner'] = [-1] * len(people)
    couples = []
    people.partner = people.apply(lambda x: -1 if np.random.random() < marriage_stats[x.age] else 0)
    # All kids are unmarried/self partnered
    people.loc[people.partner == 0, 'partner'] = people.loc[people.partner == 0].index    
    male_pop = people[people.sex == 'm']
    female_pop = people[people.sex == 'f']
    for index_1 in people.index:
        if people.loc[index_1].partner != -1:
            continue
        p1 = people.loc[index_1]
        age = p1['age']
        # Filter for minimal age and sex
        potential_partner = people[((people.age > 14) & 
                                    (people.sex != p1.sex) &
                                    (people.partner == -1)
                                   )]
        # Heuristic:
        # First, we search for partners in the 2-10 years age difference
        # Then we go for 11-20 years
        # Then, we just pick any woman of age.
        multiplier = 1 if p1.sex == 'm' else -1
        age_range_1 = range(p1.age - 10*multiplier, p1.age - 2*multiplier, multiplier)
        age_range_2 = range(p1.age - 20*multiplier, p1.age - 11*multiplier, multiplier)

        filtered_index = potential_partner[potential_partner.age.isin(age_range_1)]
        if filtered_index.empty:
            filtered_index = potential_partner[potential_partner.age.isin(age_range_2)]
        if filtered_index.empty:
            filtered_index = potential_partner
        if filtered_index.empty:
            # There are truly no eligible bachelors/bachelorettes
            people.loc[index_1].partner = index_1
        else:
            # Set them to be partnered up
            # Since people are randomly ordered, this is also random
            index_2 = filtered_index.index[0]
            people.loc[index_1].partner = index_2
            people.loc[index_2].partner = index_1
            couples.append((index_1, index_2))
    return couples

def assign_households(people, household_stats):
    """
    Given people with their couple status, generate households.
    """
    pass


if __name__ == '__main__':
    N = 50
    population_ranges = [(20, 24), (25, 29)]
    age_densities = [0.2, 0.8]
    sex_ratios_by_range = [(0.5, 0.5), (0.1, 0.9)]
    (generate_people(N, population_ranges, age_densities, sex_ratios_by_range))