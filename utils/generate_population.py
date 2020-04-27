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

from .bd_data_parser import (extract_population_ranges_and_density, 
                             extract_household_stats,
                             get_marriage_statistics, 
                             get_age_vs_sex_ratios)


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
    for age_range, density, sex_ratio in zip(population_ranges, age_densities, sex_ratios_by_range.itertuples()):
        count = int(N * density + 0.5)
        males = int(count * sex_ratio.m + 0.5)
        females = count - males
        base_age = np.array([age_range[0]] * count)
        extra_age = np.array([age_range[1] - age_range[0] + 1] * count)
        extra_age = (extra_age * np.arange(count)) // count
        ages = (base_age + extra_age).astype(int)
        sexes = np.array(['f'] * females + ['m'] * males)
        np_random.shuffle(sexes)
        dfs.append(pd.DataFrame({'age':ages,'sex':sexes}))
    df = pd.concat(dfs)
    if not (df['age'].dtype == 'int64'):
        import pdb; pdb.set_trace()
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
    people.partner = people.apply(lambda x: -1 if np.random.random() < marriage_stats[(x.sex, x.age)] else 0,
                                  axis='columns')
    # All kids are unmarried/self partnered
    people.loc[people.partner == 0, 'partner'] = people.loc[people.partner == 0].index    
    male_pop = people[people.sex == 'm']
    female_pop = people[people.sex == 'f']
    for index_1 in people.index:
        if people.loc[index_1].partner == -1:
            p1 = people.loc[index_1]
            age = p1['age']
            # Filter for minimal age and sex
            potential_partner = people.loc[((people.age > 14) & 
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

            filtered_index = potential_partner.loc[potential_partner.age.isin(age_range_1)]
            if filtered_index.empty:
                filtered_index = potential_partner.loc[potential_partner.age.isin(age_range_2)]
            if filtered_index.empty:
                filtered_index = potential_partner

            if filtered_index.empty:
                # There are truly no eligible bachelors/bachelorettes
                people.at[index_1, 'partner'] = index_1
            else:
                # Set them to be partnered up
                # Since people are randomly ordered, this is also random
                index_2 = filtered_index.index[0]
                people.at[index_1, 'partner'] = index_2
                people.at[index_2, 'partner'] = index_1
                couples.append(tuple(sorted((index_1, index_2))))

    assert((people.partner == -1).any() == False)
    return couples


def assign_households(people, couples, household_stats):
    """
    Given people with their couple status, generate households.

    Our heuristic is the following:
    1. First, separate out the single person households, and assign the single 
    adults to them.
    2. Then assign young or old couples to two person households.
    3. Then, assign an initial round of couples to 3+ people households.
    4. Fill in some gaps with kids.
    5. Assign a second round of couples to fill in the gaps, and shift around 
    kids to make them all fit.

    Parameters:
    people (DataFrame): Containing people's age, sex, and marital relations 
    couples (list): A list of tuples with two spouse ids
    household_stats (list): List of number of households of the given size
    """
    households = pd.DataFrame()
    households['size'] = np.repeat(np.arange(len(household_stats)) + 1, 
                                   household_stats)
    households['free'] = np.array(households.size)
    households['members'] = [[] for i in range(len(households))]

    people['hh_id'] = -np.ones(len(people))

    broken_up_couples = []
    # First, fill out the single person households
    N_1 = household_stats[0]
    # Take adult single people
    potentials = people.loc[(people.age >= 18) & (people.partner == people.index)]
    if len(potentials) < N_1:
        unrestricted_potentials = people.loc[(people.partner == people.index)]
        potentials = pd.concat([potentials, unrestricted_potentials])
    if len(potentials) < N_1:
        unrestricted_potentials = people.loc[(people.partner != people.index)]
        potentials = pd.concat([potentials, unrestricted_potentials])
    for i, idx in enumerate(potentials.index):
        if i >= N_1:
            break
        if people.loc[idx].hh_id != -1:
            # They are already assigned, probably b/c spouses
            N_1 += 1
            continue
        households.loc[i].members.append(idx)
        households.loc[i].free -= 1
        people.loc[idx].hh_id = i
        if people.loc[idx].partner != idx:
            # This person was married, who got broken up into seperate household
            # Thus, assign spouse to a single person household too
            partner_id = people.loc[idx, 'partner']
            # Breaking up a couple cause they didn't have enough single people
            broken_up_couples.append((min(idx, partner_id), max(idx, partner_id)))
            people.loc[idx, 'partner'] = idx
            people.loc[partner_id, 'partner'] = partner_id
            if len(households.loc[N_1 - 1].members) == 0:
                # There's no one in this household yet
                people.loc[partner_id].hh_id = N_1 - 1
                households.loc[N_1 - 1].members.append(partner_id)
                N_1 -= 1
                households.loc[N_1 - 1].free -= 1

    # Now, once we have filled out the single household quota, we start off by
    # giving every household a couple
    couple_idx = 0
    for i, idx in enumerate(households[households['size'] > 1].index):
        while couple_idx < len(couples) and couples[couple_idx] in broken_up_couples:
            couple_idx += 1
        if couple_idx == len(couples):
            # Ran out of couples, gotta fill the rest with kids and single people.
            break
        # We have a couple that is unassigned, so assign them to this house
        households.loc[idx].members.extend(couples[couple_idx])
        households.loc[idx].free -= 2
        for person in couples[couple_idx]:
            people.loc[person].hh_id = idx

    # Keep a count of unassigned couples and singles
    unassigned_total = len(people.loc[people.hh_id == -1])
    unassigned_married = people.loc[((people.hh_id == -1) & 
                                    (people.partner != people.index) )]
    unassigned_unmarried = people.loc[((people.hh_id == -1) & 
                                      (people.partner == people.index) )]

    # Now, fill in blank space in each household with 
    # 1. Kids
    # 2. Married couples,
    # 3. Unmarried adults/everyone else
    for i, idx in enumerate(households.loc[households.free > 0].index):
        # Kids have to be at least 16 years younger than their mother
        mom_age = min(people.loc[households.loc[idx].members].age)
        eligible_kids = people.loc[((people.hh_id == -1) & 
                                    (people.partner == people.index) &
                                    (people.age <= (mom_age - 16)))]
        kid_idx = 0
        for j in range(households.loc[idx].free):
            if np_random.random() < (len(eligible_kids) / unassigned_total):
                # With probability proportional to eligible kids, add kids
                if kid_idx < len(eligible_kids):
                    # There's still some kids to be had
                    id_of_kid = eligible_kids.index[kid_idx]
                    households.loc[idx].members.append(id_of_kid)
                    households.loc[idx].free -= 1
                    people.loc[id_of_kid].hh_id = idx

                    kid_idx += 1

    # Now, assign couples 
    unassigned_married_idx = 0
    for i, idx in enumerate(households.loc[households.free > 1].index):
        # Add married couples to the families
        while (unassigned_married_idx < len(unassigned_married) and
               people.loc[unassigned_married.index[unassigned_married_idx]].hh_id != -1):
            unassigned_married_idx += 1
        if unassigned_married_idx == len(unassigned_married):
            # We have reached the end of married couples
            break
        partner_1_id = unassigned_married.index[unassigned_married_idx]
        partner_2_id = people.loc[partner_1_id].partner
        households.at[idx, 'free'] = households.loc[idx].free - 2
        households.loc[idx].members.extend([partner_1_id, partner_2_id])
        people.at[partner_1_id, 'hh_id'] = idx
        people.at[partner_2_id, 'hh_id'] = idx

    # Now, whoever is left gets broken up and assigned randomly as long as 
    # people are left.
    for people_id in people.loc[people.hh_id == -1].index:
        # Household where there is space, or household where there is n+ people
        free_households = households.loc[(households.free > 0) | (households.size == household_stats[-1])]
        if free_households.empty:
            # No free household! Just choose a random one from the 8+ set
            household_to_assign_to = np_random.choice(households.index)
        else:
            household_to_assign_to = free_households.index[0]
        if people.loc[people_id].partner != people_id:
            # if they are not single break them up, ezpz
            partner_id = people.loc[people_id].partner
            people.at[people_id, 'partner'] = people_id
            people.at[partner_id, 'partner'] = partner_id
        people.at[people_id, 'hh_id'] = household_to_assign_to
        households.at[household_to_assign_to, 'free'] = households.loc[household_to_assign_to].free - 1
        households.loc[household_to_assign_to].members.append(people_id)

    # Now, no people should be left.
    assert len(people.loc[people.hh_id == -1]) == 0

    return people, households


def assign_income_to_households(households, household_income_stats):
    """
    Given income stats for households, add an 'income' column to households.
    """
    # TODO: Add the income column randomly to the households.
    households['income'] = 0
    return households


def generate_households_and_people(N, range_of_age, probability_of_age, 
                                   household_stats, household_income_stats):
    """
    Given all relevant statistics about a populace, sample a set of people with
    associated households.
    """
    # TODO: fix the global constants loading.
    sex_ratios_by_range = get_age_vs_sex_ratios()
    marriage_stats = get_marriage_statistics()
    # Otherwise this should fail.
    people = generate_people(N, range_of_age, probability_of_age, sex_ratios_by_range)
    couples = gen_couples(people, marriage_stats)
    people, households = assign_households(people, couples, household_stats)
    household_with_income = assign_income_to_households(households, household_income_stats)
    return household_with_income, people


def preprocess_data(row):
    """
    Preprocess data from a given row of the tree data.
    """
    age_ranges, age_probabilities = extract_population_ranges_and_density(row)
    household_counts = extract_household_stats(row)
    household_incomes = None # TODO (mahi): revisit
    return age_ranges, age_probabilities, household_counts, household_incomes


if __name__ == '__main__':
    N = 50
    population_ranges = [(20, 24), (25, 29)]
    age_densities = [0.2, 0.8]
    sex_ratios_by_range = [(0.5, 0.5), (0.1, 0.9)]
    (generate_people(N, population_ranges, age_densities, sex_ratios_by_range))