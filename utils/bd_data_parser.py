"""
bd_data_parser.py helps convert the data from Bangladesh into the format 
required by other parts of the code
"""

import os
import pandas as pd


BD_AGE_COLUMNS = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', 
                  '30-49', '50-59', '60-64', '65+']

BD_AGE_RANGES = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29),
                 (30, 49), (50, 59), (60, 64), (65, 69)]

HOUSEHOLD_COLUMNS = ['household_1person', 'household_2person', 
                     'household_3person', 'household_4person', 
                     'household_5person', 'household_6person', 
                     'household_7person', 'household_8+person']
TOTAL_HOUSEHOLDS_COLUMN = 'num_general_household'

BD_MARRIAGE_DATA_FILE = '../data_files/marriage_data_bd.csv'
BD_GENDER_RATIO_FILE = '../data_files/bangladesh_population_pyramid_adjusted.csv'

def extract_population_ranges_and_density(row):
    """
    Each row contains the information about the age in the range style that is
    expected by population generator.
    """
    age_columns = row[BD_AGE_COLUMNS] / 100. # Convert from %
    age_densities = list(age_columns)
    return BD_AGE_RANGES, age_densities


def extract_household_stats(row):
    """
    Extract the number of households at each household size from the row.
    """
    household_nums = row[HOUSEHOLD_COLUMNS] / 100. # convert from %
    N = row[TOTAL_HOUSEHOLDS_COLUMN]
    return list((N * household_nums).astype('int32'))


def get_marriage_statistics():
    """
    Load the age-vs.-marriage statistics for bangladesh.
    """
    data_file = os.path.join(os.path.dirname(__file__), BD_MARRIAGE_DATA_FILE)
    df = pd.read_csv(data_file).set_index('Age Groups').iloc[1:]
    male_married = df['Male_CM'].repeat(5)
    female_married = df['Female_CM'].repeat(5)
    stats = {}
    for i, (x, y) in enumerate(zip(male_married, female_married)):
        stats[('m', i)] = x
        stats[('f', i)] = y
    return stats


def get_age_vs_sex_ratios():
    """
    Load the age-vs-gender ratio statistics.
    """
    data_file = os.path.join(os.path.dirname(__file__), BD_GENDER_RATIO_FILE)
    df = pd.read_csv(data_file).set_index('Age')
    # Keep only the total number of ranges columns
    df = df.divide(df.sum(axis='columns'), axis='index')
    return df


if __name__ == '__main__':
    print (get_age_vs_sex_ratioa())