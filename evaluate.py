import pandas as pd
import numpy as np
from query import EcoinventRecommendation

if __name__ == '__main__':

    df = pd.read_excel('ecoinvent-mapping.xlsx')
    df = df[~pd.isnull(df['Activity name'])]
    activity_col_names = ['Alternate activity {}'.format(i+1) for i in range(4)] + ['Activity name']

    database_recommender = EcoinventRecommendation()

    num_correct = 0
    num_correct_websearch = 0
    num_correct_websearch_llm = 0
    for idx, row in df.iterrows():
        database_recommender.set_query(component_name=row['Component name'],
                                       producer=row['Supplier'],
                                       material=row['Material'])
        search_query, search_results = database_recommender.get_search_results()

        response, (matches, scores) = database_recommender.get_matches()
        #TODO: read labels