import pandas as pd
from query import DocumentRecommendation
import csv
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='CircularTree_BOM.xlsx')
    parser.add_argument('--sheet_name', type=str, default='Template')
    parser.add_argument('--start_row', type=int, default=4)
    parser.add_argument('--component_name_col', type=int, default=-1)
    parser.add_argument('--material_col', type=int, default=-1)
    parser.add_argument('--producer_col', type=int, default=-1)
    parser.add_argument('--product_description', type=str, default='unknown')

    args = parser.parse_args()
    start_row = args.start_row

    xl_file = pd.ExcelFile(args.filename)
    sheet = xl_file.parse(args.sheet_name)
    total_rows = len(sheet)
    num_rows = total_rows - start_row

    materials=[""]*num_rows
    if args.material_col > 0:
        materials = sheet.iloc[start_row:total_rows,args.material_col]
    components=[""]*num_rows
    if args.component_name_col > 0:
        components = sheet.iloc[start_row:total_rows,args.component_name_col]
    producers=[""]*num_rows
    if args.producer_col > 0:
        producers = sheet.iloc[start_row:total_rows,args.producer_col]

    recommender = DocumentRecommendation(model_name='llama3.1:8b')
    enable_web_search=False

    res = {'Component': [],
            'Producer': [],
            'Response': [],
            'Match1': [],
            'Match2': [],
            'Match3': [],
            'Match4': [],
            'Match5': [],
            'Score1': [],
            'Score2': [],
            'Score3': [],
            'Score4': [],
            'Score5': [],}
    for c, p, m in zip(components, producers, materials):
        suggestions, scores, response = recommender.get_matches(args.product_description,
                                                                c, p, m,
                                                                component_list=components,
                                                                producer_list=producers,
                                                                material_list=materials,
                                                                web_search=enable_web_search)

        res['Component'].append(c)
        res['Producer'].append(p)
        res['Response'].append(response)
        for i in range(len(suggestions)):
            res['Match{}'.format(i+1)].append(suggestions[i])
            res['Score{}'.format(i+1)].append(scores[i])

    df = pd.DataFrame(res)
    df.to_excel('output{}.xlsx'.format('_web_search' if enable_web_search else ''))
