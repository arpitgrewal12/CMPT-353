#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import sys
import matplotlib as plt
import json
import scipy.stats as stats


OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value:  {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value:  {more_searches_p:.3g} \n'
    '"Did more/less instructors use the search feature?" p-value:  {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value:  {more_instr_searches_p:.3g}'
)


def main():
    searchdata_file = sys.argv[1]

    #df=pd.read_json('searches.json', orient='records', keep_default_dates=True, precise_float=False, date_unit=None, encoding=None, lines=True, chunksize=None, compression='infer')
    df=pd.read_json(searchdata_file, orient='records', keep_default_dates=True, precise_float=False, date_unit=None, encoding=None, lines=True, chunksize=None, compression='infer')


    #data_list = np.array(df)


    odd_uid=df[df['uid']%2!= 0] #Odd Uids
    even_uid=df[df['uid']%2==0] #Even Uids

    #odd_uid
    #even_uid

    odd_searched_atleast_once=odd_uid.loc[odd_uid['search_count']>=1]
    rows_odd_atleat_once = odd_searched_atleast_once.shape[0]
    #rows_odd_atleat_once

    odd_never_searched=odd_uid[odd_uid['search_count']<1]
    rows_odd_never = odd_never_searched.shape[0]
    #rows_odd_never
    even_searched_atleast_once=even_uid[even_uid['search_count']>=1]
    rows_even_atleat_once = even_searched_atleast_once.shape[0]
    even_never_searched=even_uid[even_uid['search_count']<1]
    rows_even_never = even_never_searched.shape[0]

    #rows_even_atleat_once
    #rows_even_never

    data_crosstab = [[rows_odd_atleat_once , rows_odd_never],
                                 [rows_even_atleat_once , rows_even_never]]
    contingency_table=pd.DataFrame(data_crosstab,index=['odd uid ' ,'even uid'],
                                  columns=['Searched atleast once','Searched never'])

    #contingency_table

    chi, p, dof, uid= stats.chi2_contingency(contingency_table)

    odd_is_instructor=odd_uid[odd_uid['is_instructor']==True] #Instructor Uids
    even_is_instructor=even_uid[even_uid['is_instructor']==True]

    rows_odd_ins_nonzero_search=odd_is_instructor[odd_is_instructor['search_count']>=1]
    rows_odd_ins_zero_search=odd_is_instructor[odd_is_instructor['search_count']<1]
    rows_even_ins_nonzero_search=even_is_instructor[even_is_instructor['search_count']>=1]
    rows_even_ins_zero_search=even_is_instructor[even_is_instructor['search_count']<1]

    num_rows_odd_ins_nonzero_search=rows_odd_ins_nonzero_search.shape[0]
    num_rows_odd_ins_zero_search=rows_odd_ins_zero_search.shape[0]
    num_rows_even_ins_nonzero_search=rows_even_ins_nonzero_search.shape[0]
    num_rows_even_ins_zero_search=rows_even_ins_zero_search.shape[0]

    data_crosstab_ins= [num_rows_odd_ins_nonzero_search,num_rows_odd_ins_zero_search],[num_rows_even_ins_nonzero_search,num_rows_even_ins_zero_search]
    contingency_table_ins=pd.DataFrame(data_crosstab_ins,index=['odd_Is_instructor ','even_Is_instructor'],
                                       columns=['Searched atleast once','Searched never'])
    #contingency_table_ins


    chi_ins, p_ins, dof_ins, ins = stats.chi2_contingency(contingency_table_ins)


    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p=p,
        more_searches_p=stats.mannwhitneyu(odd_uid['search_count'],even_uid['search_count']).pvalue,
        more_instr_p=p_ins,
        more_instr_searches_p=stats.mannwhitneyu(odd_is_instructor['search_count'],even_is_instructor['search_count']).pvalue,
    ))


if __name__ == '__main__':
    main()
