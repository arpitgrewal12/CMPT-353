#!/usr/bin/env python
# coding: utf-8
#Arpit Kaur
#301367803
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sys
import scipy.stats as stats

OUTPUT_TEMPLATE = (
    "Initial (invalid) T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann–Whitney U-test p-value: {utest_p:.3g}"
)

def main():
    #counts=pd.read_json('reddit-counts.json.gz',lines=True)
    reddit_counts = sys.argv[1]
    counts = pd.read_json(reddit_counts, lines=True)
    # Conditions
    counts['year'] = pd.DatetimeIndex(counts['date']).year
    counts =(counts[counts['subreddit']=='canada'])
    
    counts = counts[counts['year'] >= 2012]
    counts = counts[counts['year'] <= 2013]
    
    counts['dayofweek'] = counts['date'].dt.weekday
    weekends=counts[(counts['dayofweek']).isin([5,6])]
    weekends = weekends.reset_index(drop=True)

    weekdays=counts[~((counts['dayofweek']).isin([5,6]))]
    weekdays = weekdays.reset_index(drop=True)


    #Tests without transformation
    ttest=stats.ttest_ind(weekdays['comment_count'], weekends['comment_count'] ).pvalue
    levenetest=stats.levene(weekdays['comment_count'], weekends['comment_count']).pvalue
    weekdaysnormal=stats.normaltest(weekdays['comment_count']).pvalue
    weekendsnormal=stats.normaltest(weekends['comment_count']).pvalue

    #Transformations

    #Logarithmic
    weekdaysnormal_log = stats.normaltest(np.log(weekdays['comment_count'])).pvalue
    weekendsnormal_log= stats.normaltest(np.log(weekends['comment_count'])).pvalue
    levenetest_log = stats.levene(np.log(weekdays['comment_count']),np.log(weekends['comment_count'])).pvalue


    #exponential
    #weekdaysnormal_exp = stats.normaltest(np.exp(weekdays['comment_count'])).pvalue
    #weekendsnormal_exp= stats.normaltest(np.exp(weekends['comment_count'])).pvalue
   # levenetest_exp = stats.levene(np.exp(weekdays['comment_count']),np.exp(weekends['comment_count'])).pvalue


    #square root
    weekdaysnormal_sqrt = stats.normaltest(np.sqrt(weekdays['comment_count'])).pvalue
    weekendsnormal_sqrt= stats.normaltest(np.sqrt(weekends['comment_count'])).pvalue
    levenetest_sqrt = stats.levene(np.sqrt(weekdays['comment_count']),np.sqrt(weekends['comment_count'])).pvalue

    #square
    levenetest_square=stats.levene(weekdays['comment_count']**2, weekends['comment_count']**2).pvalue
    weekdaysnormal_square=stats.normaltest(weekdays['comment_count']**2).pvalue
    weekendsnormal_square=stats.normaltest(weekends['comment_count']**2).pvalue

    #Square root transformation is closest to the normal distributions so we pick that one

    #Applying Central limit theorem
    #Dropping year colums which were calculated without isocalendar as they give different results
    weekends=weekends.drop('year', axis=1)
    weekdays=weekdays.drop('year', axis=1)


    weekdays['week'] = weekdays['date'].apply(lambda x: str(x.isocalendar()[1])).apply(pd.Series)
    weekdays['year']= weekdays['date'].apply(lambda x: str(x.isocalendar()[0])).apply(pd.Series)
    weekends['week'] = weekends['date'].apply(lambda x: str(x.isocalendar()[1])).apply(pd.Series)
    weekends['year']= weekends['date'].apply(lambda x: str(x.isocalendar()[0])).apply(pd.Series)
    
    weekdays_mean = weekdays.groupby(['year', 'week']).aggregate('mean').reset_index()
    weekends_mean = weekends.groupby(['year', 'week']).aggregate('mean').reset_index()
    
    weekly_weekday_normality_p= stats.normaltest(weekdays_mean['comment_count']).pvalue
    weekly_weekend_normality_p= stats.normaltest(weekends_mean['comment_count']).pvalue
    weekly_levene_p = stats.levene(weekdays_mean['comment_count'],weekends_mean['comment_count']).pvalue
    weekly_ttest_p = stats.ttest_ind(weekdays_mean['comment_count'], weekends_mean['comment_count']).pvalue

    #Fix3
    utest_p= stats.mannwhitneyu(weekdays['comment_count'],weekends['comment_count'], alternative= 'two-sided').pvalue

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=ttest,
        initial_weekday_normality_p=weekdaysnormal,
        initial_weekend_normality_p=weekendsnormal,
        initial_levene_p=levenetest,
        transformed_weekday_normality_p=weekdaysnormal_sqrt ,
        transformed_weekend_normality_p=weekendsnormal_sqrt,
        transformed_levene_p=levenetest_sqrt,
        
        weekly_weekday_normality_p=weekly_weekday_normality_p,
        weekly_weekend_normality_p=weekly_weekend_normality_p,
        weekly_levene_p=weekly_levene_p,
        weekly_ttest_p=weekly_ttest_p,
        utest_p=utest_p,
    ))

if __name__ == '__main__':
    main()



