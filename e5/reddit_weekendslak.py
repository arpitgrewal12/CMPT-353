import sys
import pandas as pd
from datetime import date
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

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
    counts = pd.read_json(sys.argv[1], lines=True)
    # Checking Conditions
    yearCheck = (pd.DatetimeIndex(counts['date']).year == 2013) | (pd.DatetimeIndex(counts['date']).year == 2012)
    counts = counts[yearCheck]
    counts = counts[counts['subreddit'] == 'canada']

    # Data in Weekends
    weekend = counts[counts['date'].dt.dayofweek > 4]

    # Data in Weekdays
    weekday = counts[counts['date'].dt.dayofweek < 5]

    # Using CLT
    weekday["year-week"] = weekday['date'].dt.date
    weekday["year-week"] = weekday['year-week'].apply(lambda x: str(x.isocalendar()[0]) + '-' + str(x.isocalendar()[1]).zfill(2))
    weekday_mean = weekday.groupby(['year-week']).aggregate('mean')['comment_count']

    weekend["year-week"]=weekend['date'].dt.date
    weekend["year-week"]=weekend['year-week'].apply(lambda x: str(x.isocalendar()[0]) + '-' + str(x.isocalendar()[1]).zfill(2))
    weekend_mean = weekend.groupby(['year-week']).aggregate('mean')['comment_count']
    print(stats.ttest_ind(np.sqrt (weekday['comment_count']), np.sqrt (weekend['comment_count'])).pvalue < stats.ttest_ind(weekday_mean,weekend_mean).pvalue)

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p = stats.ttest_ind(weekday['comment_count'],weekend['comment_count']).pvalue ,
        initial_weekday_normality_p= stats.normaltest(weekday['comment_count']).pvalue ,
        initial_weekend_normality_p= stats.normaltest(weekend['comment_count']).pvalue ,
        initial_levene_p= stats.levene(weekday['comment_count'],weekend['comment_count']).pvalue ,
        transformed_weekday_normality_p= stats.normaltest(np.sqrt (weekday['comment_count'])).pvalue ,
        transformed_weekend_normality_p= stats.normaltest(np.sqrt (weekend['comment_count'])).pvalue ,
        transformed_levene_p= stats.levene(np.sqrt (weekday['comment_count']), np.sqrt (weekend['comment_count'])).pvalue ,
        weekly_weekday_normality_p=stats.normaltest(weekday_mean).pvalue,
        weekly_weekend_normality_p= stats.normaltest(weekend_mean).pvalue,
        weekly_levene_p= stats.levene(weekday_mean,weekend_mean).pvalue ,
        weekly_ttest_p= stats.ttest_ind(weekday_mean,weekend_mean).pvalue ,
        utest_p= stats.mannwhitneyu(weekday['comment_count'],weekend['comment_count'], alternative= 'two-sided').pvalue,
    ))


if __name__ == '__main__':
    main()