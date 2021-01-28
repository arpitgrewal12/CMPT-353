import sys
from scipy import stats
import numpy as np
import pandas as pd
from datetime import date

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


def func(date):
    return date.isocalendar()[0], date.isocalendar()[1]


def main():
    reddit_counts = sys.argv[1]
    counts = pd.read_json(reddit_counts, lines=True)
   
    counts['month'] = counts['date'].dt.month
    counts['year'] = counts['date'].dt.year
    counts['day'] = counts['date'].dt.day
    
    #Subsetting
    counts = counts[counts['subreddit'] == 'canada']
    counts = counts.reset_index(drop=True) #realign the indexes after dropping
    
    #including only 2012 and 2013 as specified in question
    counts = counts[counts['year'].isin([2012, 2013])]
    counts = counts.drop(['year'], axis=1)
    
    weekends = counts[(counts['date'].dt.weekday).isin([5, 6])]
    weekends = weekends.reset_index(drop=True)
    
    
    weekdays = counts[~((counts['date'].dt.weekday).isin([5, 6]))]
    weekdays = weekdays.reset_index(drop=True)
    
   #initial tests without transformations
    weekday_normality_p = stats.normaltest(weekdays['comment_count']).pvalue
    weekend_normality_p = stats.normaltest(weekends['comment_count']).pvalue
    #levene test will check for equal variance
    levene_test = stats.levene(weekdays['comment_count'], weekends['comment_count']).pvalue
    
    ttest = stats.ttest_ind(weekdays['comment_count'], weekends['comment_count']).pvalue
#   Since p-value<0.05, we do not have enough information to conclude that the number of comments and weekdays and weekends are different.
    
#lets do the transformations

#######logarithmic transformation
    weekdays_log = np.log(weekdays['comment_count'])
    weekdays_log_normality_transform = stats.normaltest(weekdays_log).pvalue

    weekends_log = np.log(weekends['comment_count'])
    weekends_log_normality_transform = stats.normaltest(weekends_log).pvalue

    log_levenes_test = stats.levene(weekends_log, weekdays_log).pvalue


####### square root transformation
    weekdays_sqrt = np.sqrt(weekdays['comment_count'])
    weekdays_sqrt_normality_transform = stats.normaltest(weekdays_sqrt).pvalue

    weekends_sqrt = np.sqrt(weekends['comment_count'])
    weekends_sqrt_normality_transform = stats.normaltest(weekends_sqrt).pvalue

    sqrt_levenes_test = stats.levene(weekends_sqrt, weekdays_sqrt).pvalue

    ##square transformation
    weekdays_sqr = weekdays['comment_count'] ** 2
    weekdays_sqr_normality = stats.normaltest(weekdays_sqr).pvalue

    weekends_sqr = weekends['comment_count'] ** 2
    weekends_sqr_normality = stats.normaltest(weekends_sqr).pvalue

    sqr_levenes = stats.levene(weekends_sqr, weekdays_sqr).pvalue


#   Square root transformation loooks like a better option because  equal variance according to the levene test is met and hence we do not reject null hypothesis and weekends_sqrt comment count is also normal but weekdays_sqrt comment count is still not normal.

    
     #Central limit theorm starts here
    weekdays['week'] = weekdays['date'].apply(func)
    weekends['week'] = weekends['date'].apply(func)
    
    #aggregating on the basis of means to calculate mean comments on weekdays and weekends
    mean_weekday = weekdays.groupby('week').aggregate('mean').reset_index(drop=True)
    mean_weekend = weekends.groupby('week').aggregate('mean').reset_index(drop=True)
    
    weekly_weekday_normality_p = stats.normaltest(mean_weekday['comment_count']).pvalue
    weekly_weekend_normality_p = stats.normaltest(mean_weekend['comment_count']).pvalue
    
    #check for equal variance to satisfy assumption of t test
    weekly_levene_p = stats.levene(mean_weekday['comment_count'], mean_weekend['comment_count']).pvalue
    
    
    weekly_ttest_p = stats.ttest_ind(mean_weekday['comment_count'], mean_weekend['comment_count']).pvalue


    #Mann-whitney Test
    utest_p = stats.mannwhitneyu(weekdays['comment_count'], weekends['comment_count'], alternative='two-sided').pvalue

    # ...

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=ttest,
        initial_weekday_normality_p=weekday_normality_p,
        initial_weekend_normality_p=weekend_normality_p,
        initial_levene_p=levene_test,
        transformed_weekday_normality_p=weekdays_sqrt_normality_transform,
        transformed_weekend_normality_p=weekends_sqrt_normality_transform,
        transformed_levene_p=sqrt_levenes_test,
        weekly_weekday_normality_p=weekly_weekday_normality_p,
        weekly_weekend_normality_p=weekly_weekend_normality_p,
        weekly_levene_p=weekly_levene_p,
        weekly_ttest_p=weekly_ttest_p,
        utest_p=utest_p,
    ))


if __name__ == '__main__':
    main()
