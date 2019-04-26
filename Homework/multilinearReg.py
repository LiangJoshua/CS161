import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from pandas.plotting import scatter_matrix
from pandas import set_option
from pandas import read_csv

#Use the baseball salary dataset, load the dataset using Pandas read_csv() and treat the log
#(natural log) of the salary column as the output and the rest of the columns as input
#features (Make sure to look at the dataframe after loading and eliminate any extra column
#that may have been generated).
filename = 'Baseball_salary.csv'
data = read_csv(filename)
#Exclude all the categorical columns from the features (columns with non-numeric data, such
#as League, …)
newData = data.drop(['League','Division','NewLeague'], axis=1)
newData = newData.drop(newData.columns[0], axis=1)
newData.dropna(inplace=True)


# save features as pandas dataframe for stepwise feature selection
X1 = newData.drop(newData.columns[16], axis = 1)
Y1 = newData.drop(newData.columns[0:16], axis = 1)

# separate features and response into two different arrays
array = newData.values
X = array[:,0:16]
Y = array[:,16]


# stepwise forward-backward selection
# need to change the input types as X in this function needs to be a pandas
# dataframe

def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.01,
                       threshold_out=0.05,
                       verbose=True):
    """ Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X [included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


result = stepwise_selection(X1, Y1)

print('resulting features:')
print(result)



# Determiniation of dominant features , Method one Recursive Model Elimination,
# very similar idea to foreward selection but done recurssively. This method is gready
# which means it tries one feature at the time
NUM_FEATURES = 4
# this is kind of arbitrary but the idea should come by observing the scatter plots and correlation.
model = LinearRegression()
rfe = RFE(model, NUM_FEATURES)
fit = rfe.fit(X, Y)
print("Num Features:", fit.n_features_)
print("Selected Features:", fit.support_)
print("Feature Ranking:", fit.ranking_)
# calculate the score for the selected features
score = rfe.score(X,Y)
print("Model Score with selected features is: ", score)

"""
Results:

Based on the step-forward/backward method, the selected features are CRBI, Hits, PutOuts, AtBat, and Walks.

Based on RFE method, the selected features with 5 features are Hits, HomeRun, Walks, Years, CAtBat.

Based on both results, the final model would have features of Hits and Walks. 

"""
