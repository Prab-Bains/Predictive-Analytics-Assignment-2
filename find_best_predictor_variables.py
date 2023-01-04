# Finds the best predictor variables to use, using 3 different methods: RFE, FFS, Chi Scores.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_regression
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler


def showYPlots(y_train, y_test, title):
    print("\n ***" + title)
    plt.subplots(1, 2)

    plt.subplot(1, 2, 1)
    plt.hist(y_train)
    plt.title("Train Y: " + title)

    plt.subplot(1, 2, 2)
    plt.hist(y_test)
    plt.title("Test Y: " + title)
    plt.show()


def evaluate_model(X_test, y_test, y_train, model, title):
    showYPlots(y_train, y_test, title)

    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    print(cm)
    precision = precision_score(y_test, preds, average='binary')
    print("Precision: " + str(precision))

    recall = recall_score(y_test, preds, average='binary')
    print("Recall:    " + str(recall))

    accuracy = accuracy_score(y_test, preds)
    print("Accuracy:    " + str(accuracy))


PATH = "/Users/prabhjeetbains/Desktop/BCIT/Term 3/Predictive Modelling/Data sets/amtrack_survey.csv"
dataset = pd.read_csv(PATH, encoding="ISO-8859-1", sep=',')

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

tempDf1 = dataset[['Trip Type', 'Gender', 'Seat Type', 'Membership']]  # need to get dummy variables for
tempDf2 = dataset[['Boarding experience', 'Delayed arrival', 'Booking experience',
                   'Delayed departure', 'Cleanliness', 'Trip Distance', 'Wifi', 'Staff friendliness', 'Age',
                   'Quality Food', 'Seat comfort', 'Checkin service', 'Luggage service', 'Online experience',
                   'Departure Convenience', 'Seat type']]

tempDf3 = dataset[['Boarding experience', 'Delayed arrival', 'Booking experience',
                   'Delayed departure', 'Cleanliness', 'Trip Distance', 'Wifi', 'Staff friendliness',
                   'Quality Food', 'Seat comfort', 'Checkin service', 'Luggage service', 'Online experience',
                   'Departure Convenience',
                   'Seat type']]  # Remove 'Age' column from tempDf2 for further binning columns to join.

dummyDf = pd.get_dummies(tempDf1, columns=['Trip Type', 'Gender', 'Seat Type', 'Membership'])

tempDf2['Age'] = pd.cut(x=tempDf2['Age'],
                        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

tempDf = tempDf2[['Age']]
dummyDf2 = pd.get_dummies(tempDf, columns=['Age'])

X_before_remove_insignificant_ones = pd.concat(([tempDf3, dummyDf, dummyDf2]),
                                               axis=1)  # dataset with dummy variables

imputer = KNNImputer(n_neighbors=5)
X = pd.DataFrame(imputer.fit_transform(X_before_remove_insignificant_ones),
                 columns=X_before_remove_insignificant_ones.columns)

y = dataset[['Satisfied']]


def RFE_scores():
    # Create the object of the model
    model = LogisticRegression()

    # Specify the number of  features to select
    rfe = RFE(model, n_features_to_select=11)

    # fit the model
    rfe = rfe.fit(X, y)

    # Please uncomment the following lines to see the result
    print('\n\nFEATURES SELECTED\n\n')
    print(rfe.support_)

    # Show top features.
    for i in range(0, len(X.keys())):
        if (rfe.support_[i]):
            print(X.keys()[i])


def chi_scores():
    test = SelectKBest(score_func=chi2, k=15)

    # Use scaled data to fit KBest
    XScaled = MinMaxScaler().fit_transform(X)
    chiScores = test.fit(XScaled, y)  # Summarize scores
    np.set_printoptions(precision=3)

    # Search here for insignificant features.
    print("\nPredictor Chi-Square Scores: " + str(chiScores.scores_))

    # Create a sorted list of the top features.
    dfFeatures = pd.DataFrame()
    for i in range(0, len(chiScores.scores_)):
        headers = list(X.keys())
        featureObject = {"feature": headers[i], "chi-square score": chiScores.scores_[i]}
        dfFeatures = dfFeatures.append(featureObject, ignore_index=True)

    print("\nTop Features")
    dfFeatures = dfFeatures.sort_values(by=['chi-square score'], ascending=False)
    print(dfFeatures.head(11))


#  f_regression is a scoring function to be used in a feature selection procedure
#  f_regression will compute the correlation between each regressor and the target

def ffs_scores():
    ffs = f_regression(X, y)

    variable = []
    for i in range(0, len(X.columns) - 1):
        if ffs[0][i] >= 700:
            variable.append(X.columns[i])

    print(variable)


ffs_scores()
RFE_scores()
chi_scores()
