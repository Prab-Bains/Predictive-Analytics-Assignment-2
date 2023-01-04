import pickle
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold

f1_scores = []
accuracy_scores = []
recall_scores = []
precision_scores = []
best_model = None
best_f1_score = 0.0


def evaluate_model(X_test, Y_test, model, iteration_num):
    global best_f1_score, best_model
    preds = model.predict(X_test)
    cm = confusion_matrix(Y_test, preds)

    print("Fold", iteration_num, "values:\n")

    precision = precision_score(Y_test, preds, average='binary')
    print("Precision: " + str(precision))
    precision_scores.append(precision)

    precisionStdDev = np.std(precision_scores)
    print("Precision Standard deviation: " + str(precisionStdDev))

    recall = recall_score(Y_test, preds, average='binary')
    print("Recall: " + str(recall))
    recall_scores.append(recall)

    recallStdDev = np.std(recall_scores)
    print("Recall Standard deviation: " + str(recallStdDev))

    accuracy = accuracy_score(Y_test, preds)
    print("Accuracy: " + str(accuracy))
    accuracy_scores.append(accuracy)

    accuracyStdDev = np.std(accuracy_scores)
    print("Accuracy Standard deviation: " + str(accuracyStdDev))

    f1_score = 2 * (precision * recall) / (precision + recall)
    print("F1 Score: " + str(f1_score))
    f1_scores.append(f1_score)

    f1StdDev = np.std(f1_scores)
    print("F1 Standard deviation: " + str(f1StdDev))

    # records the best model in the cross-fold validation (the one with the best F1 score) in a file called bestModel.pkl
    if f1_score > best_f1_score:
        best_f1_score = f1_score
        best_model = model

        # Save model as pickle
        with open(b"bestModel.pkl", 'wb') as model_file:
            pickle.dump(best_model, model_file)


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

# Variables that are significant based off of p-value, chi scores, RFE and FFS (still working on which combination of
# these variables gives the best results) -> Meets requirements of 1 dummy and 1 binned variable
X = X_before_remove_insignificant_ones[
    ['Boarding experience', 'Membership_non-member', 'Staff friendliness', 'Gender_Male', 'Seat Type_economy',
     'Age_(50, 60]', 'Checkin service', 'Seat type', 'Seat Type_standard']]

# Impute the missing values
imputer = KNNImputer(n_neighbors=5)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

y = dataset[['Satisfied']]

kfold = KFold(n_splits=20, shuffle=True)

iteration = 1
for train_index, test_index in kfold.split(X):
    # use index lists to isolate rows for train and test sets.
    x_train = X.loc[X.index.intersection(train_index), :]
    x_test = X.loc[X.index.intersection(test_index), :]
    y_train = y.loc[y.index.intersection(train_index), :]
    y_test = y.loc[y.index.intersection(test_index), :]

    smt = SMOTE()
    X_train_SMOTE, y_train_SMOTE = SMOTE().fit_resample(x_train, y_train)

    clf2 = LogisticRegression(solver='newton-cg', max_iter=1000)
    clf2.fit(X_train_SMOTE, y_train_SMOTE)
    evaluate_model(x_test, y_test, clf2, iteration)

    iteration += 1

avgF1Score = np.mean(f1_scores)
avgF1StdDev = np.std(f1_scores)

avgAccuracy = np.mean(accuracy_scores)
avgAccuracyStdDev = np.std(accuracy_scores)

avgRecall = np.mean(recall_scores)
avgRecallStdDev = np.std(recall_scores)

avgPrecision = np.mean(precision_scores)
avgPrecisionStdDev = np.std(precision_scores)

print('\nAverage precision:', avgPrecision)
print('Average standard deviation for precision:', avgPrecisionStdDev)

print('Average recall:', avgRecall)
print('Average standard deviation for recall:', avgRecallStdDev)

print('Average accuracy:', avgAccuracy)
print('Average standard deviation for accuracy:', avgAccuracyStdDev)

print('Average F1 score:', avgF1Score)
print('Average standard deviation for F1 score:', avgF1StdDev)

print("Best F1 Score:", best_f1_score)