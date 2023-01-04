import pickle
import pandas as pd
from sklearn.impute import KNNImputer

PATH = "/Users/prabhjeetbains/Desktop/BCIT/Term 3/Predictive Modelling/amtrack_data/amtrack_survey_mystery copy.csv"
dataset = pd.read_csv(PATH, encoding="ISO-8859-1", sep=',')

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

# If columns to create dummy variables don't exist in the dataframe, the code below adds it
requiredDatasetColumns = ['Boarding experience', 'Delayed arrival', 'Booking experience',
                          'Delayed departure', 'Cleanliness', 'Trip Distance', 'Wifi', 'Staff friendliness', 'Age',
                          'Quality Food', 'Seat comfort', 'Checkin service', 'Luggage service', 'Online experience',
                          'Departure Convenience', 'Seat type', 'Trip Type', 'Gender', 'Seat Type', 'Membership']

dataset_columns = list(dataset.keys())

for i in range(0, len(requiredDatasetColumns)):
    columnFound = False
    for j in range(0, len(dataset_columns)):
        if dataset_columns[j] == requiredDatasetColumns[i]:
            columnFound = True
    if not columnFound:
        dataset[requiredDatasetColumns[i]] = 0  # Add non-existing dummy column and initialize all values to zero.

# *** Any data prep, imputing, binning, dummy variable creation etc. goes here.
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

# If columns don't exist in the dataframe with the dummy variables, the code below adds it
requiredColumns = ['Boarding experience', 'Membership_non-member', 'Staff friendliness', 'Gender_Male',
                   'Seat Type_economy',
                   'Age_(50, 60]', 'Checkin service', 'Seat type', 'Seat Type_standard']

columns = list(X_before_remove_insignificant_ones.keys())

for i in range(0, len(requiredColumns)):
    columnFound = False
    for j in range(0, len(columns)):
        if columns[j] == requiredColumns[i]:
            columnFound = True
    if not columnFound:
        X_before_remove_insignificant_ones[
            requiredColumns[i]] = 0  # Add non-existing dummy column and initialize all values to zero.

# Variables that are significant based off of p-value, chi scores, RFE and FFS (still working on which combination of
# these variables gives the best results) -> Meets requirements of 1 dummy and 1 binned variable
X = X_before_remove_insignificant_ones[
    ['Boarding experience', 'Membership_non-member', 'Staff friendliness', 'Gender_Male', 'Seat Type_economy',
     'Age_(50, 60]', 'Checkin service', 'Seat type', 'Seat Type_standard']]

# Impute the missing values
imputer = KNNImputer(n_neighbors=5)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Load pre-trained model.
file = open("bestModel.pkl", 'rb')
loadedModel = pickle.load(file)

predictions = loadedModel.predict(X)

# Store predictions in a dataframe
dfPredictions = pd.DataFrame()
listPredictions = []
for i in range(0, len(predictions)):
    prediction = predictions[i]
    listPredictions.append(prediction)
dfPredictions['Predictions'] = listPredictions

# writes the predictions to a CSV file
dfPredictions.to_csv('Predictions.csv', index=False)
