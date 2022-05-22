from libraries import *

def ANNModel():    
    # ANN Model
    classifier = Sequential()
    classifier.add(Dense(5, activation="relu", input_dim = 16))
    classifier.add(Dense(5, activation="relu"))
    classifier.add(Dense(1, activation="sigmoid"))
    classifier.compile(optimizer = "adam" , loss="binary_crossentropy", metrics=["accuracy"])
    return classifier

classifier = KerasClassifier(ANNModel, epochs=100, batch_size=500, verbose=0)

# Preprocessing

random_state_pos = 0

YName = "Transported"

# Read the data
data = pd.read_csv("train.csv", index_col = 0)
#print(data)

# Separate taget from predictions

X = data.drop([YName], axis=1)
y = data[YName]

# Divide data into training and validation subsets
XTrainPure, XValidPure, YTrainPure, YValidPure = train_test_split(X, y, train_size=0.8, random_state=random_state_pos)

#Select categorical columns with relatively low cardinality(convenient but arbitary)
CategoricalColumns = [CName for CName in XTrainPure.select_dtypes(include = ["object"]).columns if XTrainPure[CName].nunique() < 10]
#print(CategoricalColumns)

#Select numerical columns
NumericalColumns = list(XTrainPure.select_dtypes(exclude = ["object"]).columns)
#print(NumericalColumns)

# Keep selected columns only
MyColumns = CategoricalColumns + NumericalColumns
#print(MyColumns.__len__())
XTrain = XTrainPure[MyColumns]
XValid = XValidPure[MyColumns]

NumericalTransformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant"))
])

CategoricalTransformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

Preprocessor = ColumnTransformer(transformers=[
    ("num", NumericalTransformer, NumericalColumns),
    ("cat", CategoricalTransformer, CategoricalColumns),
])

# Bundle preprocessing and modeling code in a pipeline
MyPipeline = Pipeline(steps=[("preprocessor", Preprocessor), ("model", classifier)])

# Preprocessing of traning data and fit model
MyPipeline.fit(XTrain, YTrainPure)

# Preprocessing of validation data, get predictions
Predictions = MyPipeline.predict(XValid)
Predictions = [1 if i > 0.5 else 0 for i in Predictions]

# Evaluate the model
Score = mean_absolute_error(YValidPure, Predictions)

print("MAE: ", Score)

File = "model.save"
pc.dump(MyPipeline, open(File, "wb"))


def PreproceedData(Table, YName, YPos = True):

    if YPos:
        X = Table.drop([YName], axis=1)
        y = Table[YName]
    else:
        X = Table

    #Select categorical columns with relatively low cardinality(convenient but arbitary)
    CategoricalColumns = [CName for CName in X.select_dtypes(include = ["object"]).columns if X[CName].nunique() < 10]

    #Select numerical column
    NumericalColumns = list(X.select_dtypes(exclude = ["object"]).columns)

    # Keep selected columns only
    MyColumns = CategoricalColumns + NumericalColumns
    X = X[MyColumns]

    NumericalTransformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant"))
    ])

    CategoricalTransformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    Preprocessor = ColumnTransformer(transformers=[
        ("num", NumericalTransformer, NumericalColumns),
        ("cat", CategoricalTransformer, CategoricalColumns),
    ])

    X = Preprocessor.fit_transform(X)
    if YPos:
        return X, y
    else:
        return X

X, y = PreproceedData(data, "Transported")

classifier.fit(X, y)

TestPure= pd.read_csv("test.csv")
Test = PreproceedData(TestPure, "Transported", YPos = False)
Predictions = classifier.predict(Test)

print(Predictions)
Predictions = pd.Series(Predictions[:, 0], index=TestPure.PassengerId, name=YName)
Predictions.to_csv("submission.csv")