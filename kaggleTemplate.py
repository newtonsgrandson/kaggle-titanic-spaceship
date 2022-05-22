from libraries import *

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
print(MyColumns.__len__())
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
# Random Forest Regressor
Model = RandomForestRegressor(n_estimators=1000, random_state = random_state_pos)

# Bundle preprocessing and modeling code in a pipeline
MyPipeline = Pipeline(steps=[("preprocessor", Preprocessor), ("model", Model)])

# Preprocessing of traning data and fit model
MyPipeline.fit(XTrain, YTrainPure)

# Preprocessing of validation data, get predictions
Predictions = MyPipeline.predict(XValid)
Predictions = [1 if i > 0.5 else 0 for i in Predictions]

# Evaluate the model
Score = mean_absolute_error(YValidPure, Predictions)

print("MAE: ", Score)

ScoresCrossVal = -1 * cross_val_score(MyPipeline, X, y, cv = 5, scoring = "neg_mean_absolute_error")

print("MAE scores:\n", ScoresCrossVal)
print("Average MAE score (across experiments):", ScoresCrossVal.mean())

# Total Model
Scaler = StandardScaler()
model = RandomForestRegressor(n_estimators=100, random_state=random_state_pos)
Test = pd.read_csv("test.csv", index_col = 0)
KagglePipeline = Pipeline(steps=[("preprocess", Preprocessor), ("scaler", Scaler), ("model", model)])
KagglePipeline.fit(X, y)
predictions = pd.Series(KagglePipeline.predict(Test), index=Test.index, name=YName)
predictions.to_csv("submission.csv")

# XGBoost
Scaler = StandardScaler()
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4, random_state=random_state_pos)
Test = pd.read_csv("test.csv", index_col = 0)
KagglePipeline = Pipeline(steps=[("preprocess", Preprocessor), ("scaler", Scaler), ("model", model)])
KagglePipeline.fit(X, y)
predictions = pd.Series(KagglePipeline.predict(Test), index=Test.index, name=YName)
predictions.to_csv("submission.csv")