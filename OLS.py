from libraries import *
from keras.wrappers.scikit_learn import KerasClassifier

Table = pd.read_csv("train.csv")

def PreproceedData(Table, YName):

    X = Table.drop([YName], axis=1)
    y = Table[YName]

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
    return X, y
    

def ANNModel():    
    # ANN Model
    classifier = Sequential()
    classifier.add(Dense(5, activation="relu", input_dim = 16))
    classifier.add(Dense(5, activation="relu"))
    classifier.add(Dense(1, activation="sigmoid"))
    classifier.compile(optimizer = "adam" , loss="binary_crossentropy", metrics=["accuracy"])
    return classifier

classifier = KerasClassifier(ANNModel, epochs=100, batch_size=500, verbose=0)

def predictATKeywords(table):
    global classifier
    
    X, y = PreproceedData(table, "Transported")
    
    statModel = sm.OLS(y, X)
    results = statModel.fit()
    results_summary = results.summary()
    print(results_summary)
    results_as_html = results_summary.tables[1].as_html()
    stats = pd.read_html(results_as_html, header=0, index_col=0)[0]
    priorValues = stats.loc[:, "P>|t|"].unique()
    priorValues = pd.Series(priorValues, name = "Prior Values")
    print(priorValues)
    return priorValues        