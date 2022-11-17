import sys

import brute_force_analysis
import generate_output
import pandas as pd
import resp_pred_analysis
import sqlalchemy
from pandas import DataFrame
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# import numpy as np
# from sklearn.model_selection import TimeSeriesSplit, train_test_split


def print_heading(title):
    """Prints Heading"""

    print("\n" + "*" * 80)
    print(title)
    print("*" * 80 + "\n")
    return


def get_data() -> DataFrame:
    """Get data from mariadb"""

    db_user = "admin"
    db_pass = "password"  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )
    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """
        SELECT * FROM pitchers_stats_calc
    """

    df = pd.read_sql_query(query, sql_engine)
    df = df.fillna(0)

    return df


def model_training(X_train, X_test, y_train, y_test):
    """Calls model_training function for training models and return the stats"""

    print_heading("Model Training")
    # Create a dictionary with model names and classifiers
    models = {
        "decision_tree": {"model": DecisionTreeClassifier()},
        "knn": {"model": KNeighborsClassifier()},
        "svm": {"model": LinearSVC()},
        "random_forest": {"model": RandomForestClassifier()},
        "ada_boost": {"model": AdaBoostClassifier()},
    }

    # Look through all models and store all details in a list
    scores = []

    for model_name, model_clf in models.items():
        # Fit Model on data
        model = model_clf["model"]
        model.fit(X_train, y_train)
        # Predict the label using model
        prediction = model.predict(X_test)
        # Calculate accuracy of model
        accuracy = accuracy_score(y_test, prediction)
        # Calculate precision of model
        precision = precision_score(prediction, y_test, average="micro")
        # Calculate recall of model
        recall = recall_score(prediction, y_test, average="micro")
        stats = {
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }
        scores.append(stats)

    # Use the list to create dataframe and return the results
    model_scores = pd.DataFrame(scores)
    print(model_scores)
    return


def model_pipeline(X_train, X_test, y_train, y_test):
    """Trains Classifiers using Pipeline"""

    print_heading("Models via Pipeline Predictions")

    # Create a dictionary with model names and pipelines
    pipelines = {
        "decision_tree": {
            "pipeline": Pipeline(
                [
                    ("StandardScaler", StandardScaler()),
                    ("DecisionTree", DecisionTreeClassifier(random_state=1234)),
                ],
            )
        },
        "knn": {
            "pipeline": Pipeline(
                [
                    ("StandardScaler", StandardScaler()),
                    ("KNeighbors", KNeighborsClassifier()),
                ],
            )
        },
        "random_forest": {
            "pipeline": Pipeline(
                [
                    ("StandardScaler", StandardScaler()),
                    ("RandomForest", RandomForestClassifier(random_state=1234)),
                ],
            )
        },
    }

    # Loop through all models and use pipelines to build models
    for model_name, model_pipe in pipelines.items():
        print_heading(f"Pipeline for {model_name}")
        pipeline = model_pipe["pipeline"]
        # Fit on data using the pipeline
        pipeline.fit(X_train, y_train)
        # Calculate probability and prediction for model using pipeline
        probability = pipeline.predict_proba(X_test)
        prediction = pipeline.predict(X_test)
        score = accuracy_score(y_test, prediction)
        print(f"Probability: {probability}")
        print(f"Predictions: {prediction}")
        print(f"Score: {score}")

    return


def main():
    pd.options.display.max_columns = None
    df = get_data()
    df = df.sort_values(by="game_date")

    response = df["Home_Team_Wins"]
    predictors = df.drop(["Home_Team_Wins", "game_date"], axis=1)
    # predictors = df.drop("Home_Team_Wins", axis=1)
    resp_pred_analysis.do_analysis(df, predictors, response)
    brute_force_analysis.do_analysis(df, predictors.columns, response.name)
    generate_output.just_do_it()

    predictors["year"] = df["game_date"].dt.year
    # new = predictors[predictors['year'] == 2012]
    # 2927 for 2007
    # 2922 for 2008
    # 2991 for 2009
    # 2914 for 2010
    # 2952 for 2011
    # 1178 for 2012
    # Total = 15884
    # print(((1178+2952)/15884)*100)
    # print(df.shape)
    # tscv = TimeSeriesSplit()
    # for train_index, test_index in tscv.split(predictors):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = predictors[train_index], predictors[test_index]
    #     y_train, y_test = response[train_index], response[test_index]

    # Using numpy, split
    # train_set, test_set = np.split(df, [int(.80 * len(df))])

    X_train, X_test, y_train, y_test = train_test_split(
        predictors, response, test_size=0.2, shuffle=False
    )

    model_training(X_train, X_test, y_train, y_test)
    model_pipeline(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    sys.exit(main())
