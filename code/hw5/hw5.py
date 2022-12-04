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


def model_pipeline(X_train, X_test, y_train, y_test):
    """Trains Classifiers using Pipeline"""

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
        "svm": {
            "pipeline": Pipeline(
                [
                    ("StandardScaler", StandardScaler()),
                    ("SVM", LinearSVC(dual=False, random_state=1234)),
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
        "ada_boost": {
            "pipeline": Pipeline(
                [
                    ("StandardScaler", StandardScaler()),
                    ("AdaBoost", AdaBoostClassifier(random_state=1234)),
                ],
            )
        },
    }

    # Look through all models and store all details in a list
    scores = []

    # Loop through all models and use pipelines to build models
    for model_name, model_pipe in pipelines.items():
        pipeline = model_pipe["pipeline"]
        # Fit on data using the pipeline
        pipeline.fit(X_train, y_train)
        # probability = pipeline.predict_proba(X_test)
        prediction = pipeline.predict(X_test)
        score = accuracy_score(y_test, prediction)
        precision = precision_score(prediction, y_test)
        recall = recall_score(prediction, y_test)
        stats = {
            "model_name": model_name,
            "accuracy": score,
            "precision": precision,
            "recall": recall,
            "diff_precision_recall": abs(precision - recall),
        }
        scores.append(stats)

    # Use the list to create dataframe and return the results
    model_scores = pd.DataFrame(scores)

    return model_scores


def main():
    pd.options.display.max_columns = None

    # Get data and sort by date
    df = get_data()
    df = df.sort_values(by="game_date")
    df = df.reset_index(drop=True)

    # Take response and predictors
    response = df["Home_Team_Wins"]
    predictors = df.drop(["Home_Team_Wins", "game_date"], axis=1)

    # Perform Response and Predictors Analysis
    resp_pred_analysis.do_analysis(df, predictors, response)
    # Perform Brute Force Analysis
    brute_force_analysis.do_analysis(df, predictors.columns, response.name)

    # Train and split data
    X_train, X_test, y_train, y_test = train_test_split(
        predictors, response, test_size=0.2, shuffle=False
    )

    # Train Models
    models_stats = model_pipeline(X_train, X_test, y_train, y_test)
    brute_force_analysis.generate_html(models_stats, "models_stats")

    # Generate final output
    generate_output.just_do_it()


if __name__ == "__main__":
    sys.exit(main())
