import sys

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def print_heading(title):
    """Prints Heading"""

    print("\n" + "*" * 80)
    print(title)
    print("*" * 80 + "\n")
    return


def read_data():
    """Load dataset into dataframe and returns it"""

    # URL for downloading dataset
    url = "https://teaching.mrsharky.com/data/iris.data"
    # List for Column names
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    # Load data into dataframe
    df = pd.read_csv(url, names=cols, header=None)
    return df


def stats(df):
    """Generates summary statistics using numpy"""

    print_heading("Statistics using numpy")
    # Counter for column count
    counter = 0
    # Loop through each column of dataframe and respective values
    for (columnName, columnData) in df.iteritems():
        # Increase counter by 1 in each iteration for column number
        counter += 1
        # Calculate Mean Value
        mean = round(np.mean(columnData.values), 2)
        # Calculate Min Value
        min = round(np.min(columnData.values), 2)
        # Calculate Max Value
        max = round(np.max(columnData.values), 2)
        # Calculate first quartile
        quart_25 = round(np.quantile(columnData.values, 0.25), 2)
        # Calculate second quartile
        quart_50 = round(np.quantile(columnData.values, 0.50), 2)
        # Calculate third quartile
        quart_75 = round(np.quantile(columnData.values, 0.75), 2)

        # Print the statistics for the column
        print(f"{counter}. Column Name : {columnName}")
        print(f"    Mean: {mean}")
        print(f"    Min: {min}")
        print(f"    Max: {max}")
        print(f"    Quartiles:: 25%: {quart_25}, 50%: {quart_50}, 75%: {quart_75}")
    return


def visuals(df):
    """Plots different classes against each other"""

    # Plot scatter plot and write to html file
    scatter_plot = px.scatter(
        df, x="sepal_width", y="sepal_length", color="class", symbol="class"
    )
    scatter_plot.write_html(file="scatter_plot.html", include_plotlyjs="cdn")

    # Plot violin plot and write to html file
    violin_plot = px.violin(
        df,
        y="sepal_length",
        x="class",
        color="class",
        box=True,
        points="all",
        hover_data=df.columns,
    )
    violin_plot.write_html(file="violin_plot.html", include_plotlyjs="cdn")

    # Plot box plot and write to html file
    box_plot = px.box(df, x="class", y="sepal_width", color="class")
    box_plot.update_traces(quartilemethod="exclusive")
    box_plot.write_html(file="box_plot.html", include_plotlyjs="cdn")

    # Plot bar chart and write to html file
    bar_chart = px.bar(
        df,
        x="petal_width",
        y="petal_length",
        color="class",
        title="Iris Petal Analysis",
    )
    bar_chart.write_html(file="bar_chart.html", include_plotlyjs="cdn")

    # Plot pie chart and write to html file
    pie_chart = px.pie(df, values="petal_length", names="class")
    pie_chart.write_html(file="pie_chart.html", include_plotlyjs="cdn")

    # Plot histogram and write to html file
    hist = px.histogram(df, x="petal_width", color="class")
    hist.write_html(file="histogram.html", include_plotlyjs="cdn")

    # Print confirmation that visualisations has been generated and saved in html files
    print_heading("Visualisations generated as html files.")

    return


def transform_data(df):
    """Transforms data for training model"""

    scaler = StandardScaler()  # Load Transformer
    X_train = scaler.fit_transform(df)  # Fit and transform data using Transformer
    return X_train


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
    df = read_data()  # Read data and load it

    X_orig = df.iloc[:, :4]  # Load first four columns (Predictors) in X
    y_orig = df.iloc[:, -1:].values.ravel()  # Load last column (Predictand) in y

    stats(X_orig)  # Generate statistics for Predictors
    visuals(df)  # Plot visuals for different classes for EDA

    X = transform_data(X_orig)  # Transform data

    # Test-Train split on data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_orig, test_size=0.2, random_state=1234
    )

    # Train machine learning models
    model_training(X_train, X_test, y_train, y_test)

    # Train models using pipeline
    model_pipeline(X_train, X_test, y_train, y_test)

    print_heading("PROGRAM EXECUTED SUCCESSFULLY")


if __name__ == "__main__":
    sys.exit(main())
