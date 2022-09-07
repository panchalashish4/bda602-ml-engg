import sys

import numpy as np
import pandas as pd
import plotly.express as px
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


def model_training(model_name, model, X, y):
    """Trains machine learning models and calculates accuracy, precision and recall"""

    # Fit Model on data
    model.fit(X, y)

    # Predict the label using model
    prediction = model.predict(X)
    # Calculate accuracy of model
    accuracy = sklearn.metrics.accuracy_score(y, prediction)
    # Calculate precision of model
    precision = sklearn.metrics.precision_score(prediction, y, average="micro")
    # Calculate recall of model
    recall = sklearn.metrics.recall_score(prediction, y, average="micro")

    # Print summary of model
    print_heading(f"Model Predictions for {model_name}")
    print(f"{model_name} Accuracy:: Random Forest:", accuracy)
    print(f"{model_name} Precision:: Random Forest:", precision)
    print(f"{model_name} Recall:: Random Forest:", recall)

    return


def model_pipeline(X, y):
    """Trains Random Forest Classifier using Pipeline"""

    print_heading("Model via Pipeline Predictions")
    # Create Pipeline
    pipeline = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )

    # Fit on data using the pipeline
    pipeline.fit(X, y)

    # Calculate probability and prediction for model using pipeline
    probability = pipeline.predict_proba(X)
    prediction = pipeline.predict(X)
    print(f"Probability: {probability}")
    print(f"Predictions: {prediction}")
    return


def main():
    df = read_data()  # Read data and load it

    X_orig = df.iloc[:, :4]  # Load first four columns (Predictors) in X
    y = df.iloc[:, -1:].values.ravel()  # Load last column (Predictand) in y

    stats(X_orig)  # Generate statistics for Predictors
    visuals(df)  # Plot visuals for different classes for EDA
    X_train = transform_data(X_orig)  # Transform data

    # Load classifiers
    random_forest = RandomForestClassifier(random_state=1234)
    decision_tree = DecisionTreeClassifier()
    # Train model for random forest
    model_training("Random Forest", random_forest, X_train, y)
    # Train model for decision tree
    model_training("Decision Tree", decision_tree, X_train, y)

    # Train random forest model using pipeline
    model_pipeline(X_train, y)

    print_heading("PROGRAM EXECUTED SUCCESSFULLY")


if __name__ == "__main__":
    sys.exit(main())
