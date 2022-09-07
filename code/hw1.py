import sys

import numpy as np
import pandas as pd
import plotly.express as px


def read_data():
    url = "https://teaching.mrsharky.com/data/iris.data"
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    df = pd.read_csv(url, names=cols, header=None)
    return df


def stats(df):
    counter = 0
    for (columnName, columnData) in df.iteritems():
        counter += 1
        mean = round(np.mean(columnData.values), 2)
        min = round(np.min(columnData.values), 2)
        max = round(np.max(columnData.values), 2)
        quart_25 = round(np.quantile(columnData.values, 0.25), 2)
        quart_50 = round(np.quantile(columnData.values, 0.5), 2)
        quart_75 = round(np.quantile(columnData.values, 0.75), 2)
        print(f"{counter}. Column Name : {columnName}")
        print(f"    Mean: {mean}")
        print(f"    Min: {min}")
        print(f"    Max: {max}")
        print(f"    Quartiles:: 25%: {quart_25}, 50%: {quart_50}, 75%: {quart_75}")
    return


def visuals(df):

    fig = px.scatter(
        df,
        x="sepal_width",
        y="sepal_length",
        color="class",
        size="petal_length",
        hover_data=["petal_width"],
    )

    fig1 = px.scatter(df, x="sepal_width", y="sepal_length", color="petal_length")
    fig.show()
    fig1.show()


def main():
    df = read_data()
    X_orig = df.iloc[:, :4]
    # df.iloc[:, -1:]
    stats(X_orig)
    # visuals(df)


if __name__ == "__main__":
    sys.exit(main())
