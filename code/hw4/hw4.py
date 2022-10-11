# Import packages
import sys
from collections import defaultdict
from pathlib import Path

import numpy
from pandas import Series
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.graph_objs import Figure
from pydataset import data
from sklearn import datasets


def print_heading(title):
    """Prints Heading"""

    print("\n" + "*" * 80)
    print(title)
    print("*" * 80 + "\n")
    return


def print_subheading(title):
    """Prints Heading"""

    print("\n" + "*" * 40)
    print(title)
    print("*" * 40 + "\n")
    return


def get_data():
    # 1. Given a pandas dataframe
    # Contains both a response and predictors

    iris = data("iris")
    titanic = data("titanic")
    boston = data("Boston")
    breast_cancer = datasets.load_breast_cancer(as_frame=True)
    breast_cancer = breast_cancer["data"].join(breast_cancer["target"])

    # 2. Given a list of predictors and the response columns
    data_list = [
        {
            "dataset": iris,
            "name": "iris",
            "predictors": [
                "Sepal.Length",
                "Sepal.Width",
                "Petal.Length",
                "Petal.Width",
            ],
            "response": "Species",
        },
        {
            "dataset": titanic,
            "name": "titanic",
            "predictors": ["class", "age", "sex"],
            "response": "survived",
        },
        {
            "dataset": boston,
            "name": "boston housing",
            "predictors": [
                "crim",
                "zn",
                "indus",
                "chas",
                "nox",
                "rm",
                "age",
                "dis",
                "rad",
                "tax",
                "ptratio",
                "black",
                "lstat",
            ],
            "response": "medv",
        },
        {
            "dataset": breast_cancer,
            "name": "breast cancer",
            "predictors": [
                "mean radius",
                "mean texture",
                "mean perimeter",
                "mean area",
                "mean smoothness",
                "mean compactness",
                "mean concavity",
                "mean concave points",
                "mean symmetry",
                "mean fractal dimension",
                "radius error",
                "texture error",
                "perimeter error",
                "area error",
                "smoothness error",
                "compactness error",
                "concavity error",
                "concave points error",
                "symmetry error",
                "fractal dimension error",
                "worst radius",
                "worst texture",
                "worst perimeter",
                "worst area",
                "worst smoothness",
                "worst compactness",
                "worst concavity",
                "worst concave points",
                "worst symmetry",
                "worst fractal dimension",
            ],
            "response": "target",
        },
    ]

    return data_list


def check_response(response: Series) -> bool:
    print_subheading("Response Type")
    if len(response.unique()) > 2:
        print(f"{response.name} is continuous")
        return 1
    else:
        print(f"{response.name} is boolean")
        return 0


def check_predictors(predictor: Series) -> bool:
    print_subheading("Predictor Type")

    if (
        predictor.dtype.name in ["category", "object"]
        or 1.0 * predictor.nunique() / predictor.count() < 0.05
    ):
        print(f"{predictor.name} is categorical")
        return 0
    else:
        print(f"{predictor.name} is continuous")
        return 1


def cat_response_cat_predictor(
    dataset_name: str, response: Series, predictor: Series
) -> None:
    fig = px.density_heatmap(
        x=predictor, y=response, color_continuous_scale="Viridis", text_auto=True
    )
    title = f"Categorical Predictor ({predictor.name}) by Categorical Response ({response.name})"
    fig.update_layout(
        title=title,
        xaxis_title=f"Predictor ({predictor.name})",
        yaxis_title=f"Response ({response.name})",
    )
    # fig.show()
    save_plot(dataset_name, fig, title)
    return


def cat_resp_cont_predictor(
    dataset_name: str, response: Series, predictor: Series
) -> None:
    out = defaultdict(list)
    for key, value in zip(response.values, predictor.values):
        out[f"Response = {key}"].append(value)
    predictor_values = [out[key] for key in out]
    response_values = list(out.keys())

    fig1 = ff.create_distplot(predictor_values, response_values, bin_size=0.2)
    title1 = f"Continuous Predictor ({predictor.name}) by Categorical Response ({response.name})"
    fig1.update_layout(
        title=title1,
        xaxis_title=f"Predictor ({predictor.name})",
        yaxis_title="Distribution",
    )
    # fig1.show()
    save_plot(dataset_name, fig1, title1)

    fig2 = go.Figure()
    for curr_hist, curr_group in zip(predictor_values, response_values):
        fig2.add_trace(
            go.Violin(
                x=numpy.repeat(curr_group, len(curr_hist)),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    title2 = f"Continuous Predictor ({predictor.name}) by Categorical Response ({response.name})"
    fig2.update_layout(
        title=title2,
        xaxis_title=f"Response ({response.name})",
        yaxis_title=f"Predictor ({predictor.name})",
    )
    # fig2.show()
    save_plot(dataset_name, fig2, title2)
    return


def cont_resp_cat_predictor(
    dataset_name: str, response: Series, predictor: Series
) -> None:
    out = defaultdict(list)
    for key, value in zip(predictor.values, response.values):
        out[f"Predictor = {key}"].append(value)
    response_values = [out[key] for key in out]
    predictor_values = list(out.keys())

    fig1 = ff.create_distplot(response_values, predictor_values, bin_size=0.2)
    title1 = f"Continuous Response ({response.name}) by Categorical Predictor ({predictor.name})"
    fig1.update_layout(
        title=title1,
        xaxis_title=f"Response ({response.name})",
        yaxis_title="Distribution",
    )
    # fig1.show()
    save_plot(dataset_name, fig1, title1)

    fig2 = go.Figure()
    for curr_hist, curr_group in zip(response_values, predictor_values):
        fig2.add_trace(
            go.Violin(
                x=numpy.repeat(curr_group, len(curr_hist)),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    title2 = f"Continuous Response ({response.name}) by Categorical Predictor ({predictor.name})"
    fig2.update_layout(
        title=title2,
        xaxis_title=f"Predictor ({predictor.name})",
        yaxis_title=f"Response ({response.name})",
    )

    # fig2.show()
    save_plot(dataset_name, fig2, title2)
    return


def cont_response_cont_predictor(
    dataset_name: str, response: Series, predictor: Series
) -> None:
    fig = px.scatter(x=predictor, y=response, trendline="ols")
    title = f"Continuous Response ({response.name}) by Continuous Predictor ({predictor.name})"
    fig.update_layout(
        title=title,
        xaxis_title=f"Predictor ({predictor.name})",
        yaxis_title=f"Response ({response.name})",
    )
    # fig.show()
    save_plot(dataset_name, fig, title)

    return


def check_and_plot(dataset_name: str, response: Series, predictor: Series) -> None:
    response_type = check_response(response)
    predictor_type = check_predictors(predictor)
    if response_type == 0 and predictor_type == 0:
        cat_response_cat_predictor(dataset_name, response, predictor)
    elif response_type == 0 and predictor_type == 1:
        cat_resp_cont_predictor(dataset_name, response, predictor)
    elif response_type == 1 and predictor_type == 0:
        cont_resp_cat_predictor(dataset_name, response, predictor)
    elif response_type == 1 and predictor_type == 1:
        if response.dtypes not in ("int64", "float64"):
            response_name = response.name
            response = response.astype("category")
            response = response.cat.codes
            response.name = response_name
        cont_response_cont_predictor(dataset_name, response, predictor)
    else:
        print("Unable to plot the datatypes!!!")
        print(f"Response: {response.dtypes}, Predictor: {predictor.dtypes}")

    return


def save_plot(dataset_name: str, fig: Figure, name: str):
    cd = sys.path[0]
    path = f"{cd}/plots/{dataset_name}/"
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.write_html(f"{path}{name}.html")
    return


def main():
    data_list = get_data()
    for dataset in data_list:

        df = dataset["dataset"]
        dataset_name = dataset["name"]
        predictors = df[dataset["predictors"]]
        response = df[dataset["response"]]
        print_heading(f"For {dataset_name} dataset")

        for col_name in predictors.columns:
            predictor = df[col_name]
            check_and_plot(dataset_name, response, predictor)


if __name__ == "__main__":
    sys.exit(main())
