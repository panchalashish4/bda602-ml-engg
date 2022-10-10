# Import packages
import sys
from collections import defaultdict

import numpy
from pandas import Series
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
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


def check_response(response: Series):
    print_subheading("Response Type")
    if len(response.unique()) > 2:
        print(f"{response.name} is continuous")
        return 1
    else:
        print(f"{response.name} is boolean")
        return 0


def check_predictors(predictor: Series):
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


def cat_response_cat_predictor(response: Series, predictor: Series):
    fig = px.density_heatmap(
        x=predictor, y=response, color_continuous_scale="Viridis", text_auto=True
    )
    fig.update_layout(
        title=f"Categorical Predictor ({predictor.name}) by Categorical Response ({response.name})",
        xaxis_title=f"Predictor ({predictor.name})",
        yaxis_title=f"Response ({response.name})",
    )
    # fig.show()

    return


def cat_resp_cont_predictor(response: Series, predictor: Series):
    out = defaultdict(list)
    for key, value in zip(response.values, predictor.values):
        out[f"Response = {key}"].append(value)
    predictor_values = [out[key] for key in out]
    response_values = list(out.keys())

    fig1 = ff.create_distplot(predictor_values, response_values, bin_size=0.2)
    fig1.update_layout(
        title=f"Continuous Predictor ({predictor.name}) by Categorical Response ({response.name})",
        xaxis_title=f"Predictor ({predictor.name})",
        yaxis_title="Distribution",
    )
    # fig1.show()

    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(predictor_values, response_values):
        fig_2.add_trace(
            go.Violin(
                x=numpy.repeat(curr_group, len(curr_hist)),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title=f"Continuous Predictor ({predictor.name}) by Categorical Response ({response.name})",
        xaxis_title=f"Response ({response.name})",
        yaxis_title=f"Predictor ({predictor.name})",
    )
    # fig_2.show()


def main():
    data_list = get_data()
    for dataset in data_list:

        df = dataset["dataset"]
        predictors = df[dataset["predictors"]]
        response = df[dataset["response"]]
        print_heading(f"For {dataset['name']} dataset")
        response_type = check_response(response)

        for col_name in predictors.columns:
            predictor = df[col_name]
            predictor_type = check_predictors(predictor)
            if response_type == 0 and predictor_type == 0:
                cat_response_cat_predictor(response, predictor)
            elif response_type == 0 and predictor_type == 1:
                cat_resp_cont_predictor(response, predictor)
            elif response_type == 1 and predictor_type == 0:
                pass
            elif response_type == 0 and predictor_type == 1:
                pass


if __name__ == "__main__":
    sys.exit(main())
