# Import packages
import sys

from pandas import Series
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
    else:
        print(f"{response.name} is boolean")


def check_predictors(df: Series):
    print_subheading("Predictors Type")

    for predictor in df.columns:

        if (
            df[predictor].dtype.name in ["category", "object"]
            or 1.0 * df[predictor].nunique() / df[predictor].count() < 0.05
        ):
            print(f"{predictor} is categorical")
        else:
            print(f"{predictor} is continuous")


def main():

    data_list = get_data()
    for dataset in data_list:

        predictors = dataset["dataset"][dataset["predictors"]]
        response = dataset["dataset"][dataset["response"]]
        print_heading(f"For {dataset['name']} dataset")
        check_response(response)
        check_predictors(predictors)


if __name__ == "__main__":
    sys.exit(main())
