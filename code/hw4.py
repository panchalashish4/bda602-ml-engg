# Import packages
import sys
import webbrowser
from collections import defaultdict
from pathlib import Path

import numpy
import pandas as pd

# import plotly.data
import statsmodels
from pandas import DataFrame, Series
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

# from pydataset import data
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def print_heading(title):
    """Prints Heading"""

    print("\n" + "*" * 80)
    print(title)
    print("*" * 80 + "\n")
    return


def get_data():
    """Loading the data"""

    # 1. Given a pandas dataframe
    # Contains both a response and predictors
    # tips = plotly.data.tips(pretty_names=False)
    # titanic = data("titanic")
    # boston = data("Boston")
    breast_cancer = datasets.load_breast_cancer(as_frame=True)
    breast_cancer = breast_cancer["data"].join(breast_cancer["target"])
    diabetes = datasets.load_diabetes(as_frame=True)
    diabetes = diabetes["data"].join(diabetes["target"])

    # 2. Given a list of predictors and the response columns
    data_list = [
        # {
        #     "dataset": iris,
        #     "name": "iris",
        #     "predictors": [
        #         "Sepal.Length",
        #         "Sepal.Width",
        #         "Petal.Length",
        #         "Petal.Width",
        #     ],
        #     "response": "Species",
        # },
        # {
        #     "dataset": titanic,
        #     "name": "titanic",
        #     "predictors": ["class", "age", "sex"],
        #     "response": "survived",
        # },
        # {
        #     "dataset": tips,
        #     "name": "tips",
        #     "predictors": ["total_bill", "sex", "smoker", "day", "time", "size"],
        #     "response": "tip",
        # }  # ,
        {
            "dataset": diabetes,
            "name": "diabetes",
            "predictors": [
                "age",
                "sex",
                "bmi",
                "bp",
                "s1",
                "s2",
                "s3",
                "s4",
                "s5",
                "s6",
            ],
            "response": "target",
        }
        # {
        #     "dataset": boston,
        #     "name": "boston housing",
        #     "predictors": [
        #         "crim",
        #         "zn",
        #         "indus",
        #         "chas",
        #         "nox",
        #         "rm",
        #         "age",
        #         "dis",
        #         "rad",
        #         "tax",
        #         "ptratio",
        #         "black",
        #         "lstat",
        #     ],
        #     "response": "medv",
        # },
        # {
        #     "dataset": breast_cancer,
        #     "name": "breast cancer",
        #     "predictors": [
        #         "mean radius",
        #         "mean texture",
        #         "mean perimeter",
        #         "mean area",
        #         "mean smoothness",
        #         "mean compactness",
        #         "mean concavity",
        #         "mean concave points",
        #         "mean symmetry",
        #         "mean fractal dimension",
        #         "radius error",
        #         "texture error",
        #         "perimeter error",
        #         "area error",
        #         "smoothness error",
        #         "compactness error",
        #         "concavity error",
        #         "concave points error",
        #         "symmetry error",
        #         "fractal dimension error",
        #         "worst radius",
        #         "worst texture",
        #         "worst perimeter",
        #         "worst area",
        #         "worst smoothness",
        #         "worst compactness",
        #         "worst concavity",
        #         "worst concave points",
        #         "worst symmetry",
        #         "worst fractal dimension",
        #     ],
        #     "response": "target",
        # },
    ]

    return data_list


def check_response(response: Series) -> int:
    """Check type of response"""

    if len(response.unique()) == 2:
        return 0
    else:
        return 1


def check_predictors(predictor: Series) -> int:
    """Check type of predictors"""

    if (
        predictor.dtype.name in ["category", "object"]
        or 1.0 * predictor.nunique() / predictor.count() < 0.05
    ):
        return 0
    else:
        return 1


def cat_response_cat_predictor(response: Series, predictor: Series) -> str:
    """Create plot for categorical response by categorical predictor"""

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
    url = save_plot(fig, title)
    return url


def cat_resp_cont_predictor(response: Series, predictor: Series) -> str:
    """Create plot for categorical response by continuous predictor"""

    out = defaultdict(list)
    for key, value in zip(response.values, predictor.values):
        out[f"Response = {key}"].append(value)
    predictor_values = [out[key] for key in out]
    response_values = list(out.keys())

    fig1 = ff.create_distplot(predictor_values, response_values, bin_size=0.2)
    title = f"Continuous Predictor ({predictor.name}) by Categorical Response ({response.name})"
    fig1.update_layout(
        title=title,
        xaxis_title=f"Predictor ({predictor.name})",
        yaxis_title="Distribution",
    )
    # fig1.show()
    save_plot(fig1, title)

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

    fig2.update_layout(
        title=title,
        xaxis_title=f"Response ({response.name})",
        yaxis_title=f"Predictor ({predictor.name})",
    )
    # fig2.show()
    save_plot(fig2, title)

    # Combine both plots on one single html page
    url = figures_to_html([fig1, fig2], filename=f"{title}_combine")

    return url


def cont_resp_cat_predictor(response: Series, predictor: Series) -> str:
    """Create plot for continuous response by categorical predictor"""

    out = defaultdict(list)
    for key, value in zip(predictor.values, response.values):
        out[f"Predictor = {key}"].append(value)
    response_values = [out[key] for key in out]
    predictor_values = list(out.keys())

    fig1 = ff.create_distplot(response_values, predictor_values, bin_size=0.2)
    title = f"Continuous Response ({response.name}) by Categorical Predictor ({predictor.name})"
    fig1.update_layout(
        title=title,
        xaxis_title=f"Response ({response.name})",
        yaxis_title="Distribution",
    )
    # fig1.show()
    save_plot(fig1, title)

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

    fig2.update_layout(
        title=title,
        xaxis_title=f"Predictor ({predictor.name})",
        yaxis_title=f"Response ({response.name})",
    )

    # fig2.show()
    save_plot(fig2, title)

    # Combines both figures on one single html page
    url = figures_to_html([fig1, fig2], filename=f"{title}_combine")

    return url


def figures_to_html(figs, filename="dashboard.html") -> str:
    """Combines figures on one single html page"""

    path = "./plots/"
    Path(path).mkdir(parents=True, exist_ok=True)
    filepath = f"./plots/{filename}.html"
    with open(filepath, "w") as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")
        for fig in figs:
            inner_html = fig.to_html().split("<body>")[1].split("</body>")[0]
            dashboard.write(inner_html)
        dashboard.write("</body></html>" + "\n")

    return filepath


def cont_response_cont_predictor(response: Series, predictor: Series) -> str:
    """Create plot for continuous response by continuous predictor"""

    fig = px.scatter(x=predictor, y=response, trendline="ols")
    title = f"Continuous Response ({response.name}) by Continuous Predictor ({predictor.name})"
    fig.update_layout(
        title=title,
        xaxis_title=f"Predictor ({predictor.name})",
        yaxis_title=f"Response ({response.name})",
    )
    # fig.show()
    url = save_plot(fig, title)

    return url


def check_and_plot(dataset_name: str, response: Series, predictor: Series) -> tuple:
    """Check type of response and predictors, create plots, calculate t-value and p-value for predictors"""

    response_type = check_response(response)
    predictor_type = check_predictors(predictor)
    if response_type == 0 and predictor_type == 0:
        url = cat_response_cat_predictor(response, predictor)
        t_value = None
        p_value = None
        t_url = None
    elif response_type == 0 and predictor_type == 1:
        url = cat_resp_cont_predictor(response, predictor)
        t_value, p_value, t_url = logistic_regression_scores(response, predictor)
    elif response_type == 1 and predictor_type == 0:
        url = cont_resp_cat_predictor(response, predictor)
        t_value = None
        p_value = None
        t_url = None
    elif response_type == 1 and predictor_type == 1:
        url = cont_response_cont_predictor(response, predictor)
        t_value, p_value, t_url = linear_regression_scores(response, predictor)
    else:
        print("Unable to plot the datatypes!!!")
        print(f"Response: {response.dtypes}, Predictor: {predictor.dtypes}")

    return t_value, p_value, url, t_url


def save_plot(fig: Figure, name: str):
    """Saves plots in html file"""

    path = "./plots/"
    Path(path).mkdir(parents=True, exist_ok=True)
    filepath = f"./plots/{name}.html"
    fig.write_html(filepath)

    return filepath


def linear_regression_scores(response: Series, predictor: Series) -> tuple:
    """Calculate regression scores for continuous predictors and continuous response"""

    pred = statsmodels.api.add_constant(predictor)
    linear_regression_model = statsmodels.api.OLS(response, pred)
    linear_regression_model_fitted = linear_regression_model.fit()
    # print(linear_regression_model_fitted.summary())

    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    # Plot the figure
    fig = px.scatter(x=predictor, y=response, trendline="ols")
    title = f"Variable: {predictor.name}: (t-value={t_value}) (p-value={p_value})"
    fig.update_layout(
        title=title,
        xaxis_title=f"Variable: {predictor.name}",
        yaxis_title=f"Response: {response.name}",
    )
    # fig.show()
    url = save_plot(fig, title)

    return t_value, p_value, url


def logistic_regression_scores(response: Series, predictor: Series) -> tuple:
    """Calculate regression scores for continuous predictors and boolean response"""

    pred = statsmodels.api.add_constant(predictor)
    logistic_regression_model = statsmodels.api.Logit(response, pred)
    logistic_regression_model_fitted = logistic_regression_model.fit()
    # print(logistic_regression_model_fitted.summary())

    # Get the stats
    t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])

    # Plot the figure
    fig = px.scatter(x=predictor, y=response, trendline="ols")
    title = f"Variable: {predictor.name}: (t-value={t_value}) (p-value={p_value})"
    fig.update_layout(
        title=title,
        xaxis_title=f"Variable: {predictor.name}",
        yaxis_title=f"Response: {response.name}",
    )
    # fig.show()
    url = save_plot(fig, title)

    return t_value, p_value, url


def random_forest_scores(
    response: Series, predictors: DataFrame, df: DataFrame
) -> dict:
    """Calculate random forest scores for continuous predictors"""

    continuous_predictors = [
        predictor for predictor in predictors if check_predictors(df[predictor]) == 1
    ]
    continuous_predictors = df[continuous_predictors]

    if check_response(response) == 1 and continuous_predictors.shape[1] > 0:
        random_forest_regressor = RandomForestRegressor(random_state=42)
        random_forest_regressor.fit(continuous_predictors, response)
        features_importance = random_forest_regressor.feature_importances_
    elif check_response(response) == 0 and continuous_predictors.shape[1] > 0:
        random_forest_classifier = RandomForestClassifier(random_state=42)
        random_forest_classifier.fit(continuous_predictors, response)
        features_importance = random_forest_classifier.feature_importances_

    scores = {}
    for i, importance in enumerate(features_importance):
        scores[continuous_predictors.columns[i]] = importance
    # print(scores)

    return scores


def difference_with_mean_of_response(response: Series, predictor: Series) -> int:
    """Calculate difference with mean of response scores for all predictors and generates the plot"""

    difference_with_mean_table = pd.DataFrame()
    predictor_table = predictor.to_frame().join(response)
    population_mean = response.mean()

    if check_predictors(predictor) == 1:

        predictor_bins = pd.cut(predictor, 10, duplicates="drop")
        predictor_table["LowerBin"] = pd.Series(
            [predictor_bin.left for predictor_bin in predictor_bins]
        )
        predictor_table["UpperBin"] = pd.Series(
            [predictor_bin.right for predictor_bin in predictor_bins]
        )

        bins_center = predictor_table.groupby(by=["LowerBin", "UpperBin"]).median()
        bin_count = predictor_table.groupby(by=["LowerBin", "UpperBin"]).count()
        bin_mean = predictor_table.groupby(by=["LowerBin", "UpperBin"]).mean()

        difference_with_mean_table["BinCenters"] = bins_center[predictor.name]
        difference_with_mean_table["BinCount"] = bin_count[predictor.name]
        difference_with_mean_table["BinMean"] = bin_mean[response.name]

    elif check_predictors(predictor) == 0:

        bin_count = predictor_table.groupby(by=[predictor.name]).count()
        bin_mean = predictor_table.groupby(by=[predictor.name]).mean()

        difference_with_mean_table["BinCount"] = bin_count[response.name]
        difference_with_mean_table["BinMean"] = bin_mean[response.name]

    difference_with_mean_table["PopulationMean"] = population_mean
    mean_squared_difference = (
        difference_with_mean_table["BinMean"]
        - difference_with_mean_table["PopulationMean"]
    ) ** 2
    difference_with_mean_table["mean_squared_diff"] = mean_squared_difference
    difference_with_mean_table["Weight"] = (
        difference_with_mean_table["BinCount"] / predictor.count()
    )
    difference_with_mean_table["mean_squared_diff_weighted"] = (
        difference_with_mean_table["mean_squared_diff"]
        * difference_with_mean_table["Weight"]
    )
    difference_with_mean_table = difference_with_mean_table.reset_index()

    if check_predictors(predictor) == 1:
        x_axis = difference_with_mean_table["BinCenters"]
    elif check_predictors(predictor) == 0:
        x_axis = difference_with_mean_table[predictor.name]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=x_axis,
            y=difference_with_mean_table["BinCount"],
            name="Population",
            opacity=0.5,
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(x=x_axis, y=difference_with_mean_table["BinMean"], name="Bin Mean"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=[x_axis.min(), x_axis.max()],
            y=[
                difference_with_mean_table["PopulationMean"][0],
                difference_with_mean_table["PopulationMean"][0],
            ],
            mode="lines",
            line=dict(color="green", width=2),
            name="Population Mean",
        )
    )

    fig.add_hline(difference_with_mean_table["PopulationMean"][0], line_color="green")

    title = f"Bin Difference with Mean of Response vs Bin ({predictor.name})"
    # Add figure title
    fig.update_layout(title_text="<b>Bin Difference with Mean of Response vs Bin<b>")

    # Set x-axis title
    fig.update_xaxes(title_text="<b>Predictor Bin<b>")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Population</b>", secondary_y=True)
    fig.update_yaxes(title_text="<b>Response</b>", secondary_y=False)

    # fig.show()
    url = save_plot(fig, title)

    # print(difference_with_mean_table)
    mean_squared_diff = difference_with_mean_table["mean_squared_diff"].sum()
    mean_squared_diff_weighted = difference_with_mean_table[
        "mean_squared_diff_weighted"
    ].sum()

    return mean_squared_diff, mean_squared_diff_weighted, url


def make_clickable(val):
    """Make urls in dataframe clickable for html output"""

    if val is not None:
        if "," in val:
            x = val.split(",")
            return f'{x[0]} <a target="_blank" href="{x[1]}">link to plot</a>'
        else:
            return f'<a target="_blank" href="{val}">link to plot</a>'
    else:
        return val


def main():

    pd.options.display.max_columns = None
    # Load data
    data_list = get_data()
    # Iterate through the dictionary
    for dataset in data_list:
        df = dataset["dataset"]
        dataset_name = dataset["name"]
        predictors = df[dataset["predictors"]]
        response = df[dataset["response"]]
        print_heading(f"For {dataset_name} dataset")

        # Dictionary to score values of all the predictors
        predictors_dict = {}
        # Iterate through all predictors
        for col_name in predictors.columns:
            predictor = df[col_name]

            # Get t-value, p-value, plot with response and t-score plot
            t_value, p_value, url, t_url = check_and_plot(
                dataset_name, response, predictor
            )

            # Get mean_squared_diff, mean_squared_diff_weighted and plot
            msd, mswd, diff_url = difference_with_mean_of_response(response, predictor)

            # Get Predictors column name with category
            if check_predictors(predictor) == 1:
                predictor_text_name = f"{predictor.name} (cont)"
            elif check_predictors(predictor) == 0:
                predictor_text_name = f"{predictor.name} (cat)"

            # Save single predictor values in dictionary
            predictor_dict = {
                "Predictor": predictor_text_name,
                "Plot": url,
                "t-score": t_value.astype(str) + "," + t_url
                if t_value is not None
                else None,
                "p-value": p_value,
                "MeanSquaredDiff": msd.astype(str) + "," + diff_url,
                "MeanSquaredDiffWeighted": mswd,
            }

            # Add predictor dictionary in all predictors dictionary
            predictors_dict[predictor.name] = predictor_dict

        # Get Random Forest Scores
        scores = random_forest_scores(response, predictors, df)

        # print(predictors_dict)
        # Check if predictor has RF value and it in all predictors dictionary
        # Else give None
        for key, value in scores.items():
            for key2, value2 in predictors_dict.items():
                if key == key2:
                    predictors_dict[key2]["RF VarImp"] = value

        # Create column name for response column
        response_name = response.name
        if check_response(response) == 1:
            response_text_type = "Response (cont)"
        elif check_response(response) == 0:
            response_text_type = "Response (boolean)"
        else:
            response_text_type = "Response (Not Identified)"

        # Create dataframe from all predictors dictionary
        scores_df = pd.DataFrame(predictors_dict)
        # Transpose it and remove index
        s = scores_df.T.reset_index(drop=True)
        # Insert response as first column
        s.insert(0, response_text_type, response_name)
        # Insert None for nan values
        s = s.where(pd.notnull(s), None)

        # Make url clickable
        s = s.style.format(
            {
                "Plot": make_clickable,
                "t-score": make_clickable,
                "MeanSquaredDiff": make_clickable,
            }
        )

        # Set borders for the table
        s = s.set_table_styles(
            [
                {"selector": "", "props": [("border", "1px solid grey")]},
                {"selector": "tbody td", "props": [("border", "1px solid grey")]},
                {"selector": "th", "props": [("border", "1px solid grey")]},
            ]
        )

        # Export the html file
        s.to_html("scores.html", index=False)

        # Open the html file
        webbrowser.open("scores.html")


if __name__ == "__main__":
    sys.exit(main())
