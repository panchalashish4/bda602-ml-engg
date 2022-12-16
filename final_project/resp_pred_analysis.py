# Import packages
from collections import defaultdict

import numpy
import pandas as pd
import statsmodels.api
from pandas import DataFrame, Series
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from user_functions import (
    cat_response_cat_predictor,
    check_predictors,
    check_response,
    cont_resp_cat_predictor,
    figures_to_html,
    generate_html,
    linear_regression_scores,
    make_clickable,
    save_plot,
)


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


def check_and_plot(response: Series, predictor: Series) -> tuple:
    """Check type of response and predictors, create plots, calculate t-value and p-value for predictors"""

    response_type = check_response(response)
    predictor_type = check_predictors(predictor)
    if response_type == 0 and predictor_type == 0:
        url = cat_response_cat_predictor(response, predictor, "resp_pred")
        t_value = None
        p_value = None
        t_url = None
    elif response_type == 0 and predictor_type == 1:
        url = cat_resp_cont_predictor(response, predictor)
        t_value, p_value, t_url = logistic_regression_scores(response, predictor)
    elif response_type == 1 and predictor_type == 0:
        url = cont_resp_cat_predictor(response, predictor, "resp_pred")
        t_value = None
        p_value = None
        t_url = None
    elif response_type == 1 and predictor_type == 1:
        url = cont_response_cont_predictor(response, predictor)
        t_value, p_value, t_url = linear_regression_scores(
            response, predictor, "resp_pred"
        )
    else:
        print("Unable to plot the datatypes!!!")
        print(f"Response: {response.dtypes}, Predictor: {predictor.dtypes}")

    return t_value, p_value, url, t_url


def logistic_regression_scores(response: Series, predictor: Series) -> tuple:
    """Calculate regression scores for continuous predictors and boolean response"""

    pred = statsmodels.api.add_constant(predictor)
    logistic_regression_model = statsmodels.api.Logit(response, pred)
    logistic_regression_model_fitted = logistic_regression_model.fit(disp=0)
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

    # print(difference_with_mean_table)

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
    fig.update_xaxes(title_text=f"<b>Predictor Bin ({predictor.name})<b>")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>Population</b>", secondary_y=True)
    fig.update_yaxes(title_text=f"<b>Response {response.name}</b>", secondary_y=False)

    # fig.show()
    url = save_plot(fig, title)

    # print(difference_with_mean_table)
    mean_squared_diff = difference_with_mean_table["mean_squared_diff"].sum()
    mean_squared_diff_weighted = difference_with_mean_table[
        "mean_squared_diff_weighted"
    ].sum()

    return mean_squared_diff, mean_squared_diff_weighted, url


def do_analysis(df: DataFrame, predictors: DataFrame, response: Series):

    # Dictionary to score values of all the predictors
    predictors_dict = {}
    # Iterate through all predictors
    for col_name in predictors.columns:
        predictor = df[col_name]

        # Get t-value, p-value, plot with response and t-score plot
        t_value, p_value, url, t_url = check_and_plot(response, predictor)

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
    rf = s.pop("RF VarImp")
    s.insert(5, "RF VarImp", rf)
    df_return = s

    click_dict = {
        "Plot": make_clickable,
        "t-score": make_clickable,
        "MeanSquaredDiff": make_clickable,
    }

    generate_html(s, click_dict, "scores")
    print("Response - Predictor Analysis Completed")

    return df_return
