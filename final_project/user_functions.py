from collections import defaultdict
from pathlib import Path

import numpy
import statsmodels.api
from pandas import DataFrame, Series
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.graph_objs import Figure


def check_response(response: Series) -> int:
    """Check type of response"""

    if len(response.unique()) == 2:
        return 0
    else:
        return 1


def check_predictors(predictor: Series) -> int:
    """Check type of predictors"""

    if predictor.dtype.name in ["category", "object", "bool"]:
        return 0
    else:
        return 1


def save_plot(fig: Figure, name: str, outside=False) -> str:
    """Saves plots in html file"""

    path = "./files/plots/"
    Path(path).mkdir(parents=True, exist_ok=True)
    if outside:
        filepath = f"./files/{name}.html"
        fig.write_html(file=filepath, include_plotlyjs="cdn")
    else:
        filepath = f"./files/plots/{name}.html"
        fig.write_html(file=filepath, include_plotlyjs="cdn")
        filepath = f"./plots/{name}.html"

    return filepath


def figures_to_html(figs, filename="dashboard.html") -> str:
    """Combines figures on one single html page"""

    path = "./files/plots/"
    Path(path).mkdir(parents=True, exist_ok=True)
    filepath = f"./files/plots/{filename}.html"
    with open(filepath, "w") as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")
        for fig in figs:
            inner_html = fig.to_html().split("<body>")[1].split("</body>")[0]
            dashboard.write(inner_html)
        dashboard.write("</body></html>" + "\n")
    filepath = f"./plots/{filename}.html"

    return filepath


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


def linear_regression_scores(response: Series, predictor: Series, type: str) -> tuple:
    """Calculate regression scores for continuous and continuous predictors"""

    pred = statsmodels.api.add_constant(predictor)
    linear_regression_model = statsmodels.api.OLS(response, pred)
    linear_regression_model_fitted = linear_regression_model.fit()
    # print(linear_regression_model_fitted.summary())

    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    # Plot the figure
    fig = px.scatter(x=predictor, y=response, trendline="ols")

    if type == "resp_pred":
        title = f"Variable: {predictor.name}: (t-value={t_value}) (p-value={p_value})"
        fig.update_layout(
            title=title,
            xaxis_title=f"Variable: {predictor.name}",
            yaxis_title=f"Response: {response.name}",
        )
    elif type == "pred_pred":
        fig.update_layout(
            title=f"(t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"{predictor.name}",
            yaxis_title=f"{response.name}",
        )

    # fig.show()
    plot_name = f"{predictor.name}_{response.name}_linear_regression".lower()
    url = save_plot(fig, plot_name)

    return t_value, p_value, url


def cat_response_cat_predictor(response: Series, predictor: Series) -> str:
    """Create plot for categorical predictor by categorical predictor"""

    fig = px.density_heatmap(
        x=predictor, y=response, color_continuous_scale="Viridis", text_auto=True
    )

    if type == "resp_pred":
        title = f"Categorical Predictor ({predictor.name}) by Categorical Response ({response.name})"
        fig.update_layout(
            title=title,
            xaxis_title=f"Predictor ({predictor.name})",
            yaxis_title=f"Response ({response.name})",
        )
    elif type == "pred_pred":
        title = f"Categorical Predictor ({predictor.name}) by Categorical Predictor ({response.name})"
        fig.update_layout(
            title=title,
            xaxis_title=f"Predictor ({predictor.name})",
            yaxis_title=f"Predictor ({response.name})",
        )

    # fig.show()
    url = save_plot(fig, title)
    return url


def cont_resp_cat_predictor(response: Series, predictor: Series) -> str:
    """Create plot for continuous predictor by categorical predictor"""

    out = defaultdict(list)
    for key, value in zip(predictor.values, response.values):
        out[f"Predictor = {key}"].append(value)
    response_values = [out[key] for key in out]
    predictor_values = list(out.keys())

    fig1 = ff.create_distplot(
        response_values, predictor_values, bin_size=0.2, curve_type="normal"
    )

    if type == "resp_pred":
        title = f"Continuous Response ({response.name}) by Categorical Predictor ({predictor.name})"
        fig1.update_layout(
            title=title,
            xaxis_title=f"Response ({response.name})",
            yaxis_title="Distribution",
        )
    elif type == "pred_pred":
        title = f"Continuous Predictor ({response.name}) by Categorical Predictor ({predictor.name})"
        fig1.update_layout(
            title=title,
            xaxis_title=f"Predictor ({response.name})",
            yaxis_title=f"Distribution: {predictor.name}",
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

    if type == "resp_pred":
        fig2.update_layout(
            title=title,
            xaxis_title=f"Predictor ({predictor.name})",
            yaxis_title=f"Response ({response.name})",
        )
    elif type == "pred_pred":
        fig2.update_layout(
            title=title,
            xaxis_title=f"Predictor ({predictor.name})",
            yaxis_title=f"Predictor ({response.name})",
        )

    # fig2.show()
    save_plot(fig2, title)

    # Combines both figures on one single html page
    url = figures_to_html([fig1, fig2], filename=f"{title}_combine")

    return url


def generate_html(s: DataFrame, click_dict: dict, title: str):
    """Generate html files for Score Tables and make plots clickable"""

    s = s.style.format(click_dict)

    # Reference Link
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
    s = s.set_table_styles(
        [
            {"selector": "tr:hover", "props": [("background-color", "yellow")]},
            {
                "selector": ".index_name",
                "props": "font-style: italic; color: darkgrey; font-weight:normal;",
            },
            {
                "selector": "th:not(.index_name)",
                "props": "background-color: #000066; color: white;",
            },
            {"selector": "th.col_heading", "props": "text-align: center;"},
            {
                "selector": "th.col_heading.level0",
                "props": 'font-family: "Times New Roman", Times, serif;'
                "font-size: 1.2em;",
            },
            {
                "selector": "td",
                "props": 'font-family: "Times New Roman", Times, serif;'
                "text-align: center; font-size: 1em; font-weight: normal;",
            },
            {"selector": "", "props": [("border", "1.5px solid black")]},
            {"selector": "tbody td", "props": [("border", "1px solid grey")]},
            {"selector": "th", "props": [("border", "1px solid grey")]},
        ]
    )

    s.hide(axis="index")
    s.to_html(f"./files/{title}.html", index=False)

    return
