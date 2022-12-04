import itertools
import warnings
from collections import defaultdict
from pathlib import Path

import numpy
import pandas
import pandas as pd
import statsmodels.api
from pandas import DataFrame, Series
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.graph_objs import Figure
from scipy import stats
from scipy.stats import pearsonr
from sklearn import preprocessing


def fill_na(data):
    if isinstance(data, pandas.Series):
        return data.fillna(0)
    else:
        return numpy.array([value if value is not None else 0 for value in data])


def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    """
    Calculates correlation statistic for categorical-categorical association.
    The two measures supported are:
    1. Cramer'V ( default )
    2. Tschuprow'T

    SOURCES:
    1.) CODE: https://github.com/MavericksDS/pycorr
    2.) Used logic from:
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
        to ignore yates correction factor on 2x2
    3.) Haven't validated Tschuprow

    Bias correction and formula's taken from :
    https://www.researchgate.net/publication/270277061_A_bias-correction_for_Cramer's_V_and_Tschuprow's_T

    Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T
    Parameters:
    -----------
    x : list / ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
    tschuprow : Boolean, default = False
               For choosing Tschuprow as measure
    Returns:
    --------
    float in the range of [0,1]
    """
    corr_coeff = numpy.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pandas.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = numpy.sqrt(
                    phi2_corrected / numpy.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = numpy.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = numpy.sqrt(phi2 / numpy.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = numpy.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def cat_cont_correlation_ratio(categories, values):
    """
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """
    f_cat, _ = pandas.factorize(categories)
    cat_num = numpy.max(f_cat) + 1
    y_avg_array = numpy.zeros(cat_num)
    n_array = numpy.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[numpy.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = numpy.average(cat_measures)
    y_total_avg = numpy.sum(numpy.multiply(y_avg_array, n_array)) / numpy.sum(n_array)
    numerator = numpy.sum(
        numpy.multiply(
            n_array, numpy.power(numpy.subtract(y_avg_array, y_total_avg), 2)
        )
    )
    denominator = numpy.sum(numpy.power(numpy.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numpy.sqrt(numerator / denominator)
    return eta


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

    path = "./plots/"
    Path(path).mkdir(parents=True, exist_ok=True)
    if outside:
        filepath = f"./{name}.html"
    else:
        filepath = f"./plots/{name}.html"
    fig.write_html(file=filepath, include_plotlyjs="cdn")

    return filepath


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


def linear_regression_scores(response: Series, predictor: Series) -> tuple:
    """Calculate regression scores for continuous and continuous predictors"""

    pred = statsmodels.api.add_constant(predictor)
    linear_regression_model = statsmodels.api.OLS(response, pred)
    linear_regression_model_fitted = linear_regression_model.fit()
    # print(linear_regression_model_fitted.summary())

    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    # print(predictor.name, predictor.values, response.name, response.values)
    # Plot the figure
    fig = px.scatter(x=predictor, y=response, trendline="ols")
    title = f"(t-value={t_value}) (p-value={p_value})"
    fig.update_layout(
        title=title,
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
    # print(response_values)
    # print(predictor_values)

    fig1 = ff.create_distplot(
        response_values, predictor_values, bin_size=0.2, curve_type="normal"
    )
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


def correlation_matrix(x1, x2, score) -> Figure:
    """Create Correlation Matrix between the predictors"""

    figure = go.Figure(
        data=go.Heatmap(
            x=x1.values,
            y=x2.values,
            z=score,
            zmin=0.0,
            zmax=1.0,
            type="heatmap",
            colorscale="RdBu",
        ),
        layout={
            "title": f"{score.name}",
        },
    )
    return figure


def brute_force_matrix(
    x1_bin, x2_bin, score, bin_count, x1_name, x2_name, title
) -> Figure:
    """Create Bin Plot between the predictors with the value of Weighted Difference of Mean Response"""

    properties = []

    for index, value in enumerate(score):
        properties.append(
            {
                "x": x1_bin[index],
                "y": x2_bin[index],
                "font": {"color": "black"},
                "xref": "x1",
                "yref": "y1",
                "text": f"{round(value, 5)} (bin_count: {bin_count[index]})",
                "showarrow": False,
            }
        )

    figure = go.Figure(
        data=go.Heatmap(
            x=x1_bin,
            y=x2_bin,
            z=score,
            # zmid=0,
            type="heatmap",
            colorscale="RdBu",
        ),
        layout={
            "title": title,
            "xaxis_title": x1_name,
            "yaxis_title": x2_name,
            "annotations": properties,
        },
    )
    return figure


def corr_cont_cont(df: DataFrame, cont_cont: DataFrame, plot=True) -> DataFrame:
    """Calculates Correlation between Continuous-Continuous Predictor Pairs and Create Plot"""

    score_list = []
    for i in cont_cont:
        x = df[i[0]]
        y = df[i[1]]

        score, p_value = pearsonr(x=x, y=y)
        if plot:
            # print(y,x)
            tvalue, pvalue, url = linear_regression_scores(y, x)
            score = {
                "Predictor 1": i[0],
                "Predictor 2": i[1],
                "Pearson's r": score,
                "Absolute Value of Correlation": abs(score),
                "Plot": url,
            }
        else:
            score = {
                "Predictor 1": i[0],
                "Predictor 2": i[1],
                "Absolute Value of Correlation": abs(score),
            }
        score_list.append(score)

    score_df = pd.DataFrame(score_list)
    score_df = score_df.sort_values(by="Absolute Value of Correlation", ascending=False)

    return score_df


def corr_catg_catg(df: DataFrame, catg_catg: DataFrame, plot=True) -> DataFrame:
    """Calculates Correlation between Category-Category Predictor Pairs and Create Plot"""

    score_list = []
    for i in catg_catg:
        x = df[i[0]]
        y = df[i[1]]

        corr = cat_correlation(x, y)

        if plot:
            url = cat_response_cat_predictor(x, y)
            score = {
                "Predictor 1": i[0],
                "Predictor 2": i[1],
                "Cramer's V": corr,
                "Absolute Value of Correlation": abs(corr),
                "Plot": url,
            }
        else:
            score = {
                "Predictor 1": i[0],
                "Predictor 2": i[1],
                "Absolute Value of Correlation": abs(corr),
            }

        score_list.append(score)

    score_df = pd.DataFrame(score_list)
    score_df = score_df.sort_values(by="Absolute Value of Correlation", ascending=False)

    return score_df


def corr_catg_cont(df: DataFrame, catg_cont: DataFrame, plot=True) -> DataFrame:
    """Calculates Correlation between Category-Continuous Predictor Pairs and Create Plot"""

    score_list = []
    for i in catg_cont:
        x = df[i[0]]
        y = df[i[1]]

        ratio = cat_cont_correlation_ratio(categories=x, values=y)

        if plot:
            url = cont_resp_cat_predictor(response=y, predictor=x)
            score = {
                "Predictor 1": i[0],
                "Predictor 2": i[1],
                "Correlation Ratio": ratio,
                "Absolute Value of Correlation": abs(ratio),
                "Plot": url,
            }
        else:
            score = {
                "Predictor 1": i[0],
                "Predictor 2": i[1],
                "Absolute Value of Correlation": abs(ratio),
            }

        score_list.append(score)

    score_df = pd.DataFrame(score_list)
    score_df = score_df.sort_values(by="Absolute Value of Correlation", ascending=False)

    return score_df


def diff_mean_table(df, cont_cont_df, response_df, predictors_type) -> DataFrame:
    """Calculates Difference of Mean Response Scores between Predictors Pairs and Create Plot"""

    population_mean = response_df.mean()
    population_count = response_df.count()
    response_name = response_df.name
    final = []

    for i in cont_cont_df:

        x1 = df[i[0]]
        x2 = df[i[1]]

        predictors = x1.to_frame().join(x2)
        predictor_table = predictors.join(response_df)

        if predictors_type == "cont_cont":
            predictor_table["x1_interval"] = pd.cut(x=predictor_table[x1.name], bins=10)
            predictor_table["x2_interval"] = pd.cut(x=predictor_table[x2.name], bins=10)
            x1_unique = predictor_table["x1_interval"].sort_values().unique()
            x2_unique = predictor_table["x2_interval"].sort_values().unique()
        elif predictors_type == "catg_catg":
            x1_unique = x1.sort_values().unique()
            x2_unique = x2.sort_values().unique()
        elif predictors_type == "catg_cont":
            predictor_table["x2_interval"] = pd.cut(x=predictor_table[x2.name], bins=10)
            x1_unique = x1.sort_values().unique()
            x2_unique = predictor_table["x2_interval"].sort_values().unique()

        diff_with_mean = []
        for x2_bin in x2_unique:
            for x1_bin in x1_unique:

                if predictors_type == "cont_cont":
                    x1_check = predictor_table["x1_interval"] == x1_bin
                    x2_check = predictor_table["x2_interval"] == x2_bin
                elif predictors_type == "catg_catg":
                    x1_check = x1 == x1_bin
                    x2_check = x2 == x2_bin
                elif predictors_type == "catg_cont":
                    x1_check = x1 == x1_bin
                    x2_check = predictor_table["x2_interval"] == x2_bin

                bin_data = predictor_table[x1_check & x2_check]

                if len(bin_data) > 0:

                    bin_count = bin_data[response_df.name].count()
                    bin_weight = bin_count / population_count
                    bin_mean = bin_data[response_name].mean()
                    bin_mse = (bin_data[response_name].mean() - population_mean) ** 2

                    bin = {
                        "x1_bin": x1_bin,
                        "x2_bin": x2_bin,
                        "bin_count": bin_count,
                        "bin_mean": bin_mean,
                        "bin_mse": bin_mse,
                        "bin_weight": bin_weight,
                    }
                else:
                    bin = {
                        "x1_bin": x1_bin,
                        "x2_bin": x2_bin,
                        "bin_count": 0,
                        "bin_mean": 0,
                        "bin_mse": 0,
                        "bin_weight": 0,
                    }
                diff_with_mean.append(bin)

        diff_with_mean_table = pd.DataFrame(diff_with_mean)

        weighted_mse = pandas.Series(
            diff_with_mean_table["bin_mse"] * diff_with_mean_table["bin_weight"]
        )
        mean_squared_diff = diff_with_mean_table["bin_mse"].mean()
        mean_squared_diff_weighted = weighted_mse.sum()

        x1_bin = diff_with_mean_table["x1_bin"].astype(str)
        x2_bin = diff_with_mean_table["x2_bin"].astype(str)
        score = weighted_mse
        bin_count = diff_with_mean_table["bin_count"]

        title = f"Weighted Difference of Mean Response: {x1.name} & {x2.name}"
        figure = brute_force_matrix(
            x1_bin, x2_bin, score, bin_count, x1.name, x2.name, title
        )

        url = save_plot(figure, title)

        scores = {
            "Predictor 1": x1.name,
            "Predictor 2": x2.name,
            "Difference of Mean Response": mean_squared_diff,
            "Weighted Difference of Mean Response": mean_squared_diff_weighted,
            "Plot": url,
        }

        final.append(scores)

    final_df = pd.DataFrame(final)
    final_df = final_df.sort_values(
        by="Weighted Difference of Mean Response", ascending=False
    )

    return final_df


def cont_cont_calc(
    df: DataFrame, continuous_predictors: DataFrame, response_df: Series
) -> tuple:
    """Creates Correlation Table, Correlation Matrix and Difference of Mean Response
    for Continuous-Continuous Predictor Pairs"""

    # 1. Continuous-Continuous Pairs
    cont_cont = itertools.combinations(continuous_predictors, r=2)
    cont_cont = [combination for combination in cont_cont]
    cont_cont_df = corr_cont_cont(df, cont_cont)

    cont_cont_plot = itertools.product(
        continuous_predictors, continuous_predictors, repeat=1
    )
    cont_cont_plot = [combination for combination in cont_cont_plot]
    cont_cont_plot_df = corr_cont_cont(df, cont_cont_plot, plot=False)

    figure = correlation_matrix(
        cont_cont_plot_df["Predictor 1"],
        cont_cont_plot_df["Predictor 2"],
        cont_cont_plot_df["Absolute Value of Correlation"],
    )

    corr_url = save_plot(figure, "cont_cont_corr_matrix", True)
    mean_df = diff_mean_table(df, cont_cont, response_df, "cont_cont")

    return cont_cont_df, corr_url, mean_df


def catg_catg_calc(
    df: DataFrame, categorical_predictors: DataFrame, response_df: Series
) -> tuple:
    """Creates Correlation Table, Correlation Matrix and Difference of Mean Response
    for Category-Category Predictor Pairs"""

    # 2. Categorical-Categorical Pairs
    catg_catg = itertools.combinations(categorical_predictors, r=2)
    catg_catg = [combination for combination in catg_catg]
    catg_catg_plot = itertools.product(
        categorical_predictors, categorical_predictors, repeat=1
    )
    catg_catg_plot = [combination for combination in catg_catg_plot]

    catg_catg_df = corr_catg_catg(df, catg_catg)
    catg_catg_plot_df = corr_catg_catg(df, catg_catg_plot, plot=False)

    figure = correlation_matrix(
        catg_catg_plot_df["Predictor 1"],
        catg_catg_plot_df["Predictor 2"],
        catg_catg_plot_df["Absolute Value of Correlation"],
    )

    corr_url = save_plot(figure, "catg_catg_corr_matrix", True)
    mean_df = diff_mean_table(df, catg_catg, response_df, "catg_catg")

    return catg_catg_df, corr_url, mean_df


def catg_cont_calc(
    df: DataFrame,
    categorical_predictors: DataFrame,
    continuous_predictors: DataFrame,
    response_df: Series,
) -> tuple:
    """Creates Correlation Table, Correlation Matrix and Difference of Mean Response
    for Categorical-Continuous Predictor Pairs"""

    # 3. Categorical-Continuous Pairs
    catg_cont = itertools.product(
        categorical_predictors, continuous_predictors, repeat=1
    )
    catg_cont = [combination for combination in catg_cont]

    if len(catg_cont) > 0:
        catg_cont_df = corr_catg_cont(df, catg_cont)
        figure = correlation_matrix(
            catg_cont_df["Predictor 1"],
            catg_cont_df["Predictor 2"],
            catg_cont_df["Absolute Value of Correlation"],
        )

        corr_url = save_plot(figure, "catg_cont_corr_matrix", True)
        mean_df = diff_mean_table(df, catg_cont, response_df, "catg_cont")

        return catg_cont_df, corr_url, mean_df


def generate_html(s, title):
    """Generate html files for Score Tables and make plots clickable"""

    s = s.style.format(
        {
            "Plot": make_clickable,
        }
    )

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
    s.to_html(f"{title}.html", index=False)

    return


def do_analysis(df: DataFrame, predictors: list, response: str):
    numpy.seterr(invalid="ignore")
    response_df = df[response]

    # Check response datatype and change if it is string
    if check_response(response_df) == 0 and response_df.dtype.name == "object":
        label_encoder = preprocessing.LabelEncoder()
        df[response] = label_encoder.fit_transform(df[response])
        response_df = df[response]

    # List of continuous predictors
    continuous_predictors = [
        predictor for predictor in predictors if check_predictors(df[predictor]) == 1
    ]
    # List of categorical predictors
    categorical_predictors = [
        predictor for predictor in predictors if check_predictors(df[predictor]) == 0
    ]

    # Dataframe of continuous predictors and categorical predictors
    continuous_predictors = df[continuous_predictors]
    categorical_predictors = df[categorical_predictors]

    # Generate scores and plots when continuous predictors columns are more than one
    if continuous_predictors.shape[1] > 1:
        cont_cont_df, cont_cont_corr_url, cont_cont_mean_df = cont_cont_calc(
            df, continuous_predictors, response_df
        )
        generate_html(cont_cont_df, "cont_cont_corr_table")
        generate_html(cont_cont_mean_df, "cont_cont_mean_table")
        print("Continuous-Continuous Pairs: Successful! 3 Files Generated.")

    else:
        print(
            "Continuous-Continuous Pairs: Could not generate anything as the number of continuous predictors is "
            "less than two."
        )

    # Generate scores and plots when categorical predictors columns are more than one
    if categorical_predictors.shape[1] > 1:
        catg_catg_df, catg_catg_corr_url, catg_catg_mean_df = catg_catg_calc(
            df, categorical_predictors, response_df
        )
        generate_html(catg_catg_df, "catg_catg_corr_table")
        generate_html(catg_catg_mean_df, "catg_catg_mean_table")
        print("Categorical-Categorical Pairs: Successful! 3 Files Generated.")

    else:
        print(
            "Categorical-Categorical Pairs: Could not generate anything as the number of categorical predictors is "
            "less than two."
        )

    # Generate scores and plots when categorical and continuous predictors columns are more than zero
    if categorical_predictors.shape[1] > 0 and continuous_predictors.shape[1] > 0:
        catg_cont_df, catg_cont_corr_url, catg_cont_mean_df = catg_cont_calc(
            df, categorical_predictors, continuous_predictors, response_df
        )
        generate_html(catg_cont_df, "catg_cont_corr_table")
        generate_html(catg_cont_mean_df, "catg_cont_mean_table")
        print("Categorical-Continuous Pairs: Successful! 3 Files Generated.")

    else:
        print(
            "Categorical-Continuous Pairs: Could not generate anything as the number of categorical or continuous "
            "predictors is less than one."
        )

    return
