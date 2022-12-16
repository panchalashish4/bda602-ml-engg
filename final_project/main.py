import sys
import warnings

import brute_force_analysis
import generate_output
import pandas as pd
import resp_pred_analysis
import sqlalchemy
from pandas import DataFrame, Series
from plotly import graph_objects as go
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import text
from user_functions import generate_html, save_plot

warnings.filterwarnings("ignore")


def get_data() -> DataFrame:
    """Get data from mariadb"""

    db_user = "root"
    db_pass = "password123"  # pragma: allowlist secret
    db_host = "mariadb5:3306"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )
    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """SELECT * FROM pitchers_diff_calc"""
    with sql_engine.begin() as connection:
        df = pd.read_sql_query(text(query), connection)
        df = df.fillna(df.median())

    print("Database Connection Successful")

    return df


def model_pipeline(predictors, response):
    """Trains Classifiers using Pipeline"""

    # Train and split data
    X_train, X_test, y_train, y_test = train_test_split(
        predictors, response, test_size=0.2, shuffle=False
    )

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
        "svm": {
            "pipeline": Pipeline(
                [
                    ("StandardScaler", StandardScaler()),
                    ("SVM", LinearSVC(dual=False, random_state=1234)),
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
        "ada_boost": {
            "pipeline": Pipeline(
                [
                    ("StandardScaler", StandardScaler()),
                    ("AdaBoost", AdaBoostClassifier(random_state=1234)),
                ],
            )
        },
        "grad_boost": {
            "pipeline": Pipeline(
                [
                    ("StandardScaler", StandardScaler()),
                    ("GradientBoost", GradientBoostingClassifier(random_state=1234)),
                ],
            )
        },
    }

    # Look through all models and store all details in a list
    scores = []

    # Create an empty figure, and iteratively add new lines
    # every time we compute a new class
    fig = go.Figure()
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

    # Loop through all models and use pipelines to build models
    for model_name, model_pipe in pipelines.items():
        pipeline = model_pipe["pipeline"]
        # Fit on data using the pipeline
        pipeline.fit(X_train, y_train)
        # probability = pipeline.predict_proba(X_test)
        prediction = pipeline.predict(X_test)
        score = accuracy_score(y_test, prediction)
        precision = precision_score(prediction, y_test)
        recall = recall_score(prediction, y_test)
        mse = mean_squared_error(prediction, y_test)
        if model_name == "svm":
            pass
        else:
            probability = pipeline.predict_proba(X_test)
            fpr, tpr, _ = roc_curve(y_test, probability[:, 1])
            auc_score = roc_auc_score(y_test, prediction)
            name = f"{model_name} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))

        stats = {
            "model_name": model_name,
            "accuracy": score,
            "precision": precision,
            "recall": recall,
            "diff_precision_recall": abs(precision - recall),
            "mse": mse,
            "auc": auc_score,
        }
        scores.append(stats)

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain="domain"),
        width=700,
        height=500,
    )
    # fig.show()

    # Use the list to create dataframe and return the results
    model_scores = pd.DataFrame(scores)

    return model_scores, fig


def analysis(df: DataFrame, predictors: DataFrame, response: Series) -> tuple:
    """Performs analysis of Response-Predictors and Brute Force Analysis"""

    rpa_df = resp_pred_analysis.do_analysis(df, predictors, response)
    bfa_df_list = brute_force_analysis.do_analysis(
        df, predictors.columns, response.name
    )

    # Train Models
    models_stats, fig = model_pipeline(predictors, response)

    generate_html(models_stats, {}, "models_stats")
    save_plot(fig, "models_stats_roc")

    return models_stats, rpa_df, bfa_df_list


def model_pt_values(df: DataFrame, rpa_df: DataFrame, response: Series) -> DataFrame:
    """Remove features based on p and t value significance and train models"""

    # scores = pd.read_csv("./files/rpa_df.csv")
    scores = rpa_df
    scores[["t-score", "tplot"]] = scores["t-score"].str.split(",", expand=True)
    scores[["Predictor", "ptype"]] = scores["Predictor"].str.split(" ", expand=True)
    scores[["MeanSquaredDiff", "msd_plot"]] = scores["MeanSquaredDiff"].str.split(
        ",", expand=True
    )
    scores["p-value"] = pd.to_numeric(scores["p-value"])
    scores["t-score"] = pd.to_numeric(scores["t-score"])

    scores = scores.drop(
        ["Response (boolean)", "Plot", "tplot", "ptype", "msd_plot"], axis=1
    )
    # Shape = 96 rows x 6 columns
    scores = scores[(scores["p-value"] < 0.05) & (abs(scores["t-score"]) >= 1.96)]
    # print(scores.shape)
    # Shape = (81, 6) and RF>0.01 (53, 6)
    col_list = scores["Predictor"].tolist()
    # print(col_list)

    predictors = df[df.columns.intersection(col_list)]

    # Train Models
    models_stats_pt, fig = model_pipeline(predictors, response)
    generate_html(models_stats_pt, {}, "models_stats_pt")
    save_plot(fig, "models_stats_pt_roc")
    # 54.7
    return models_stats_pt


def corr_removal(df, bfa_df_list, response):
    """Gets the list of high correlated predictors to drop"""

    # corr_score = pd.read_csv("./files/0.csv")
    for dataframe in bfa_df_list:
        corr_score = dataframe
        corr_score = corr_score.drop(["Plot"], axis=1)
        corr_score = corr_score[corr_score["Absolute Value of Correlation"] > 0.95]
        # print(corr_score.shape)
        # Shape (102, 4)
        corr_score = corr_score.drop(
            ["Pearson's r", "Absolute Value of Correlation"], axis=1
        )

        drop_list = []
        for a, b in corr_score.itertuples(index=False):
            predictors = df.drop(
                ["game_id", "h_team", "a_team", "game_date", "Home_Team_Wins", a],
                axis=1,
            )
            print("After removing:", a)
            model_stats = analysis(df, predictors, response)
            a_accuracy = model_stats["accuracy"].max()
            a_mse = model_stats.loc[model_stats["accuracy"] == a_accuracy, "mse"].iloc[
                0
            ]
            print(f"Mean accuracy after dropping {a}: {a_accuracy}, MSE: {a_mse}")

            predictors = df.drop(
                ["game_id", "h_team", "a_team", "game_date", "Home_Team_Wins", b],
                axis=1,
            )
            print("After removing:", b)
            model_stats = analysis(df, predictors, response)
            b_accuracy = model_stats["accuracy"].max()
            b_mse = model_stats.loc[model_stats["accuracy"] == b_accuracy, "mse"].iloc[
                0
            ]
            print(f"Mean accuracy after dropping {b}: {b_accuracy}, MSE: {b_mse}")

            if b_accuracy == a_accuracy:
                if b_mse >= a_mse:
                    drop_list.append(b)
                else:
                    drop_list.append(a)
            elif b_accuracy > a_accuracy:
                drop_list.append(b)
            else:
                drop_list.append(a)

    return drop_list


def main():
    pd.options.display.max_columns = None

    # Get data and sort by date
    df = get_data()
    df = df.sort_values(by="game_date")
    df = df.reset_index(drop=True)

    # Take response and predictors
    response = df["Home_Team_Wins"]
    predictors = df.drop(
        ["game_id", "h_team", "a_team", "game_date", "Home_Team_Wins"], axis=1
    )

    # Model Training with all data
    models_stats, rpa_df, bfa_df_list = analysis(df, predictors, response)

    # Model Training by removing p and t value significance
    model_pt_values(df, rpa_df, response)

    # Dropping high correlated features
    # Based on corr_removal() function
    # It takes a lot of time therefore using list
    drop_list = [
        "r_CERA",
        "r_CERA_100",
        "r_oppBattingAvg_100",
        "r_DICE_100",
        "r_sp_innings_pitched",
        "r_strikeout9_100",
        "r_Strikeout9",
        "r_powerFinesseRatio_100",
        "r_WHIP",
        "r_WHIP_100",
        "r_DICE",
        "r_oppBattingAvg",
        "r_powerFinesseRatio",
        "r_sp_innings_pitched_100",
        "r_perf_ratio_100",
        "r_hitsAllowed9_100",
        "r_perf_ratio",
        "r_hitsAllowed9",
        "r_sp_games_started",
        "r_sp_games_started_100",
        "r_pyth_exp_100",
        "r_pyth_exp",
        "rcp_strikeoutToWalkRatio_100",
        "r_strikeoutToWalkRatio",
        "rcp_basesOnBalls9_100",
        "r_basesOnBalls9",
        "rcp_homeRuns9",
        "d_sp_innings_pitched",
        "r_hitsAllowed9",
        "d_hitsAllowed9",
        "r_sp_basesOnBalls9",
        "rcp_homeRuns9_100",
        "r_oppBattingAvg",
        "d_oppBattingAvg",
        "d_WHIP",
        "d_WHIP",
        "rcp_sp_basesOnBalls9_100",
        "r_perf_ratio",
        "rcp_perf_ratio",
        "d_oppBattingAvg",
        "rcp_oppBattingAvg",
        "r_oppBattingAvg",
        "r_oppBattingAvg",
        "rcp_oppBattingAvg",
        "r_WHIP_100",
        "d_WHIP_100",
        "r_oppBattingAvg_100",
        "rcp_oppBattingAvg_100",
        "d_perf_ratio_100",
        "d_perf_ratio_100",
        "d_sp_games_started_100",
        "d_oppBattingAvg",
        "d_oppBattingAvg",
        "d_basesOnBalls9",
        "rcp_oppBattingAvg",
        "r_oppBattingAvg",
        "d_hitsAllowed9_100",
        "r_hitsAllowed9_100",
        "rcp_sp_complete_games",
        "d_WHIP",
        "d_strikeoutToWalkRatio",
        "r_basesOnBalls9",
        "d_strikeoutToWalkRatio",
    ]

    predictors = predictors.drop(drop_list, axis=1)
    model_stats_corr, fig = model_pipeline(predictors, response)
    generate_html(model_stats_corr, {}, "models_stats_corr")
    save_plot(fig, "models_stats_corr_roc")
    print("Model Training Completed")

    generate_output.just_do_it()


if __name__ == "__main__":
    sys.exit(main())
