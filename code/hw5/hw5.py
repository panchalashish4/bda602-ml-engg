import sys

import brute_force_analysis
import generate_output
import pandas as pd
import resp_pred_analysis
import sqlalchemy
from pandas import DataFrame


def get_data() -> DataFrame:
    """Get data from mariadb"""

    db_user = "admin"
    db_pass = "password"  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )
    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """
        SELECT * FROM pitchers_stats_calc
    """

    df = pd.read_sql_query(query, sql_engine)
    df = df.fillna(0)

    return df


def main():
    pd.options.display.max_columns = None
    df = get_data()
    response = df["Home_Team_Wins"]
    predictors = df.drop(["Home_Team_Wins", "game_date"], axis=1)
    resp_pred_analysis.do_analysis(df, predictors, response)
    brute_force_analysis.do_analysis(df, predictors.columns, response.name)
    generate_output.just_do_it()


if __name__ == "__main__":
    sys.exit(main())
