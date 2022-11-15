import sys

import pandas
import sqlalchemy


def main():
    db_user = "admin"
    db_pass = "password"  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """
        SELECT
                g.game_id
                , away_t.name AS away_team
                , home_t.name AS home_team
                , MAX(home_score) AS home_score
                , MAX(away_score) AS away_score
                , CASE
                    WHEN MAX(home_score) > MAX(away_score) THEN CONCAT(home_t.name, " Wins")
                    WHEN MAX(home_score) < MAX(away_score) THEN CONCAT(away_t.name, " Wins")
                    ELSE CONCAT(home_t.name, " and ", away_t.name, " tied") END AS winning_team
                , MAX(inning) AS last_inning
            FROM inning i
            JOIN game g ON g.game_id = i.game_id
            JOIN team away_t ON g.away_team_id = away_t.team_id
            JOIN team home_t ON g.home_team_id = home_t.team_id
            GROUP BY g.game_id, away_t.name, home_t.name
            ORDER BY g.game_id, away_t.name, home_t.name
    """
    df = pandas.read_sql_query(query, sql_engine)
    print(df.head())


if __name__ == "__main__":
    sys.exit(main())
