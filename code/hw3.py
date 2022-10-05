import sys

from pyspark import StorageLevel, keyword_only
from pyspark.ml import Transformer
from pyspark.sql import DataFrame, SparkSession

# Setting up variables for mariadb connection
database = "baseball"
user = "admin"
password = "password"
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"


class RollingAvgTransformer(Transformer):
    """Class for calculating rolling average transformation"""

    @keyword_only
    def __init__(self):
        super(RollingAvgTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, spark_session, df):
        ra_df = rolling_avg(spark_session, df)
        return ra_df


def read_data(spark_session: SparkSession, table_query: str) -> DataFrame:
    """Read data from mariadb using sql query"""

    df = (
        spark_session.read.format("jdbc")
        .options(
            url=jdbc_url,
            driver=jdbc_driver,
            query=table_query,
            user=user,
            password=password,
        )
        .load()
    )
    return df


def get_data(spark_session: SparkSession) -> DataFrame:
    """Get data from game and batter_counts table, joins into a new dataframe."""

    game_query = """
    SELECT 
        game_id,
        date(local_date) as local_date
    FROM game
    """

    batter_count_query = """
    SELECT 
        game_id,
        batter,
        atBat,
        Hit
    FROM batter_counts
    """

    game = read_data(spark_session, game_query)
    batter_counts = read_data(spark_session, batter_count_query)
    # Join game and batter_counts table
    batter_data = batter_counts.join(game, on="game_id")

    return batter_data


def rolling_avg(spark_session: SparkSession, df: DataFrame) -> DataFrame:
    """Function to calculate rolling average given a batter_data dataframe is passed accordingly."""

    sql = """
        SELECT
    		bd1.batter,
    		bd1.local_date,
    		IF(SUM(bd2.atbat) = 0, 0, SUM(bd2.Hit) / SUM(bd2.atbat)) AS b_rolling_avg
    	FROM batter_data AS bd1
    	JOIN batter_data AS bd2
    	ON bd1.batter = bd2.batter
    	AND bd2.local_date >= DATE_ADD(bd1.local_date, -100)
    	AND bd2.local_date < bd1.local_date
    	GROUP BY bd1.batter, bd1.local_date
        ORDER BY bd1.batter, bd1.local_date
    """

    # Create temporary view for self join
    df.createOrReplaceTempView("batter_data")
    df.persist(StorageLevel.DISK_ONLY)
    # Calculate rolling average using the sql
    batter_rolling_avg = spark_session.sql(sql)

    return batter_rolling_avg


def main():

    # Create Spark Session
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    # Get data from mariadb and save it in a dataframe
    batter_data = get_data(spark)
    # Load Transformer
    t = RollingAvgTransformer()
    # Pass dataframe into Transformer to calculate rolling average
    rolling_average = t._transform(spark, batter_data)
    # Show rolling average of players
    rolling_average.show(25)


if __name__ == "__main__":
    sys.exit(main())
