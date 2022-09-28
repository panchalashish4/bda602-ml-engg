-- Use baseball database
USE baseball;

-- Check if table exists
DROP TABLE IF EXISTS batter_data;
-- Creating temporary table to get batter stats
CREATE TEMPORARY TABLE batter_data
    SELECT
           g.game_id,
           bc.batter,
           date(g.local_date) AS game_date,
           bc.atBat AS atbat,
           bc.Hit AS hit
    FROM batter_counts bc
    JOIN game g ON bc.game_id = g.game_id;

-- Check if table exists
DROP TABLE IF EXISTS batter_hist_avg;
-- Create table for batter historical average
CREATE TABLE batter_hist_avg
    SELECT
        batter,
        IF(SUM(atbat) = 0, 0, SUM(hit) / SUM(atbat)) AS b_avg_hist
    FROM batter_data
    GROUP BY batter
    ORDER BY batter;

-- Check if table exists
DROP TABLE IF EXISTS batter_annual_avg;
-- Create table for batter annual average
CREATE TABLE batter_annual_avg
    SELECT
        batter,
        YEAR(game_date) AS year,
        IF(SUM(atbat) = 0, 0, SUM(hit) / SUM(atbat)) AS b_avg_annual
    FROM batter_data
    GROUP BY batter, year
    ORDER BY batter, year;

-- Check if table exists
DROP TABLE IF EXISTS batterData_byGameDate;
-- Create table for batter data by each date
CREATE TEMPORARY TABLE batterData_byGameDate
(INDEX game_id_ix (game_id), INDEX batter_ix (batter), INDEX game_date_ix (game_date))
    SELECT
           game_id,
           batter,
           game_date,
           SUM(atBat) AS atbat,
           SUM(Hit) AS hit
    FROM batter_data
    GROUP BY batter, game_date;

-- Check if table exists
DROP TABLE IF EXISTS batter_rolling_avg;
-- Create table for batter rolling average for 100 days
CREATE TABLE batter_rolling_avg
	SELECT
		bdg1.batter,
		bdg1.game_date,
		IF(SUM(bdg2.atbat) = 0, 0, SUM(bdg2.Hit) / SUM(bdg2.atbat)) AS b_rolling_avg
	FROM batterData_byGameDate AS bdg1
	JOIN batterData_byGameDate AS bdg2
	ON bdg1.batter = bdg2.batter
	AND bdg2.game_date >= DATE_ADD(bdg1.game_date, INTERVAL -100 DAY)
	AND bdg2.game_date < bdg1.game_date
	GROUP BY bdg1.batter, bdg1.game_date
    ORDER BY bdg1.batter, bdg1.game_date;

-- Drop temporary table
DROP TABLE batter_data;
DROP TABLE batterData_byGameDate;

-- Take a look at data
SELECT * FROM batter_annual_avg LIMIT 0,20;
SELECT * FROM batter_hist_avg LIMIT 0,20;
SELECT * FROM batter_rolling_avg LIMIT 0,20;

