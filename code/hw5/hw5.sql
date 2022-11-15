USE baseball;

# Take a look at existing tables columns
# SELECT *
# FROM pitcher_counts
# ORDER BY game_id;
#
# SELECT *
# FROM game
# ORDER BY game_id;
#
# SELECT *
# FROM pitcher_counts;
#
# SELECT * FROM boxscore;


# Create Temporary table for pitchers
DROP TEMPORARY TABLE IF EXISTS pitcher_temp;
CREATE TEMPORARY TABLE pitcher_temp AS
SELECT g.game_id,
       g.local_date,
       pc.team_id,
       pc.pitcher,
       pc.startingInning,
       pc.endingInning,
       pc.outsPlayed / 3 as inningsPitched,
       pc.Hit,
       pc.Strikeout,
       pc.atBat,
       pc.toBase,
       pc.Walk,
       pc.plateApperance,
       pc.Home_Run,
       pc.Hit_By_Pitch,
       pc.Sac_Bunt,
       pc.Sac_Fly,
       pc.Fan_interference + pc.Batter_Interference + pc.Catcher_Interference as Interference,
       pc.Intent_Walk,
       0.89 * (1.255 * (pc.Hit-pc.Home_Run) + 4 * pc.Home_Run)
           + 0.56 * (pc.Walk + pc.Hit_By_Pitch - pc.Intent_Walk) as pitchersTotalBases,
       pc.pitchesThrown,
       pc.outsPlayed,
       pc.startingPitcher,
       pc.homeTeam,
       pc.awayTeam
FROM pitcher_counts pc
JOIN game g on pc.game_id = g.game_id
ORDER BY pc.game_id, pc.pitcher;

# Take a look at columns
# SELECT * FROM pitcher_temp;

# Using Pitcher's Temporary table create features
# Creating features for pitchers in pitcher_stats_temp
DROP TEMPORARY TABLE IF EXISTS pitcher_stats_temp;
CREATE TEMPORARY TABLE pitcher_stats_temp
(INDEX game_id_ix (game_id), INDEX pitcher_ix (pitcher), INDEX local_date_ix (local_date))
SELECT pt.game_id,
       pt.local_date,
       pt.pitcher,
       pt.startingPitcher,
       pt.homeTeam,
       pt.awayTeam,
       pt.outsPlayed,
       pt.startingInning,
       pt.endingInning,
       pt.inningsPitched,
       pt.plateApperance,
       pt.Walk as basesOnBalls,

       IF(SUM(pt.inningsPitched)=0, 0,
           9 * (SUM(pt.Walk) / SUM(pt.inningsPitched))) as basesOnBalls9,

       pt.Hit,

       IF(SUM(pt.inningsPitched)=0, 0,
           9 * (SUM(pt.Hit) / SUM(pt.inningsPitched))) as hitsAllowed9,

       pt.Home_Run,

       IF(SUM(pt.inningsPitched)=0, 0,
           9 * (SUM(pt.Home_Run) / SUM(pt.inningsPitched))) as homeRuns9,

       pt.Strikeout,

       IF(SUM(pt.inningsPitched)=0, 0,
           9 * (SUM(pt.Strikeout) / SUM(pt.inningsPitched))) as Strikeout9,

       IF(SUM(pt.Walk) = 0, 0,
           SUM(pt.Strikeout) / SUM(pt.Walk)) as strikeoutToWalkRatio,

       pt.Interference,

       IF((SUM(pt.plateApperance)
                   - SUM(pt.toBase)
                   - SUM(pt.Hit_By_Pitch)
                   - SUM(pt.Sac_Bunt)
                   - SUM(pt.Sac_Fly)
                   - SUM(pt.Interference)) = 0, 0,
           SUM(pt.Hit)/(SUM(pt.plateApperance)
                   - SUM(pt.toBase)
                   - SUM(pt.Hit_By_Pitch)
                   - SUM(pt.Sac_Bunt)
                   - SUM(pt.Sac_Fly)
                   - SUM(pt.Interference))) as oppBattingAvg,

       pt.pitchersTotalBases,

       IF(SUM(pt.inningsPitched)=0 OR SUM(pt.plateApperance)=0, 0,
           9 * (((SUM(pt.Hit) + SUM(pt.Walk) + SUM(pt.Hit_By_Pitch)) * SUM(pt.pitchersTotalBases))
                / (SUM(pt.plateApperance) * SUM(pt.inningsPitched)))) as CERA,

       IF(SUM(pt.inningsPitched)=0, 0,
           (SUM(pt.Strikeout) + SUM(pt.Walk)) / SUM(pt.inningsPitched)) as powerFinesseRatio,

       IF(SUM(pt.inningsPitched)=0, 0,
           (SUM(pt.Walk) + SUM(pt.Hit)) / SUM(pt.inningsPitched)) as WHIP,

       IF(SUM(pt.inningsPitched)=0, 0,
           3 + (((13 * SUM(pt.Home_Run)) + (3 * (SUM(pt.Walk) + SUM(pt.Hit_By_Pitch)))
                     - (2 * SUM(pt.Strikeout)))
               / SUM(pt.inningsPitched))) as DICE

FROM pitcher_temp pt
GROUP BY game_id, pitcher
ORDER BY game_id, pitcher;

# Take a look at Pitcher's Stats with new Features
# SELECT * FROM pitcher_stats_temp;

# Create Temporary table for Starting Pitcher
# This table will create new Feature game_started
# If pitcher is Starting Pitcher, it will count how many times he started game before
DROP TEMPORARY TABLE IF EXISTS starting_pitcher;
CREATE TEMPORARY TABLE starting_pitcher
(INDEX game_id_sp_ix (game_id), INDEX pitcher_sp_ix (pitcher), INDEX local_date_sp_ix (local_date))
ENGINE=MEMORY
SELECT pst1.game_id,
       pst1.pitcher,
       pst1.local_date,
       pst1.startingPitcher,
       pst1.homeTeam,
       pst1.awayTeam,
       COUNT(DISTINCT pst2.game_id) as game_started
FROM pitcher_stats_temp pst1
JOIN pitcher_stats_temp pst2
ON pst1.pitcher = pst2.pitcher
AND pst2.local_date < pst1.local_date
WHERE pst2.startingPitcher = 1 and pst1.startingPitcher = 1
GROUP BY pst1.game_id, pst1.pitcher
ORDER BY pst1.game_id, pst1.pitcher;

# SELECT * FROM starting_pitcher
# WHERE pitcher = 408206
# ORDER BY game_id, pitcher;

# SELECT * FROM pitcher_stats_temp
# WHERE pitcher = 446452;

# Combined table for pitchers_stats and starting_pitcher stats
DROP TEMPORARY TABLE IF EXISTS pitchers_combined;
CREATE TEMPORARY TABLE pitchers_combined
(INDEX game_id_cm_ix (game_id), INDEX pitcher_cm_ix (pitcher), INDEX local_date_cm_ix (local_date))
SELECT pst.*, SUM(sp.game_started) as game_started
FROM pitcher_stats_temp pst
LEFT JOIN starting_pitcher sp
ON pst.game_id = sp.game_id
AND pst.pitcher = sp.pitcher
# WHERE pst.pitcher = 408206
GROUP BY pst.game_id, pst.pitcher
ORDER BY pst.game_id desc, pst.pitcher;

# Take a look at columns
# ELECT * FROM pitchers_combined;

# Calculate Average stats of pitchers of last 100 days
DROP TEMPORARY TABLE IF EXISTS  pitcher_rolling_100;
CREATE TEMPORARY TABLE pitcher_rolling_100 ENGINE=MEMORY
SELECT
    pst1.game_id,
    pst1.pitcher,
    AVG(pst2.basesOnBalls9) as basesOnBalls9_100,
    AVG(pst2.hitsAllowed9) as hitsAllowed9_100,
    AVG(pst2.homeRuns9) as homeRuns9_100,
    AVG(pst2.Strikeout9) as Strikeout9_100,
    AVG(pst2.strikeoutToWalkRatio) as strikeoutToWalkRatio_100,
    AVG(pst2.oppBattingAvg) as oppBattingAvg_100,
    AVG(pst2.CERA) as CERA_100,
    AVG(pst2.powerFinesseRatio) as powerFinesseRatio_100,
    AVG(pst2.WHIP) as WHIP_100,
    AVG(pst2.DICE) as DICE_100
FROM pitchers_combined pst1
JOIN pitchers_combined pst2
ON pst1.pitcher = pst2.pitcher
AND pst2.local_date >= DATE_ADD(pst1.local_date, INTERVAL -100 DAY)
AND pst2.local_date < pst1.local_date
GROUP BY pst1.game_id, pst1.pitcher
ORDER BY pst1.game_id desc, pst1.pitcher;

# Take a look at columns
# SELECT * FROM pitcher_rolling_100;

# Create index on rolling avg table
CREATE INDEX game_id_100_ix ON pitcher_rolling_100 (game_id);
CREATE INDEX pitcher_100_ix ON pitcher_rolling_100 (pitcher);

# Create final stats table for pitchers
DROP TABLE IF EXISTS pitchers_stats_calc;
CREATE TABLE pitchers_stats_calc
(INDEX game_id_sc_ix (game_id), INDEX pitcher_sc_ix (pitcher), INDEX local_date_sc_ix (local_date))
SELECT pst.*,
       pr100.basesOnBalls9_100,
       pr100.hitsAllowed9_100,
       pr100.homeRuns9_100,
       pr100.Strikeout9_100,
       pr100.strikeoutToWalkRatio_100,
       pr100.oppBattingAvg_100,
       pr100.CERA_100,
       pr100.powerFinesseRatio_100,
       pr100.WHIP_100,
       pr100.DICE_100
FROM pitchers_combined pst
LEFT JOIN pitcher_rolling_100 pr100
ON pst.game_id = pr100.game_id
AND pst.pitcher = pr100.pitcher
ORDER BY pst.game_id, pst.pitcher;

# Take a look at columns
# SELECT * FROM pitchers_stats_calc
# ORDER BY game_id desc;










