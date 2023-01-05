USE baseball;

# SELECT * FROM boxscore ORDER BY game_id;
# SELECT * FROM team_results order by game_id;

# Create temporary table from team_pitching_counts
DROP TEMPORARY TABLE IF EXISTS pitcher_temp;
CREATE TEMPORARY TABLE pitcher_temp
(INDEX game_id_pt_ix (game_id), INDEX team_id_pt_ix (team_id), INDEX local_date_pt_ix (game_date))
SELECT pc.*,
       0.89 * (1.255 * (pc.Hit-pc.Home_Run) + 4 * pc.Home_Run)
           + 0.56 * (pc.Walk + pc.Hit_By_Pitch - pc.Intent_Walk) as pitchersTotalBases,
       pc.Fan_interference + pc.Batter_Interference + pc.Catcher_Interference as Interference,
       g.local_date AS game_date,
       tbc.inning,
       IF(pc.win=0, 1, 0) as lose,
       tbc.finalScore as runs_scored,
       tbc.opponent_finalScore as runs_allowed
FROM team_pitching_counts pc
JOIN team_batting_counts tbc
ON tbc.game_id = pc.game_id
AND tbc.team_id = pc.team_id
JOIN game g
ON pc.game_id = g.game_id
ORDER BY pc.game_id, pc.team_id;

# SELECT * FROM pitcher_temp;

# Create features using pitcher_temp table
DROP TEMPORARY TABLE IF EXISTS pitcher_stats_temp;
CREATE TEMPORARY TABLE pitcher_stats_temp
(INDEX game_id_pst_ix (game_id), INDEX team_id_pst_ix (team_id), INDEX game_date_pst_ix (game_date))
ENGINE=MEMORY
SELECT pt1.*,
       IF(SUM(pt.inning)=0, 0,
           9 * (SUM(pt.Walk) / SUM(pt.inning))) as basesOnBalls9,

       IF(SUM(pt.inning)=0, 0,
           9 * (SUM(pt.Hit) / SUM(pt.inning))) as hitsAllowed9,

       IF(SUM(pt.inning)=0, 0,
           9 * (SUM(pt.Home_Run) / SUM(pt.inning))) as homeRuns9,

       IF(SUM(pt.inning)=0, 0,
           9 * (SUM(pt.Strikeout) / SUM(pt.inning))) as Strikeout9,

       IF(SUM(pt.Walk) = 0, 0,
           SUM(pt.Strikeout) / SUM(pt.Walk)) as strikeoutToWalkRatio,

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

       IF(SUM(pt.inning)=0 OR SUM(pt.plateApperance)=0, 0,
           9 * (((SUM(pt.Hit) + SUM(pt.Walk) + SUM(pt.Hit_By_Pitch)) * SUM(pt.pitchersTotalBases))
                / (SUM(pt.plateApperance) * SUM(pt.inning)))) as CERA,

       IF(SUM(pt.inning)=0, 0,
           (SUM(pt.Strikeout) + SUM(pt.Walk)) / SUM(pt.inning)) as powerFinesseRatio,

       IF(SUM(pt.inning)=0, 0,
           (SUM(pt.Walk) + SUM(pt.Hit)) / SUM(pt.inning)) as WHIP,

       IF(SUM(pt.inning)=0, 0,
           3 + (((13 * SUM(pt.Home_Run)) + (3 * (SUM(pt.Walk) + SUM(pt.Hit_By_Pitch)))
                     - (2 * SUM(pt.Strikeout)))
               / SUM(pt.inning))) as DICE,

       IF(SUM(pt.win)+SUM(pt.lose) = 0, 1000,
           (1000 + (400 * (SUM(pt.win) - SUM(pt.lose))) / (SUM(pt.win)+SUM(pt.lose)))) as performance_ratio,

       IF(SUM(pt.runs_allowed) / SUM(pt.runs_scored) = 0, 0,
           1 / (POWER((1 + SUM(pt.runs_allowed) / SUM(pt.runs_scored)), 2))) as pyth_exp

FROM pitcher_temp pt1
JOIN pitcher_temp pt
ON pt1.team_id = pt.team_id
AND pt.game_date < pt1.game_date
GROUP BY pt1.game_id, pt1.team_id
ORDER BY pt1.game_id, pt1.team_id;

# Create features using pitcher_temp table for the last 100 days
DROP TEMPORARY TABLE IF EXISTS pitcher_stats_ra100_temp;
CREATE TEMPORARY TABLE pitcher_stats_ra100_temp
(INDEX game_id_pst100_ix (game_id), INDEX team_id_pst100_ix (team_id), INDEX game_date_pst100_ix (game_date))
ENGINE=MEMORY
SELECT pt1.*,
       IF(SUM(pt.inning)=0, 0,
           9 * (SUM(pt.Walk) / SUM(pt.inning))) as basesOnBalls9_100,

       IF(SUM(pt.inning)=0, 0,
           9 * (SUM(pt.Hit) / SUM(pt.inning))) as hitsAllowed9_100,

       IF(SUM(pt.inning)=0, 0,
           9 * (SUM(pt.Home_Run) / SUM(pt.inning))) as homeRuns9_100,

       IF(SUM(pt.inning)=0, 0,
           9 * (SUM(pt.Strikeout) / SUM(pt.inning))) as Strikeout9_100,

       IF(SUM(pt.Walk) = 0, 0,
           SUM(pt.Strikeout) / SUM(pt.Walk)) as strikeoutToWalkRatio_100,

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
                   - SUM(pt.Interference))) as oppBattingAvg_100,

       IF(SUM(pt.inning)=0 OR SUM(pt.plateApperance)=0, 0,
           9 * (((SUM(pt.Hit) + SUM(pt.Walk) + SUM(pt.Hit_By_Pitch)) * SUM(pt.pitchersTotalBases))
                / (SUM(pt.plateApperance) * SUM(pt.inning)))) as CERA_100,

       IF(SUM(pt.inning)=0, 0,
           (SUM(pt.Strikeout) + SUM(pt.Walk)) / SUM(pt.inning)) as powerFinesseRatio_100,

       IF(SUM(pt.inning)=0, 0,
           (SUM(pt.Walk) + SUM(pt.Hit)) / SUM(pt.inning)) as WHIP_100,

       IF(SUM(pt.inning)=0, 0,
           3 + (((13 * SUM(pt.Home_Run)) + (3 * (SUM(pt.Walk) + SUM(pt.Hit_By_Pitch)))
                     - (2 * SUM(pt.Strikeout)))
               / SUM(pt.inning))) as DICE_100,

       IF(SUM(pt.win)+SUM(pt.lose) = 0, 1000,
           (1000 + (400 * (SUM(pt.win) - SUM(pt.lose))) / (SUM(pt.win)+SUM(pt.lose)))) as performance_ratio_100,

       IF(SUM(pt.runs_allowed) / SUM(pt.runs_scored) = 0, 0,
       1 / (POWER((1 + SUM(pt.runs_allowed) / SUM(pt.runs_scored)), 2))) as pyth_exp_100

FROM pitcher_temp pt1
JOIN pitcher_temp pt
ON pt1.team_id = pt.team_id
AND pt.game_date >= DATE_ADD(pt1.game_date, INTERVAL -100 DAY)
AND pt.game_date < pt1.game_date
GROUP BY pt1.game_id, pt1.team_id
ORDER BY pt1.game_id, pt1.team_id;

# Combine career and last 100 days stats
DROP TEMPORARY TABLE IF EXISTS pitcher_stats_100_combine_temp;
CREATE TEMPORARY TABLE pitcher_stats_100_combine_temp
(INDEX game_id_pst_100_ix (game_id), INDEX team_id_pst_100_ix (team_id))
SELECT  pst.*,
        pst100.basesOnBalls9_100,
        pst100.hitsAllowed9_100,
        pst100.homeRuns9_100,
        pst100.strikeout9_100,
        pst100.strikeoutToWalkRatio_100,
        pst100.oppBattingAvg_100,
        pst100.CERA_100,
        pst100.powerFinesseRatio_100,
        pst100.WHIP_100,
        pst100.DICE_100,
        pst100.performance_ratio_100,
        pst100.pyth_exp_100
FROM pitcher_stats_temp pst
JOIN pitcher_stats_ra100_temp pst100
ON pst.game_id = pst100.game_id
AND pst.team_id = pst100.team_id
GROUP BY pst.game_id, pst.team_id
ORDER BY pst.game_id, pst.team_id;

# Create Temporary Table for Starting Pitchers
DROP TEMPORARY TABLE IF EXISTS starting_pitcher_temp;
CREATE TEMPORARY TABLE starting_pitcher_temp
(INDEX game_id_spt_ix (game_id), INDEX pitcher_spt_ix (pitcher), INDEX local_date_spt_ix (local_date))
SELECT pc.*,
       g.local_date,
       tbc.inning,
       pc.outsPlayed/3 as innings_pitched,
       IF(tbc.inning=pc.endingInning, 1, 0) as completeGame
FROM pitcher_counts pc
JOIN game g on pc.game_id = g.game_id
JOIN team_batting_counts tbc on g.game_id = tbc.game_id
GROUP BY pc.game_id, pc.pitcher
ORDER BY pc.game_id, pc.pitcher;

# Create Starting Pitcher Stats using starting_pitcher_temp table
DROP TEMPORARY TABLE IF EXISTS starting_pitcher_stats_temp;
CREATE TEMPORARY TABLE starting_pitcher_stats_temp
(INDEX game_id_spst_ix (game_id), INDEX team_id_spst_ix (team_id), INDEX local_date_spst_ix (local_date))
ENGINE=MEMORY
SELECT pst1.game_id,
       pst1.pitcher,
       pst1.team_id,
       pst1.local_date,
       pst1.startingPitcher,
       pst1.endingInning,
       pst1.homeTeam,
       pst1.awayTeam,
       pst1.outsPlayed,
       SUM(pst2.innings_pitched) as sp_innings_pitched,
       COUNT(DISTINCT pst2.game_id) as sp_games_started,
       SUM(pst2.completeGame) as sp_complete_games,
       IF(SUM(pst2.innings_pitched)=0, 0,
           9 * (SUM(pst2.Walk) / SUM(pst2.innings_pitched))) as sp_basesOnBalls9
FROM starting_pitcher_temp pst1
JOIN starting_pitcher_temp pst2
ON pst1.pitcher = pst2.pitcher
AND pst2.local_date < pst1.local_date
WHERE pst2.startingPitcher = 1 and pst1.startingPitcher = 1
GROUP BY pst1.game_id, pst1.pitcher
ORDER BY pst1.game_id, pst1.pitcher;

# Create Starting Pitcher Stats for last 100 days using starting_pitcher_temp table
DROP TEMPORARY TABLE IF EXISTS starting_pitcher_stats_100_temp;
CREATE TEMPORARY TABLE starting_pitcher_stats_100_temp
(INDEX game_id_spst_100_ix (game_id), INDEX team_id_spst_100_ix (team_id), INDEX local_date_spst_100_ix (local_date))
ENGINE=MEMORY
SELECT pst1.game_id,
       pst1.pitcher,
       pst1.team_id,
       pst1.local_date,
       pst1.startingPitcher,
       pst1.endingInning,
       pst1.homeTeam,
       pst1.awayTeam,
       pst1.outsPlayed,
       SUM(pst2.innings_pitched) as sp_innings_pitched_100,
       COUNT(DISTINCT pst2.game_id) as sp_games_started_100,
       SUM(pst2.completeGame) as sp_complete_games_100,
       IF(SUM(pst2.innings_pitched)=0, 0,
           9 * (SUM(pst2.Walk) / SUM(pst2.innings_pitched))) as sp_basesOnBalls9_100
FROM starting_pitcher_temp pst1
JOIN starting_pitcher_temp pst2
ON pst1.pitcher = pst2.pitcher
AND pst2.local_date >= DATE_ADD(pst1.local_date, INTERVAL -100 DAY)
AND pst2.local_date < pst1.local_date
WHERE pst2.startingPitcher = 1 and pst1.startingPitcher = 1
GROUP BY pst1.game_id, pst1.pitcher
ORDER BY pst1.game_id, pst1.pitcher;

# Combine career and last 100 days stats for the Starting Pitcher
DROP TEMPORARY TABLE IF EXISTS starting_pitcher_stats_100_combine_temp;
CREATE TEMPORARY TABLE starting_pitcher_stats_100_combine_temp
(INDEX game_id_spst100_ix (game_id), INDEX team_id_spst100_ix (team_id))
SELECT spst.game_id,
       spst.team_id,
       spst.sp_innings_pitched,
       spst100.sp_innings_pitched_100,
       spst.sp_games_started,
       spst100.sp_games_started_100,
       spst.sp_complete_games,
       spst100.sp_complete_games_100,
       spst.sp_basesOnBalls9,
       spst100.sp_basesOnBalls9_100
FROM starting_pitcher_stats_temp spst
JOIN starting_pitcher_stats_100_temp spst100
ON spst.game_id = spst100.game_id
AND spst.team_id = spst100.team_id
GROUP BY spst.game_id, spst.team_id
ORDER BY spst.game_id, spst.team_id;

# Combine career and last 100 days features in one table
DROP TEMPORARY TABLE IF EXISTS pitchers_combined;
CREATE TEMPORARY TABLE pitchers_combined
(INDEX game_id_cm_ix (game_id), INDEX team_id_cm_ix (team_id), INDEX local_date_cm_ix (game_date))
SELECT  tpc.*,
        g.local_date as game_date,
        tbc.inning,
        IF(tbc.homeTeam=1, tbc.finalScore, 0) as Home_Team_Score,
        IF(tbc.finalScore, tbc.opponent_finalScore, 0) as Away_Team_Score,
        pst.basesOnBalls9,
        pst.basesOnBalls9_100,
        pst.hitsAllowed9,
        pst.hitsAllowed9_100,
        pst.homeRuns9,
        pst.homeRuns9_100,
        pst.Strikeout9,
        pst.strikeout9_100,
        pst.strikeoutToWalkRatio,
        pst.strikeoutToWalkRatio_100,
        pst.oppBattingAvg,
        pst.oppBattingAvg_100,
        pst.CERA,
        pst.CERA_100,
        pst.powerFinesseRatio,
        pst.powerFinesseRatio_100,
        pst.WHIP,
        pst.WHIP_100,
        pst.DICE,
        pst.DICE_100,
        spst.sp_innings_pitched,
        spst.sp_innings_pitched_100,
        spst.sp_games_started,
        spst.sp_games_started_100,
        spst.sp_complete_games,
        spst.sp_complete_games_100,
        spst.sp_basesOnBalls9,
        spst.sp_basesOnBalls9_100,
        pst.performance_ratio,
        pst.performance_ratio_100,
        pst.pyth_exp,
        pst.pyth_exp_100
FROM team_pitching_counts tpc
JOIN game g on tpc.game_id = g.game_id
JOIN team_batting_counts tbc on g.game_id = tbc.game_id
LEFT JOIN pitcher_stats_100_combine_temp pst
ON tpc.game_id = pst.game_id AND tpc.team_id = pst.team_id
LEFT JOIN starting_pitcher_stats_100_combine_temp spst
ON pst.game_id = spst.game_id AND pst.team_id = spst.team_id
GROUP BY tpc.game_id, tpc.team_id
ORDER BY tpc.game_id, tpc.team_id;

# Create final stats table for teams based on their pitching features
DROP TABLE IF EXISTS pitchers_stats_calc;
CREATE TABLE pitchers_stats_calc
(INDEX game_id_psc_ix (game_id), INDEX hteam_id_psc_ix (h_team), INDEX ateam_id_psc_ix (a_team), INDEX local_date_psc_ix (game_date))
SELECT
        h.game_id,
        h.team_id as h_team,
        a.team_id as a_team,
        h.game_date,
        h.basesOnBalls9 as h_basesOnBalls9,
        a.basesOnBalls9 as a_basesOnBalls9,
        h.basesOnBalls9_100 as h_basesOnBalls9_100,
        a.basesOnBalls9_100 as a_basesOnBalls9_100,
        h.hitsAllowed9 as h_hitsAllowed9,
        a.hitsAllowed9 as a_hitsAllowed9,
        h.hitsAllowed9_100 as h_hitsAllowed9_100,
        a.hitsAllowed9_100 as a_hitsAllowed9_100,
        h.homeRuns9 as h_homeRuns9,
        a.homeRuns9 as a_homeRuns9,
        h.homeRuns9_100 as h_homeRuns9_100,
        a.homeRuns9_100 as a_homeRuns9_100,
        h.Strikeout9 as h_Strikeout9,
        a.Strikeout9 as a_Strikeout9,
        h.strikeout9_100 as h_strikeout9_100,
        a.strikeout9_100 as a_strikeout9_100,
        h.strikeoutToWalkRatio as h_strikeoutToWalkRatio,
        a.strikeoutToWalkRatio as a_strikeoutToWalkRatio,
        h.strikeoutToWalkRatio_100 as h_strikeoutToWalkRatio_100,
        a.strikeoutToWalkRatio_100 as a_strikeoutToWalkRatio_100,
        h.oppBattingAvg as h_oppBattingAvg,
        a.oppBattingAvg as a_oppBattingAvg,
        h.oppBattingAvg_100 as h_oppBattingAvg_100,
        a.oppBattingAvg_100 as a_oppBattingAvg_100,
        h.CERA as h_CERA,
        a.CERA as a_CERA,
        h.CERA_100 as h_CERA_100,
        a.CERA_100 as a_CERA_100,
        h.powerFinesseRatio as h_powerFinesseRatio,
        a.powerFinesseRatio as a_powerFinesseRatio,
        h.powerFinesseRatio_100 as h_powerFinesseRatio_100,
        a.powerFinesseRatio_100 as a_powerFinesseRatio_100,
        h.WHIP as h_WHIP,
        a.WHIP as a_WHIP,
        h.WHIP_100 as h_WHIP_100,
        a.WHIP_100 as a_WHIP_100,
        h.DICE as h_DICE,
        a.DICE as a_DICE,
        h.DICE_100 as h_DICE_100,
        a.DICE_100 as a_DICE_100,
        h.sp_innings_pitched as h_sp_innings_pitched,
        a.sp_innings_pitched as a_sp_innings_pitched,
        h.sp_innings_pitched_100 as h_sp_innings_pitched_100,
        a.sp_innings_pitched_100 as a_sp_innings_pitched_100,
        h.sp_games_started as h_sp_games_started,
        a.sp_games_started as a_sp_games_started,
        h.sp_games_started_100 as h_sp_games_started_100,
        a.sp_games_started_100 as a_sp_games_started_100,
        h.sp_complete_games as h_sp_complete_games,
        a.sp_complete_games as a_sp_complete_games,
        h.sp_complete_games_100 as h_sp_complete_games_100,
        a.sp_complete_games_100 as a_sp_complete_games_100,
        h.sp_basesOnBalls9 as h_sp_basesOnBalls9,
        a.sp_basesOnBalls9 as a_sp_basesOnBalls9,
        h.sp_basesOnBalls9_100 as h_sp_basesOnBalls9_100,
        a.sp_basesOnBalls9_100 as a_sp_basesOnBalls9_100,
        h.performance_ratio as h_performance_ratio,
        a.performance_ratio as a_performance_ratio,
        h.performance_ratio_100 as h_performance_ratio_100,
        a.performance_ratio_100 as a_performace_ratio_100,
        h.pyth_exp as h_pyth_exp,
        a.pyth_exp as a_pyth_exp,
        h.pyth_exp_100 as h_pyth_exp_100,
        a.pyth_exp_100 as a_pyth_exp_100,
        h.win as Home_Team_Wins
FROM pitchers_combined h
JOIN pitchers_combined a
ON h.game_id = a.game_id AND a.awayTeam = 1
WHERE h.homeTeam = 1
GROUP BY h.game_id, h.team_id
ORDER BY h.game_id, h.team_id;

# # Check Table
# SELECT * FROM pitchers_stats_calc
# ORDER BY game_id desc;

#### Boxscore Table Analysis ####
## Values in opponent_finalScore are incorrect ##

# SELECT pst.*,
#        b.winner_home_or_away,
#        tbc.homeTeam,
#        tbc.awayTeam,
#        tbc.finalScore,
#        tbc.opponent_finalScore
# FROM pitchers_stats_calc pst
# JOIN boxscore b on pst.game_id = b.game_id
# JOIN team_batting_counts tbc on pst.game_id = tbc.game_id
# WHERE tbc.homeTeam=1
# order by pst.game_id;
#
# SELECT * FROM boxscore
# WHERE game_id IN (333417, 325875);
#
# SELECT game_id, home_runs, away_runs, winner_home_or_away
# FROM boxscore
# WHERE game_id IN (333417, 325875)
# ORDER BY game_id;
#
#
# SELECT * FROM team_pitching_counts
# WHERE game_id=333417 and homeTeam=1;
#
# SELECT * FROM team_batting_counts
# WHERE game_id=333417 and homeTeam=1;


DROP TABLE IF EXISTS pitchers_diff_calc;
CREATE TABLE pitchers_diff_calc
SELECT
        game_id as game_id,
        h_team as h_team,
        a_team as a_team,
        game_date as game_date,
        IF(SUM(h_basesOnBalls9) = 0 , 0,
            ((SUM(a_basesOnBalls9) - SUM(h_basesOnBalls9)) / SUM(h_basesOnBalls9)) * 100) as rcp_basesOnBalls9,
        SUM(a_basesOnBalls9) - SUM(h_basesOnBalls9) as d_basesOnBalls9,
        IF(SUM(h_basesOnBalls9)=0, 0,
            SUM(a_basesOnBalls9) / SUM(h_basesOnBalls9)) as r_basesOnBalls9,
        IF(SUM(h_basesOnBalls9_100) = 0, 0,
            ((SUM(a_basesOnBalls9_100) - SUM(h_basesOnBalls9_100)) / SUM(h_basesOnBalls9_100)) * 100) as rcp_basesOnBalls9_100,
        SUM(a_basesOnBalls9_100) - SUM(h_basesOnBalls9_100) as d_basesOnBalls9_100,
        IF(SUM(h_basesOnBalls9_100)=0, 0,
            SUM(a_basesOnBalls9_100) / SUM(h_basesOnBalls9_100)) as r_basesOnBalls9_100,
        IF(SUM(h_hitsAllowed9) = 0, 0,
            ((SUM(a_hitsAllowed9) - SUM(h_hitsAllowed9)) / SUM(h_hitsAllowed9)) * 100) as rcp_hitsAllowed9,
        SUM(a_hitsAllowed9) - SUM(h_hitsAllowed9) as d_hitsAllowed9,
        IF(SUM(h_hitsAllowed9)=0, 0,
            SUM(a_hitsAllowed9) / SUM(h_hitsAllowed9)) as r_hitsAllowed9,
        IF(SUM(h_hitsAllowed9_100) = 0, 0,
            ((SUM(a_hitsAllowed9_100) - SUM(h_hitsAllowed9_100)) / SUM(h_hitsAllowed9_100)) * 100) as rcp_hitsAllowed9_100,
        SUM(a_hitsAllowed9_100) - SUM(h_hitsAllowed9_100) as d_hitsAllowed9_100,
        IF(SUM(h_hitsAllowed9_100)=0, 0,
            SUM(a_hitsAllowed9_100) / SUM(h_hitsAllowed9_100)) as r_hitsAllowed9_100,
        IF(SUM(h_homeRuns9) = 0, 0,
            ((SUM(a_homeRuns9) - SUM(h_homeRuns9)) / SUM(h_homeRuns9)) * 100) as rcp_homeRuns9,
        SUM(a_homeRuns9) - SUM(h_homeRuns9) as d_homeRuns9,
        IF(SUM(h_homeRuns9)=0, 0,
            SUM(a_homeRuns9) / SUM(h_homeRuns9)) as r_homeRuns9,
        IF(SUM(h_homeRuns9_100) = 0, 0,
            ((SUM(a_homeRuns9_100) - SUM(h_homeRuns9_100)) / SUM(h_homeRuns9_100)) * 100) as rcp_homeRuns9_100,
        SUM(a_homeRuns9_100) - SUM(h_homeRuns9_100) as d_homeRuns9_100,
        IF(SUM(h_homeRuns9_100)=0, 0,
            SUM(a_homeRuns9_100) / SUM(h_homeRuns9_100)) as r_homeRuns9_100,
        IF(SUM(h_Strikeout9) = 0, 0,
            ((SUM(a_Strikeout9) - SUM(h_Strikeout9)) / SUM(h_Strikeout9)) * 100) as rcp_Strikeout9,
        SUM(a_Strikeout9) - SUM(h_Strikeout9) as d_Strikeout9,
        IF(SUM(h_Strikeout9)=0, 0,
            SUM(a_Strikeout9) / SUM(h_Strikeout9)) as r_Strikeout9,
        IF(SUM(h_strikeout9_100) = 0, 0,
            ((SUM(a_strikeout9_100) - SUM(h_strikeout9_100)) / SUM(h_strikeout9_100)) * 100) as rcp_strikeout9_100,
        SUM(a_strikeout9_100) - SUM(h_strikeout9_100) as d_strikeout9_100,
        IF(SUM(h_strikeout9_100)=0, 0,
            SUM(a_strikeout9_100) / SUM(h_strikeout9_100)) as r_strikeout9_100,
        IF(SUM(h_strikeoutToWalkRatio) = 0, 0,
          ((SUM(a_strikeoutToWalkRatio) - SUM(h_strikeoutToWalkRatio)) / SUM(h_strikeoutToWalkRatio)) * 100) as rcp_strikeoutToWalkRatio,
        SUM(a_strikeoutToWalkRatio) - SUM(h_strikeoutToWalkRatio) as d_strikeoutToWalkRatio,
        IF(SUM(h_strikeoutToWalkRatio)=0, 0,
            SUM(a_strikeoutToWalkRatio) / SUM(h_strikeoutToWalkRatio)) as r_strikeoutToWalkRatio,
        IF(SUM(h_strikeoutToWalkRatio_100) = 0, 0,
            ((SUM(a_strikeoutToWalkRatio_100) - SUM(h_strikeoutToWalkRatio_100)) / SUM(h_strikeoutToWalkRatio_100)) * 100) as rcp_strikeoutToWalkRatio_100,
        SUM(a_strikeoutToWalkRatio_100) - SUM(h_strikeoutToWalkRatio_100) as d_strikeoutToWalkRatio_100,
        IF(SUM(h_strikeoutToWalkRatio_100)=0, 0,
            SUM(a_strikeoutToWalkRatio_100) / SUM(h_strikeoutToWalkRatio_100)) as r_strikeoutToWalkRatio_100,
        IF(SUM(h_oppBattingAvg) = 0, 0 ,
            ((SUM(a_oppBattingAvg) - SUM(h_oppBattingAvg)) / SUM(h_oppBattingAvg)) * 100) as rcp_oppBattingAvg,
        SUM(a_oppBattingAvg) - SUM(h_oppBattingAvg) as d_oppBattingAvg,
        IF(SUM(h_oppBattingAvg)=0, 0,
            SUM(a_oppBattingAvg) / SUM(h_oppBattingAvg)) as r_oppBattingAvg,
        IF(SUM(h_oppBattingAvg_100) = 0, 0 ,
            ((SUM(a_oppBattingAvg_100) - SUM(h_oppBattingAvg_100)) / SUM(h_oppBattingAvg_100)) * 100) as rcp_oppBattingAvg_100,
        SUM(a_oppBattingAvg_100) - SUM(h_oppBattingAvg_100) as d_oppBattingAvg_100,
        IF(SUM(h_oppBattingAvg_100)=0, 0,
            SUM(a_oppBattingAvg_100) / SUM(h_oppBattingAvg_100)) as r_oppBattingAvg_100,
        IF(SUM(h_CERA) = 0, 0 ,
            ((SUM(a_CERA) - SUM(h_CERA)) / SUM(h_CERA)) * 100) as rcp_CERA,
        SUM(a_CERA) - SUM(h_CERA) as d_CERA,
        IF(SUM(h_CERA)=0, 0,
            SUM(a_CERA) / SUM(h_CERA)) as r_CERA,
        IF(SUM(h_CERA_100) = 0, 0 ,
            ((SUM(a_CERA_100) - SUM(h_CERA_100)) / SUM(h_CERA_100)) * 100) as rcp_CERA_100,
        SUM(a_CERA_100) - SUM(h_CERA_100) as d_CERA_100,
        IF(SUM(h_CERA_100)=0, 0,
            SUM(a_CERA_100) / SUM(h_CERA_100)) as r_CERA_100,
        IF(SUM(h_powerFinesseRatio) = 0, 0 ,
            ((SUM(a_powerFinesseRatio) - SUM(h_powerFinesseRatio)) / SUM(h_powerFinesseRatio)) * 100) as rcp_powerFinesseRatio,
        SUM(a_powerFinesseRatio) - SUM(h_powerFinesseRatio) as d_powerFinesseRatio,
        IF(SUM(h_powerFinesseRatio)=0, 0,
            SUM(a_powerFinesseRatio) / SUM(h_powerFinesseRatio)) as r_powerFinesseRatio,
        IF(SUM(h_powerFinesseRatio_100) = 0, 0 ,
            ((SUM(a_powerFinesseRatio_100) - SUM(h_powerFinesseRatio_100)) / SUM(h_powerFinesseRatio_100)) * 100) as rcp_powerFinesseRatio_100,
        SUM(a_powerFinesseRatio_100) - SUM(h_powerFinesseRatio_100) as d_powerFinesseRatio_100,
        IF(SUM(h_powerFinesseRatio_100)=0, 0,
            SUM(a_powerFinesseRatio_100) / SUM(h_powerFinesseRatio_100)) as r_powerFinesseRatio_100,
        IF(SUM(h_WHIP) = 0, 0 ,
            ((SUM(a_WHIP) - SUM(h_WHIP)) / SUM(h_WHIP)) * 100) as rcp_WHIP,
        SUM(a_WHIP) - SUM(h_WHIP) as d_WHIP,
        IF(SUM(h_WHIP)=0, 0,
            SUM(a_WHIP) / SUM(h_WHIP)) as r_WHIP,
        IF(SUM(h_WHIP_100) = 0, 0 ,
            ((SUM(a_WHIP_100) - SUM(h_WHIP_100)) / SUM(h_WHIP_100)) * 100) as rcp_WHIP_100,
        SUM(a_WHIP_100) - SUM(h_WHIP_100) as d_WHIP_100,
        IF(SUM(h_WHIP_100)=0, 0,
            SUM(a_WHIP_100) / SUM(h_WHIP_100)) as r_WHIP_100,
        IF(SUM(h_DICE) = 0, 0 ,
            ((SUM(a_DICE) - SUM(h_DICE)) / SUM(h_DICE)) * 100) as rcp_DICE,
        SUM(a_DICE) - SUM(h_DICE) as d_DICE,
        IF(SUM(h_DICE)=0, 0,
            SUM(a_DICE) / SUM(h_DICE)) as r_DICE,
        IF(SUM(h_DICE_100) = 0, 0 ,
            ((SUM(a_DICE_100) - SUM(h_DICE_100)) / SUM(h_DICE_100)) * 100) as rcp_DICE_100,
        SUM(a_DICE_100) - SUM(h_DICE_100) as d_DICE_100,
        IF(SUM(h_DICE_100)=0, 0,
            SUM(a_DICE_100) / SUM(h_DICE_100)) as r_DICE_100,
        IF(SUM(h_sp_innings_pitched) = 0, 0 ,
            ((SUM(a_sp_innings_pitched) - SUM(h_sp_innings_pitched)) / SUM(h_sp_innings_pitched)) * 100) as rcp_sp_innings_pitched,
        SUM(a_sp_innings_pitched) - SUM(h_sp_innings_pitched) as d_sp_innings_pitched,
        IF(SUM(h_sp_innings_pitched)=0, 0,
            SUM(a_sp_innings_pitched) / SUM(h_sp_innings_pitched)) as r_sp_innings_pitched,
        IF(SUM(h_sp_innings_pitched_100) = 0, 0 ,
            ((SUM(a_sp_innings_pitched_100) - SUM(h_sp_innings_pitched_100)) / SUM(h_sp_innings_pitched_100)) * 100) as rcp_sp_innings_pitched_100,
        SUM(a_sp_innings_pitched_100) - SUM(h_sp_innings_pitched_100) as d_sp_innings_pitched_100,
        IF(SUM(h_sp_innings_pitched_100)=0, 0,
            SUM(a_sp_innings_pitched_100) / SUM(h_sp_innings_pitched_100)) as r_sp_innings_pitched_100,
        IF(SUM(h_sp_games_started) = 0, 0 ,
            ((SUM(a_sp_games_started) - SUM(h_sp_games_started)) / SUM(h_sp_games_started)) * 100) as rcp_sp_games_started,
        SUM(a_sp_games_started) - SUM(h_sp_games_started) as d_sp_games_started,
        IF(SUM(h_sp_games_started)=0, 0,
            SUM(a_sp_games_started) / SUM(h_sp_games_started)) as r_sp_games_started,
        IF(SUM(h_sp_games_started_100) = 0, 0 ,
            ((SUM(a_sp_games_started_100) - SUM(h_sp_games_started_100)) / SUM(h_sp_games_started_100)) * 100) as rcp_sp_games_started_100,
        SUM(a_sp_games_started_100) - SUM(h_sp_games_started_100) as d_sp_games_started_100,
        IF(SUM(h_sp_games_started_100)=0, 0,
            SUM(a_sp_games_started_100) / SUM(h_sp_games_started_100)) as r_sp_games_started_100,
        IF(SUM(h_sp_complete_games) = 0, 0 ,
            ((SUM(a_sp_complete_games) - SUM(h_sp_complete_games)) / SUM(h_sp_complete_games)) * 100) as rcp_sp_complete_games,
        SUM(a_sp_complete_games) - SUM(h_sp_complete_games) as d_sp_complete_games,
        IF(SUM(h_sp_complete_games)=0, 0,
            SUM(a_sp_complete_games) / SUM(h_sp_complete_games)) as r_sp_complete_games,
        IF(SUM(h_sp_complete_games_100) = 0, 0 ,
            ((SUM(a_sp_complete_games_100) - SUM(h_sp_complete_games_100)) / SUM(h_sp_complete_games_100)) * 100) as rcp_sp_complete_games_100,
        SUM(a_sp_complete_games_100) - SUM(h_sp_complete_games_100) as d_sp_complete_games_100,
        IF(SUM(h_sp_complete_games_100)=0, 0,
            SUM(a_sp_complete_games_100) / SUM(h_sp_complete_games_100)) as r_sp_complete_games_100,
        IF(SUM(h_sp_basesOnBalls9) = 0, 0 ,
            ((SUM(a_sp_basesOnBalls9) - SUM(h_sp_basesOnBalls9)) / SUM(h_sp_basesOnBalls9)) * 100) as rcp_sp_basesOnBalls9,
        SUM(a_sp_basesOnBalls9) - SUM(h_sp_basesOnBalls9) as d_sp_basesOnBalls9,
        IF(SUM(h_sp_basesOnBalls9)=0, 0,
            SUM(a_sp_basesOnBalls9) / SUM(h_sp_basesOnBalls9)) as r_sp_basesOnBalls9,
        IF(SUM(h_sp_basesOnBalls9_100) = 0, 0 ,
            ((SUM(a_sp_basesOnBalls9_100) - SUM(h_sp_basesOnBalls9_100)) / SUM(h_sp_basesOnBalls9_100)) * 100) as rcp_sp_basesOnBalls9_100,
        SUM(a_sp_basesOnBalls9_100) - SUM(h_sp_basesOnBalls9_100) as d_sp_basesOnBalls9_100,
        IF(SUM(h_sp_basesOnBalls9_100)=0, 0,
            SUM(a_sp_basesOnBalls9_100) / SUM(h_sp_basesOnBalls9_100)) as r_sp_basesOnBalls9_100,
        IF(SUM(h_performance_ratio) = 0, 0,
            ((SUM(a_performance_ratio) - SUM(h_performance_ratio)) / SUM(h_performance_ratio)) * 100) as rcp_perf_ratio,
        SUM(a_performance_ratio) - SUM(h_performance_ratio) as d_perf_ratio,
        IF(SUM(h_performance_ratio)=0, 0,
            SUM(a_performance_ratio) / SUM(h_performance_ratio)) as r_perf_ratio,
        IF(SUM(h_performance_ratio_100) = 0, 0,
            ((SUM(a_performace_ratio_100) - SUM(h_performance_ratio_100)) / SUM(h_performance_ratio_100)) * 100) as rcp_perf_ratio_100,
        SUM(a_performace_ratio_100) - SUM(h_performance_ratio_100) as d_perf_ratio_100,
        IF(SUM(h_performance_ratio_100)=0, 0,
            SUM(a_performace_ratio_100) / SUM(h_performance_ratio_100)) as r_perf_ratio_100,
        IF(SUM(h_pyth_exp) = 0, 0,
            ((SUM(a_pyth_exp) - SUM(h_pyth_exp)) / SUM(h_pyth_exp)) * 100) as rcp_pyth_exp,
        SUM(a_pyth_exp) - SUM(h_pyth_exp) as d_pyth_exp,
        IF(SUM(h_pyth_exp)=0, 0,
            SUM(a_pyth_exp) / SUM(h_pyth_exp)) as r_pyth_exp,
        IF(SUM(h_pyth_exp_100) = 0, 0,
            ((SUM(a_pyth_exp_100) - SUM(h_pyth_exp_100)) / SUM(h_pyth_exp_100)) * 100) as rcp_pyth_exp_100,
        SUM(a_pyth_exp_100) - SUM(h_pyth_exp_100) as d_pyth_exp_100,
        IF(SUM(h_pyth_exp_100)=0, 0,
            SUM(a_pyth_exp_100) / SUM(h_pyth_exp_100)) as r_pyth_exp_100,
        Home_Team_Wins as Home_Team_Wins
FROM pitchers_stats_calc
GROUP BY game_id, h_team
ORDER BY game_id, h_team;

# SELECT *
# FROM pitchers_diff_calc
# ORDER BY game_id desc;



# DROP TABLE IF EXISTS pitchers_stats_diff_comb;
# CREATE TABLE pitchers_stats_diff_comb
# (INDEX game_id_psdc_ix (game_id), INDEX hteam_id_psdc_ix (h_team), INDEX ateam_id_psdc_ix (a_team), INDEX local_date_psdc_ix (game_date))
# SELECT psc.*,
#        pdc.*
# FROM pitchers_stats_calc psc
# JOIN pitchers_diff_calc pdc
# ON psc.game_id = pdc.game_id2
# AND psc.h_team = pdc.h_team2
# ORDER BY psc.game_id;

# SELECT *
# FROM pitchers_stats_diff_comb
# ORDER BY game_id desc;
#
# SELECT game_id, home_runs, away_runs, winner_home_or_away
# FROM boxscore
# WHERE game_id IN (333417, 325875)
# ORDER BY game_id;
