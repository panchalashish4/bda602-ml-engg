USE baseball;

DROP TEMPORARY TABLE IF EXISTS pitcher_temp;
CREATE TEMPORARY TABLE pitcher_temp
(INDEX game_id_pt_ix (game_id), INDEX team_id_pt_ix (team_id), INDEX local_date_pt_ix (game_date))
SELECT pc.*,
       0.89 * (1.255 * (pc.Hit-pc.Home_Run) + 4 * pc.Home_Run)
           + 0.56 * (pc.Walk + pc.Hit_By_Pitch - pc.Intent_Walk) as pitchersTotalBases,
       pc.Fan_interference + pc.Batter_Interference + pc.Catcher_Interference as Interference,
       g.local_date AS game_date,
       tbc.inning
FROM team_pitching_counts pc
JOIN team_batting_counts tbc
ON tbc.game_id = pc.game_id
AND tbc.team_id = pc.team_id
JOIN game g
ON pc.game_id = g.game_id
ORDER BY pc.game_id, pc.team_id;

# SELECT * FROM pitcher_temp
# WHERE game_id=276;

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

       IF(SUM(pt.atbat) = 0, 0, SUM(pt.hit) / SUM(pt.atbat)) AS b_avg

FROM pitcher_temp pt1
JOIN pitcher_temp pt
ON pt1.team_id = pt.team_id
AND pt.game_date < pt1.game_date
GROUP BY pt1.game_id, pt1.team_id
ORDER BY pt1.game_id, pt1.team_id;

# SELECT * FROM pitcher_stats_temp;
# 5642

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

# SELECT * FROM starting_pitcher_temp
# WHERE startingPitcher=1
# AND completeGame=1;
# WHERE pitcher = 424324;

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

# SELECT * FROM starting_pitcher_stats_temp
# ORDER BY game_id desc;

DROP TEMPORARY TABLE IF EXISTS pitchers_combined;
CREATE TEMPORARY TABLE pitchers_combined
(INDEX game_id_cm_ix (game_id), INDEX team_id_cm_ix (team_id), INDEX local_date_cm_ix (game_date))
ENGINE=MEMORY
SELECT  tpc.*,
        g.local_date as game_date,
        tbc.inning,
        IF(tbc.homeTeam=1, tbc.finalScore, 0) as Home_Team_Score,
        IF(tbc.finalScore, tbc.opponent_finalScore, 0) as Away_Team_Score,
        pst.b_avg,
        pst.basesOnBalls9 as basesOnBalls9,
        pst.hitsAllowed9 as hitsAllowed9,
        pst.homeRuns9 as homeRuns9,
        pst.Strikeout9 as Strikeout9,
        pst.strikeoutToWalkRatio as strikeoutToWalkRatio,
        pst.oppBattingAvg as oppBattingAvg,
        pst.CERA as CERA,
        pst.powerFinesseRatio as powerFinesseRatio,
        pst.WHIP as WHIP,
        pst.DICE as DICE,
        spst.sp_innings_pitched,
        spst.sp_games_started,
        spst.sp_complete_games,
        spst.sp_basesOnBalls9
FROM team_pitching_counts tpc
JOIN game g on tpc.game_id = g.game_id
JOIN team_batting_counts tbc on g.game_id = tbc.game_id
LEFT JOIN pitcher_stats_temp pst
ON tpc.game_id = pst.game_id AND tpc.team_id = pst.team_id
LEFT JOIN starting_pitcher_stats_temp spst
ON pst.game_id = spst.game_id AND pst.team_id = spst.team_id
GROUP BY tpc.game_id, tpc.team_id
ORDER BY tpc.game_id, tpc.team_id;

# SELECT * FROM pitchers_combined
# ORDER BY game_id;

# SELECT * FROM team_pitching_counts;
# SELECT * FROM team_batting_counts;


# Create final stats table for pitchers
DROP TABLE IF EXISTS pitchers_stats_calc;
CREATE TABLE pitchers_stats_calc
SELECT
        h.game_id,
        h.team_id as h_team,
        a.team_id as a_team,
        h.game_date,
        h.b_avg as h_bat_avg,
        a.b_avg as a_bat_avg,
        h.basesOnBalls9 as h_basesOnBalls9,
        a.basesOnBalls9 as a_basesOnBalls9,
        h.hitsAllowed9 as h_hitsAllowed9,
        a.hitsAllowed9 as a_hitsAllowed9,
        h.homeRuns9 as h_homeRuns9,
        a.homeRuns9 as a_homeRuns9,
        h.Strikeout9 as h_Strikeout9,
        a.Strikeout9 as a_Strikeout9,
        h.strikeoutToWalkRatio as h_strikeoutToWalkRatio,
        a.strikeoutToWalkRatio as a_strikeoutToWalkRatio,
        h.oppBattingAvg as h_oppBattingAvg,
        a.oppBattingAvg as a_oppBattingAvg,
        h.CERA as h_CERA,
        a.CERA as a_CERA,
        h.powerFinesseRatio as h_powerFinesseRatio,
        a.powerFinesseRatio as a_powerFinesseRatio,
        h.WHIP as h_WHIP,
        a.WHIP as a_WHIP,
        h.DICE as h_DICE,
        a.DICE as a_DICE,
        h.sp_innings_pitched as h_sp_innings_pitched,
        a.sp_innings_pitched as a_sp_innings_pitched,
        h.sp_games_started as h_sp_games_started,
        a.sp_games_started as a_sp_games_started,
        h.sp_complete_games as h_sp_complete_games,
        a.sp_complete_games as a_sp_complete_games,
        h.sp_basesOnBalls9 as h_sp_basesOnBalls9,
        a.sp_basesOnBalls9 as a_sp_basesOnBalls9,
        h.win as Home_Team_Wins
FROM pitchers_combined h
JOIN pitchers_combined a
ON h.game_id = a.game_id AND a.awayTeam = 1
WHERE h.homeTeam = 1
ORDER BY h.game_id, h.team_id;


SELECT * FROM pitchers_stats_calc
ORDER BY game_id desc;

# SELECT * FROM boxscore
# WHERE game_id = 356069;


