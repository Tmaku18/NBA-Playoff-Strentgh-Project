"""Four Factors: eFG%, TOV%, ORB% (Opp DRB from other row in same game), FT rate. No net_rating."""

from __future__ import annotations

import pandas as pd


def four_factors_from_team_logs(tgl: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    tgl: team_game_logs with game_id, team_id, fgm, fga, fg3m, ftm, fta, tov, oreb, pts.
    games: game_id, home_team_id, away_team_id.
    For ORB% we need Opp_DRB. Infer from the other team's row in the same game (DREB).
    eFG% = (FGM + 0.5*FG3M)/FGA; TOV% = TOV/(FGA+0.44*FTA+TOV); FT_rate = FTA/FGA;
    ORB% = OREB/(OREB+Opp_DRB). If Opp_DRB missing, use league-avg proxy or NaN.
    """
    tgl = tgl.copy()
    tgl["FGA"] = tgl["fga"].fillna(0).replace(0, 1)
    tgl["eFG"] = (tgl["fgm"].fillna(0) + 0.5 * tgl["fg3m"].fillna(0)) / tgl["FGA"]
    tgl["TOV_pct"] = tgl["tov"].fillna(0) / (tgl["fga"].fillna(0) + 0.44 * tgl["fta"].fillna(0) + tgl["tov"].fillna(0).replace(0, 1))
    tgl["FT_rate"] = tgl["fta"].fillna(0) / tgl["FGA"]

    # Opp DRB: for each (game_id, team_id) the opponent is the other team; get their dreb
    g = games[["game_id", "home_team_id", "away_team_id"]].drop_duplicates()
    tgl = tgl.merge(g, on="game_id", how="left")
    tgl["opp_team_id"] = tgl.apply(
        lambda r: r["away_team_id"] if r["team_id"] == r["home_team_id"] else r["home_team_id"],
        axis=1,
    )
    opp = tgl[["game_id", "team_id", "dreb"]].rename(columns={"team_id": "opp_team_id", "dreb": "opp_dreb"})
    tgl = tgl.merge(opp, on=["game_id", "opp_team_id"], how="left")
    oreb = tgl["oreb"].fillna(0)
    odrb = tgl["opp_dreb"].fillna(0)
    tgl["ORB_pct"] = oreb / (oreb + odrb).replace(0, 1)

    return tgl[["game_id", "team_id", "eFG", "TOV_pct", "FT_rate", "ORB_pct"]].copy()
