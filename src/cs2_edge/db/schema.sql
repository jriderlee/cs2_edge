-- CS2 Edge — DuckDB schema (Phase 1 scaffold)
-- Goal: detect longshot bias by comparing Kalshi implied probability
-- against realized match outcomes, adjusted for roster changes.

CREATE TABLE IF NOT EXISTS match_results (
    match_id        BIGINT PRIMARY KEY,
    event_name      VARCHAR,
    match_date      DATE,
    team_a          VARCHAR,
    team_b          VARCHAR,
    team_a_score    INTEGER,
    team_b_score    INTEGER,
    winner          VARCHAR,
    best_of         INTEGER,
    map_count       INTEGER,
    hltv_url        VARCHAR,
    ingested_at     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS player_ratings (
    player_id       BIGINT,
    match_id        BIGINT,
    player_name     VARCHAR,
    team            VARCHAR,
    rating_date     DATE,
    rating          DOUBLE,
    maps_played     INTEGER,
    kpr             DOUBLE,
    dpr             DOUBLE,
    impact          DOUBLE,
    PRIMARY KEY (player_id, match_id)
);

CREATE TABLE IF NOT EXISTS kalshi_contracts (
    ticker          VARCHAR PRIMARY KEY,
    match_id        BIGINT,
    series_ticker   VARCHAR,
    market_type     VARCHAR,
    team            VARCHAR,
    yes_bid         DOUBLE,
    yes_ask         DOUBLE,
    no_bid          DOUBLE,
    no_ask          DOUBLE,
    last_price      DOUBLE,
    volume          BIGINT,
    open_interest   BIGINT,
    status          VARCHAR,
    result          VARCHAR,
    created_at      TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ,
    resolved_at     TIMESTAMPTZ,
    event_start     TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS roster_changes (
    change_id       BIGINT PRIMARY KEY,
    player_id       BIGINT,
    player_name     VARCHAR,
    from_team       VARCHAR,
    to_team         VARCHAR,
    change_date     DATE,
    change_type     VARCHAR,
    source_url      VARCHAR,
    ingested_at     TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_match_results_date ON match_results(match_date);
CREATE INDEX IF NOT EXISTS idx_player_ratings_team ON player_ratings(team, rating_date);
CREATE INDEX IF NOT EXISTS idx_player_ratings_match ON player_ratings(match_id);
CREATE INDEX IF NOT EXISTS idx_kalshi_contracts_match ON kalshi_contracts(match_id);
CREATE INDEX IF NOT EXISTS idx_kalshi_contracts_status ON kalshi_contracts(status);
CREATE INDEX IF NOT EXISTS idx_roster_changes_player ON roster_changes(player_id, change_date);
