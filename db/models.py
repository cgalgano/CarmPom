"""
models.py
---------
SQLAlchemy ORM table definitions.

Each class here maps to one table in carmpom.db.
This is the single source of truth for the database schema.

Phase 0 tables: teams, games, box_scores
Future phases will add: carm_pom_ratings, predictions, bracket_simulations
"""

from datetime import date, datetime
from sqlalchemy import (
    Integer, String, Float, Boolean, Date, DateTime,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.database import Base


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

class Team(Base):
    """
    One row per NCAA Division I men's basketball program.
    Populated by pipeline/fetch_teams.py in Phase 1.
    """
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)            # e.g. "Duke Blue Devils"
    short_name: Mapped[str] = mapped_column(String(50), nullable=True)        # e.g. "Duke"
    conference: Mapped[str] = mapped_column(String(50), nullable=True)        # e.g. "ACC"
    espn_id: Mapped[str] = mapped_column(String(20), nullable=True, unique=True)  # ESPN's internal team ID
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships — let SQLAlchemy join these automatically
    home_games: Mapped[list["Game"]] = relationship("Game", foreign_keys="Game.home_team_id", back_populates="home_team")
    away_games: Mapped[list["Game"]] = relationship("Game", foreign_keys="Game.away_team_id", back_populates="away_team")
    box_scores: Mapped[list["BoxScore"]] = relationship("BoxScore", back_populates="team")
    ratings: Mapped[list["CarmPomRating"]] = relationship("CarmPomRating", back_populates="team")

    def __repr__(self) -> str:
        return f"<Team id={self.id} name='{self.name}' conference='{self.conference}'>"


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

class Game(Base):
    """
    One row per completed game.
    home_team_id / away_team_id are NULL for neutral-site games
    where "home" is arbitrary — use neutral_site=True instead.
    """
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    espn_game_id: Mapped[str] = mapped_column(String(20), nullable=True, unique=True)  # ESPN's event ID
    season: Mapped[int] = mapped_column(Integer, nullable=False)              # e.g. 2026 = 2025-26 season
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    home_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=True)
    away_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=True)
    home_score: Mapped[int] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int] = mapped_column(Integer, nullable=True)
    neutral_site: Mapped[bool] = mapped_column(Boolean, default=False)        # True for tournament / bowl games
    tournament: Mapped[str] = mapped_column(String(50), nullable=True)        # "NCAA", "NIT", "Regular Season", etc.
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    home_team: Mapped["Team"] = relationship("Team", foreign_keys=[home_team_id], back_populates="home_games")
    away_team: Mapped["Team"] = relationship("Team", foreign_keys=[away_team_id], back_populates="away_games")
    box_scores: Mapped[list["BoxScore"]] = relationship("BoxScore", back_populates="game")

    def __repr__(self) -> str:
        return (
            f"<Game id={self.id} date={self.game_date} "
            f"home={self.home_team_id} away={self.away_team_id} "
            f"score={self.home_score}-{self.away_score}>"
        )


# ---------------------------------------------------------------------------
# Box Scores
# ---------------------------------------------------------------------------

class BoxScore(Base):
    """
    One row per team per game — the raw box score statistics.
    These are the inputs to all efficiency calculations in Phase 2.

    Column naming follows standard basketball notation:
      fga  = field goal attempts
      fgm  = field goals made
      fg3a = 3-point attempts
      fg3m = 3-point makes
      fta  = free throw attempts
      ftm  = free throws made
      oreb = offensive rebounds
      dreb = defensive rebounds
      tov  = turnovers
      pts  = total points

    possessions is computed and stored here so Phase 2 doesn't re-derive it:
      possessions = fga - oreb + tov + (0.475 * fta)
    """
    __tablename__ = "box_scores"
    __table_args__ = (
        UniqueConstraint("game_id", "team_id", name="uq_box_score_game_team"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(Integer, ForeignKey("games.id"), nullable=False)
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)

    # Raw counting stats
    pts: Mapped[int] = mapped_column(Integer, nullable=True)
    fga: Mapped[int] = mapped_column(Integer, nullable=True)
    fgm: Mapped[int] = mapped_column(Integer, nullable=True)
    fg3a: Mapped[int] = mapped_column(Integer, nullable=True)
    fg3m: Mapped[int] = mapped_column(Integer, nullable=True)
    fta: Mapped[int] = mapped_column(Integer, nullable=True)
    ftm: Mapped[int] = mapped_column(Integer, nullable=True)
    oreb: Mapped[int] = mapped_column(Integer, nullable=True)
    dreb: Mapped[int] = mapped_column(Integer, nullable=True)
    tov: Mapped[int] = mapped_column(Integer, nullable=True)
    ast: Mapped[int] = mapped_column(Integer, nullable=True)
    stl: Mapped[int] = mapped_column(Integer, nullable=True)
    blk: Mapped[int] = mapped_column(Integer, nullable=True)
    pf: Mapped[int] = mapped_column(Integer, nullable=True)                   # personal fouls

    # Computed and stored at ingest time
    possessions: Mapped[float] = mapped_column(Float, nullable=True)          # fga - oreb + tov + 0.475*fta

    # Relationships
    game: Mapped["Game"] = relationship("Game", back_populates="box_scores")
    team: Mapped["Team"] = relationship("Team", back_populates="box_scores")

    def compute_possessions(self) -> float | None:
        """
        Calculate possessions using the standard formula.
        Call this after setting fga, oreb, tov, fta.
        Stores result in self.possessions and returns it.
        """
        if any(v is None for v in [self.fga, self.oreb, self.tov, self.fta]):
            return None
        self.possessions = self.fga - self.oreb + self.tov + (0.475 * self.fta)
        return self.possessions

    def __repr__(self) -> str:
        return f"<BoxScore game={self.game_id} team={self.team_id} pts={self.pts} poss={self.possessions:.1f}>"


# ---------------------------------------------------------------------------
# CarmPom Ratings  (populated in Phase 2)
# ---------------------------------------------------------------------------

class CarmPomRating(Base):
    """
    One row per team per season — the computed CarmPom adjusted efficiency ratings.
    Populated and updated by ratings/engine.py in Phase 2.
    """
    __tablename__ = "carm_pom_ratings"
    __table_args__ = (
        UniqueConstraint("team_id", "season", name="uq_rating_team_season"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False)

    # Core adjusted efficiency metrics
    adjo: Mapped[float] = mapped_column(Float, nullable=True)                  # Adjusted Offensive Efficiency
    adjd: Mapped[float] = mapped_column(Float, nullable=True)                  # Adjusted Defensive Efficiency
    adjem: Mapped[float] = mapped_column(Float, nullable=True)                 # AdjO - AdjD (the headline number)
    adjt: Mapped[float] = mapped_column(Float, nullable=True)                  # Adjusted Tempo (possessions/40 min)

    # Secondary metrics
    luck: Mapped[float] = mapped_column(Float, nullable=True)                  # actual win% - expected win%
    sos: Mapped[float] = mapped_column(Float, nullable=True)                   # Strength of Schedule (WIN50)
    rank: Mapped[int] = mapped_column(Integer, nullable=True)                  # Current rank by AdjEM

    # Record
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)

    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    team: Mapped["Team"] = relationship("Team", back_populates="ratings")

    def __repr__(self) -> str:
        return (
            f"<CarmPomRating team={self.team_id} season={self.season} "
            f"rank={self.rank} AdjEM={self.adjem:.2f}>"
        )
