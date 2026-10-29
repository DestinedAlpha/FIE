from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Position = Literal[1, 2, 3, 4]
Chip = Literal["wildcard", "freehit", "bboost", "3xc"]
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Player(Strict):
    id: int
    code: int
    season: str
    name: str
    web_name: str
    team: int
    position: Position
    price: int = Field(ge=0)
    selectable: bool = True
    ownership: float = Field(ge=0, le=100)
    status: str
    news: str
    chance: float | None = Field(default=None, ge=0, le=1)
    raw: dict


class Lot(Strict):
    player_id: int
    purchase_price: int | None = Field(default=None, ge=0)
    sell_price: int = Field(ge=0)
    quote_market_price: int = Field(ge=0)
    provenance: str = "manual"

    @model_validator(mode="after")
    def feasible_quote(self):
        if self.sell_price > self.quote_market_price:
            raise ValueError("Selling quote cannot exceed its contemporaneous market price")
        return self


class ChipUse(Strict):
    chip: Chip
    event: int = Field(ge=1, le=38)


class TeamState(Strict):
    season: str
    event: int = Field(ge=1, le=38)
    name: str = "My team"
    entry_id: int | None = None
    bank: int = Field(ge=0)
    free_transfers: int = Field(ge=0, le=5)
    lots: list[Lot]
    used_chips: list[ChipUse] = Field(default_factory=list)
    points: int = 0
    transfers_made: int = Field(default=0, ge=0, le=20)
    hit_paid: int = Field(default=0, ge=0)
    deadline_lots: list[Lot] | None = None
    deadline_bank: int | None = Field(default=None, ge=0)
    initial: bool = False
    grandfathered_clubs: list[int] = Field(default_factory=list)
    status: Literal["dated_seed", "public_deadline", "confirmed"] = "confirmed"
    known_at: str

    @field_validator("known_at")
    @classmethod
    def timestamp(cls, value):
        if datetime.fromisoformat(value).tzinfo is None:
            raise ValueError("Account knowledge time requires an explicit timezone")
        return value

    @model_validator(mode="after")
    def unique_lots(self):
        if len({lot.player_id for lot in self.lots}) != len(self.lots):
            raise ValueError("Duplicate player in squad")
        return self


class Lineup(Strict):
    xi: list[int]
    bench: list[int]
    captain: int
    vice: int


class Action(Strict):
    sell: list[int] = Field(default_factory=list)
    buy: list[int] = Field(default_factory=list)
    chip: Chip | None = None
    lineup: Lineup | None = None

    @model_validator(mode="after")
    def unique_transfers(self):
        if len(set(self.buy)) != len(self.buy) or len(set(self.sell)) != len(self.sell):
            raise ValueError("Duplicate transfer")
        if set(self.buy) & set(self.sell):
            raise ValueError("Buying and selling the same player in a bundle")
        return self


class OptimizeRequest(Strict):
    state_id: str
    forecast_id: str
    end_event: int = Field(ge=1, le=38)
    time_limit: float = Field(default=30, ge=1, le=300)
    objective: Literal["expected_points", "beat_both"] = "expected_points"
    sacrifice_budget: float = Field(default=5, ge=0, le=100)
    chips: dict[int, Chip] = Field(default_factory=dict)
    max_transfers: int = Field(default=3, ge=0, le=15)
