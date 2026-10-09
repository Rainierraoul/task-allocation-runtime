"""A small 2D simulation for Scenario 5. Units: nautical miles, knots, seconds.

Heading 0 is north (+y) and 90 is east (+x). The tanker flies a circular
orbit. Positions, callsigns and channels are sample values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Aircraft:
    callsign: str
    x: float
    y: float
    heading_deg: float
    speed_kts: float

    def step(self, dt_s: float) -> None:
        d = self.speed_kts * dt_s / 3600.0
        h = math.radians(self.heading_deg)
        self.x += d * math.sin(h)
        self.y += d * math.cos(h)


@dataclass
class Tanker:
    callsign: str
    cx: float
    cy: float
    radius_nm: float
    speed_kts: float
    angle_rad: float = 0.0  # position on the orbit, measured from north, clockwise

    @property
    def x(self) -> float:
        return self.cx + self.radius_nm * math.sin(self.angle_rad)

    @property
    def y(self) -> float:
        return self.cy + self.radius_nm * math.cos(self.angle_rad)

    def position_after(self, dt_s: float) -> tuple[float, float]:
        a = self.angle_rad + (self.speed_kts * dt_s / 3600.0) / self.radius_nm
        return self.cx + self.radius_nm * math.sin(a), self.cy + self.radius_nm * math.cos(a)

    def step(self, dt_s: float) -> None:
        self.angle_rad += (self.speed_kts * dt_s / 3600.0) / self.radius_nm


@dataclass
class World:
    transmission: str
    channel_state: str = "clear"  # clear | in_use | jammed
    jamming_inject_scheduled: bool = False
    trainee_level: str = "initial_employment"
    deviation_inject_scheduled: bool = False
    time_s: float = 0.0
    aircraft: dict[str, Aircraft] = field(default_factory=dict)
    tanker: Optional[Tanker] = None
    selected: Optional[str] = None
    distance_line: Optional[tuple[str, str]] = None
    map_centre: Optional[str] = None
    radio_log: list[tuple[float, str, str]] = field(default_factory=list)
    plan: dict[str, str] = field(default_factory=dict)
    control_acknowledged: bool = False

    @classmethod
    def sample(cls, transmission: str, **kwargs) -> "World":
        w = cls(transmission=transmission, **kwargs)
        w.aircraft = {
            "aircraft 1": Aircraft("aircraft 1", 0.0, 10.0, 90.0, 400.0),
            "aircraft 2": Aircraft("aircraft 2", 0.0, 0.0, 90.0, 420.0),
        }
        w.tanker = Tanker("SHELL 1", cx=60.0, cy=40.0, radius_nm=8.0, speed_kts=300.0)
        return w

    def entities(self) -> list[str]:
        names = list(self.aircraft)
        if self.tanker:
            names.append(self.tanker.callsign)
        return names

    def distance_to_tanker(self, callsign: str = "aircraft 2") -> float:
        a = self.aircraft[callsign]
        assert self.tanker
        return math.hypot(self.tanker.x - a.x, self.tanker.y - a.y)

    def step(self, dt_s: float) -> None:
        for a in self.aircraft.values():
            a.step(dt_s)
        if self.tanker:
            self.tanker.step(dt_s)
        self.time_s += dt_s

    def clock(self) -> str:
        m, s = divmod(int(self.time_s), 60)
        return f"{m:02d}:{s:02d}"
