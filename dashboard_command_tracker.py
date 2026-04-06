"""
Lightweight human-command compliance for sim_test (PyQuaticus JSONL).

Kept separate from dashboard_tracker.py so the full analytics module stays unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict
import math
import time


@dataclass
class HumanCommand:
    command_id: str
    issued_at: float
    expires_at: float
    scope: str  # "player" | "team" | "all"
    target: str  # player_id or team_id or "all"
    text: str
    is_compliance_event: Callable[[dict], bool]
    is_violation_event: Callable[[dict], bool]


@dataclass
class CommandStatus:
    compliant_events: int = 0
    violation_events: int = 0
    first_compliance_time: Optional[float] = None
    active_time: float = 0.0


@dataclass
class PlayerState:
    team_id: Optional[str] = None


@dataclass
class MatchState:
    match_id: str
    start_time: float
    players: Dict[str, PlayerState] = field(default_factory=dict)


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


class GameAnalyticsTracker:
    def __init__(self) -> None:
        self.matches: Dict[str, MatchState] = {}
        self.active_commands: Dict[str, List[HumanCommand]] = defaultdict(list)
        self.command_status: Dict[str, Dict[str, Dict[str, CommandStatus]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(CommandStatus))
        )

    def start_match(self, match_id: str, timestamp: Optional[float] = None) -> None:
        ts = time.time() if timestamp is None else float(timestamp)
        self.matches[match_id] = MatchState(match_id=match_id, start_time=ts)

    def add_command(self, match_id: str, cmd: HumanCommand) -> None:
        if match_id not in self.matches:
            self.start_match(match_id, timestamp=cmd.issued_at)
        self.active_commands[match_id].append(cmd)

    def ingest_event(self, event: Dict[str, Any]) -> None:
        if "match_id" not in event or "event" not in event or "player_id" not in event:
            raise ValueError("Event missing required keys: match_id, event, player_id")
        match_id = str(event["match_id"])
        ts = float(event.get("timestamp", time.time()))
        pid = str(event.get("player_id"))

        if match_id not in self.matches:
            self.start_match(match_id, timestamp=ts)
        ms = self.matches[match_id]
        ps = ms.players.setdefault(pid, PlayerState())
        if event.get("team_id") is not None:
            ps.team_id = str(event["team_id"])

        self._evaluate_commands(match_id, event)

    def _evaluate_commands(self, match_id: str, event: Dict[str, Any]) -> None:
        ms = self.matches[match_id]
        ts = float(event.get("timestamp", time.time()))
        pid = str(event.get("player_id"))

        if match_id not in self.active_commands:
            return

        for cmd in list(self.active_commands[match_id]):
            if ts < cmd.issued_at or ts > cmd.expires_at:
                continue

            if cmd.scope == "player":
                entity = pid if cmd.target == pid else None
            elif cmd.scope == "team":
                team = ms.players.get(pid, PlayerState()).team_id
                entity = team if (team is not None and cmd.target == team) else None
            else:
                entity = "all"

            if entity is None:
                continue

            st = self.command_status[match_id][cmd.command_id][entity]

            if cmd.is_compliance_event(event):
                st.compliant_events += 1
                if st.first_compliance_time is None:
                    st.first_compliance_time = ts

            if cmd.is_violation_event(event):
                st.violation_events += 1

    def make_move_command(
        self,
        command_id: str,
        issued_at: float,
        expires_at: float,
        scope: str,
        target: str,
        text: str,
        target_x: float,
        target_y: float,
        compliance_radius: float = 4.0,
        violation_radius: float = 25.0,
    ) -> HumanCommand:
        def is_compliance(ev: dict) -> bool:
            if str(ev.get("event")) != "move":
                return False
            x = ev.get("x")
            y = ev.get("y")
            if x is None or y is None:
                return False
            return _dist(float(x), float(y), target_x, target_y) <= compliance_radius

        def is_violation(ev: dict) -> bool:
            if str(ev.get("event")) != "move":
                return False
            x = ev.get("x")
            y = ev.get("y")
            if x is None or y is None:
                return False
            return _dist(float(x), float(y), target_x, target_y) >= violation_radius

        return HumanCommand(
            command_id=command_id,
            issued_at=issued_at,
            expires_at=expires_at,
            scope=scope,
            target=target,
            text=text,
            is_compliance_event=is_compliance,
            is_violation_event=is_violation,
        )

    def make_hold_command(
        self,
        command_id: str,
        issued_at: float,
        expires_at: float,
        text: str,
        anchor_x: float,
        anchor_y: float,
        player_id: str,
        compliance_radius: float = 4.0,
        violation_radius: float = 18.0,
    ) -> HumanCommand:
        pid = str(player_id)

        def is_compliance(ev: dict) -> bool:
            if str(ev.get("event")) != "move":
                return False
            if str(ev.get("player_id")) != pid:
                return False
            x = ev.get("x")
            y = ev.get("y")
            if x is None or y is None:
                return False
            return _dist(float(x), float(y), anchor_x, anchor_y) <= compliance_radius

        def is_violation(ev: dict) -> bool:
            if str(ev.get("event")) != "move":
                return False
            if str(ev.get("player_id")) != pid:
                return False
            x = ev.get("x")
            y = ev.get("y")
            if x is None or y is None:
                return False
            return _dist(float(x), float(y), anchor_x, anchor_y) >= violation_radius

        return HumanCommand(
            command_id=command_id,
            issued_at=issued_at,
            expires_at=expires_at,
            scope="player",
            target=pid,
            text=text,
            is_compliance_event=is_compliance,
            is_violation_event=is_violation,
        )

    def make_attack_command(
        self,
        command_id: str,
        issued_at: float,
        expires_at: float,
        text: str,
        goal_x: float,
        goal_y: float,
        player_id: str,
        compliance_radius: float = 12.0,
        violation_far_radius: float = 55.0,
        closer_eps: float = 0.35,
        farther_eps: float = 0.35,
    ) -> HumanCommand:
        pid = str(player_id)

        def is_compliance(ev: dict) -> bool:
            if str(ev.get("event")) != "move":
                return False
            if str(ev.get("player_id")) != pid:
                return False
            x = ev.get("x")
            y = ev.get("y")
            if x is None or y is None:
                return False
            d_cur = _dist(float(x), float(y), goal_x, goal_y)
            if d_cur <= compliance_radius:
                return True
            px = ev.get("prev_x")
            py = ev.get("prev_y")
            if px is None or py is None:
                return False
            d_prev = _dist(float(px), float(py), goal_x, goal_y)
            return d_cur < d_prev - closer_eps

        def is_violation(ev: dict) -> bool:
            if str(ev.get("event")) != "move":
                return False
            if str(ev.get("player_id")) != pid:
                return False
            x = ev.get("x")
            y = ev.get("y")
            if x is None or y is None:
                return False
            d_cur = _dist(float(x), float(y), goal_x, goal_y)
            if d_cur >= violation_far_radius:
                return True
            px = ev.get("prev_x")
            py = ev.get("prev_y")
            if px is None or py is None:
                return False
            d_prev = _dist(float(px), float(py), goal_x, goal_y)
            return d_cur > d_prev + farther_eps

        return HumanCommand(
            command_id=command_id,
            issued_at=issued_at,
            expires_at=expires_at,
            scope="player",
            target=pid,
            text=text,
            is_compliance_event=is_compliance,
            is_violation_event=is_violation,
        )

    def make_spread_command(
        self,
        command_id: str,
        issued_at: float,
        expires_at: float,
        text: str,
        min_pairwise_ok: float = 14.0,
        cluster_violation_below: float = 8.0,
    ) -> HumanCommand:
        def is_compliance(ev: dict) -> bool:
            if str(ev.get("event")) != "spread_eval":
                return False
            try:
                d = float(ev.get("min_pairwise_blue", 0.0))
            except (TypeError, ValueError):
                return False
            return d >= min_pairwise_ok

        def is_violation(ev: dict) -> bool:
            if str(ev.get("event")) != "spread_eval":
                return False
            try:
                d = float(ev.get("min_pairwise_blue", 0.0))
            except (TypeError, ValueError):
                return False
            return d < cluster_violation_below

        return HumanCommand(
            command_id=command_id,
            issued_at=issued_at,
            expires_at=expires_at,
            scope="all",
            target="all",
            text=text,
            is_compliance_event=is_compliance,
            is_violation_event=is_violation,
        )
