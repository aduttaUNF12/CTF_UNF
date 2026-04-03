from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict
import time
import math


@dataclass
class HumanCommand:
    command_id: str
    issued_at: float
    expires_at: float
    scope: str  # "player" | "team" | "all"
    target: str  # player_id or team_id or "all"
    text: str

    # Functions that decide compliance/violation from an incoming event
    is_compliance_event: Callable[[dict], bool]
    is_violation_event: Callable[[dict], bool]


@dataclass
class CommandStatus:
    compliant_events: int = 0
    violation_events: int = 0
    first_compliance_time: Optional[float] = None
    active_time: float = 0.0  # accumulated in finalize (optional)


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
    """
    Lightweight event + command compliance tracker.

    This is a minimal subset of the larger dashboard code, intended to be
    safe to import even if pandas/sklearn/etc. are not installed.
    """

    def __init__(self) -> None:
        self.matches: Dict[str, MatchState] = {}
        self.active_commands: Dict[str, List[HumanCommand]] = defaultdict(list)  # match_id -> commands
        # match_id -> command_id -> entity_id(player/team/all) -> status
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

            # determine which entity key we score against
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

    # Convenience: build a move-to command
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

