"""
Lightweight human-command compliance for sim_test (PyQuaticus JSONL).

Kept separate from dashboard_tracker.py so the full analytics module stays unchanged.

Compliance / violation match Block 11 in sim_test (progress-based, same eps):
  - Compliance: within goal radius OR distance to target decreased vs previous position.
  - Violation: distance to target increased vs previous position (and previous exists).
  - Move/hold/attack: BLUE team only; events with is_tagged=True or has_flag=True are not scored
    (carrier / return-home behavior would inflate violations).
  - Spread: uses prev_min_pairwise_blue on spread_eval events (team bunched vs last step).
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


# Match sim_test Block 11 (_pq_goal_ok / _pq_cmd_eps) for JSONL dashboard_command_status.
PQ_GOAL_OK = 4.0
PQ_CMD_EPS = 0.35


def _blue_untagged_move(ev: dict) -> bool:
    """Human blue commands: only score BLUE agents that are not tagged and not carrying the flag."""
    if str(ev.get("event")) != "move":
        return False
    if str(ev.get("team_id")) != "BLUE":
        return False
    if bool(ev.get("is_tagged")):
        return False
    if bool(ev.get("has_flag")):
        return False
    return True


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
        goal_ok: float = PQ_GOAL_OK,
        progress_eps: float = PQ_CMD_EPS,
    ) -> HumanCommand:
        def is_compliance(ev: dict) -> bool:
            if not _blue_untagged_move(ev):
                return False
            x = ev.get("x")
            y = ev.get("y")
            if x is None or y is None:
                return False
            d_cur = _dist(float(x), float(y), target_x, target_y)
            if d_cur <= goal_ok:
                return True
            px = ev.get("prev_x")
            py = ev.get("prev_y")
            if px is None or py is None:
                return False
            d_prev = _dist(float(px), float(py), target_x, target_y)
            return d_cur < d_prev - progress_eps

        def is_violation(ev: dict) -> bool:
            if not _blue_untagged_move(ev):
                return False
            x = ev.get("x")
            y = ev.get("y")
            if x is None or y is None:
                return False
            d_cur = _dist(float(x), float(y), target_x, target_y)
            px = ev.get("prev_x")
            py = ev.get("prev_y")
            if px is None or py is None:
                return False
            d_prev = _dist(float(px), float(py), target_x, target_y)
            return d_cur > d_prev + progress_eps

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
        goal_ok: float = PQ_GOAL_OK,
        progress_eps: float = PQ_CMD_EPS,
    ) -> HumanCommand:
        pid = str(player_id)

        def is_compliance(ev: dict) -> bool:
            if not _blue_untagged_move(ev):
                return False
            if str(ev.get("player_id")) != pid:
                return False
            x = ev.get("x")
            y = ev.get("y")
            if x is None or y is None:
                return False
            d_cur = _dist(float(x), float(y), anchor_x, anchor_y)
            if d_cur <= goal_ok:
                return True
            px = ev.get("prev_x")
            py = ev.get("prev_y")
            if px is None or py is None:
                return False
            d_prev = _dist(float(px), float(py), anchor_x, anchor_y)
            return d_cur < d_prev - progress_eps

        def is_violation(ev: dict) -> bool:
            if not _blue_untagged_move(ev):
                return False
            if str(ev.get("player_id")) != pid:
                return False
            x = ev.get("x")
            y = ev.get("y")
            if x is None or y is None:
                return False
            d_cur = _dist(float(x), float(y), anchor_x, anchor_y)
            px = ev.get("prev_x")
            py = ev.get("prev_y")
            if px is None or py is None:
                return False
            d_prev = _dist(float(px), float(py), anchor_x, anchor_y)
            return d_cur > d_prev + progress_eps

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
        goal_ok: float = PQ_GOAL_OK,
        progress_eps: float = PQ_CMD_EPS,
    ) -> HumanCommand:
        pid = str(player_id)

        def is_compliance(ev: dict) -> bool:
            if not _blue_untagged_move(ev):
                return False
            if str(ev.get("player_id")) != pid:
                return False
            x = ev.get("x")
            y = ev.get("y")
            if x is None or y is None:
                return False
            d_cur = _dist(float(x), float(y), goal_x, goal_y)
            if d_cur <= goal_ok:
                return True
            px = ev.get("prev_x")
            py = ev.get("prev_y")
            if px is None or py is None:
                return False
            d_prev = _dist(float(px), float(py), goal_x, goal_y)
            return d_cur < d_prev - progress_eps

        def is_violation(ev: dict) -> bool:
            if not _blue_untagged_move(ev):
                return False
            if str(ev.get("player_id")) != pid:
                return False
            x = ev.get("x")
            y = ev.get("y")
            if x is None or y is None:
                return False
            d_cur = _dist(float(x), float(y), goal_x, goal_y)
            px = ev.get("prev_x")
            py = ev.get("prev_y")
            if px is None or py is None:
                return False
            d_prev = _dist(float(px), float(py), goal_x, goal_y)
            return d_cur > d_prev + progress_eps

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
        progress_eps: float = PQ_CMD_EPS,
    ) -> HumanCommand:
        """
        Progress-based spread: compliance when spread is good or improved vs prev snapshot;
        violation when min pairwise distance drops vs previous step (see prev_min_pairwise_blue).
        """

        def is_compliance(ev: dict) -> bool:
            if str(ev.get("event")) != "spread_eval":
                return False
            try:
                m = float(ev.get("min_pairwise_blue", 0.0))
            except (TypeError, ValueError):
                return False
            if m >= min_pairwise_ok:
                return True
            pv = ev.get("prev_min_pairwise_blue")
            if pv is None:
                return False
            try:
                pv_f = float(pv)
            except (TypeError, ValueError):
                return False
            return m > pv_f + 1e-6

        def is_violation(ev: dict) -> bool:
            if str(ev.get("event")) != "spread_eval":
                return False
            try:
                m = float(ev.get("min_pairwise_blue", 0.0))
            except (TypeError, ValueError):
                return False
            pv = ev.get("prev_min_pairwise_blue")
            if pv is None:
                return False
            try:
                pv_f = float(pv)
            except (TypeError, ValueError):
                return False
            return m < pv_f - progress_eps

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
