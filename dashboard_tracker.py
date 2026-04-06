from __future__ import annotations

from dataclasses import dataclass, field

from typing import Any, Dict, List, Optional, Tuple

from collections import defaultdict, deque

import time

import json

import math

import numpy as np

import pandas as pd

import networkx as nx

from sklearn.metrics import f1_score

from sklearn.ensemble import IsolationForest

from sklearn.preprocessing import StandardScaler


# ----------------------------

# Event schema (JSON dict)

# ----------------------------

# Required minimal keys:

# {

#   "match_id": "m1",

#   "timestamp": 123.45,

#   "event": "kill" | "death" | "assist" | "shot" | "hit" | "miss" | "damage" | "heal"

#            | "objective" | "move" | "spawn" | "disconnect" | "reconnect" | "end",

#   "player_id": "P1",

#   ... optional keys based on event type ...

# }

#

# Common optional keys:

#  - target_id (for kill/damage/hit/miss)

#  - team_id ("A"/"B")

#  - amount (damage/heal)

#  - obj_id / obj_type for objective

#  - x,y (position)

#  - dx,dy (movement delta) OR prev_x,prev_y,x,y

#  - weapon, headshot, etc. (ignored unless you use them)


@dataclass

class PlayerState:

    team_id: Optional[str] = None

    connected: bool = True

    # Core counts

    kills: int = 0

    deaths: int = 0

    assists: int = 0

    shots: int = 0

    hits: int = 0

    misses: int = 0

    damage_dealt: float = 0.0

    damage_taken: float = 0.0

    healing_done: float = 0.0

    support_actions: int = 0

    objective_captures: int = 0

    # Time + movement

    spawn_time: Optional[float] = None

    total_survival_time: float = 0.0

    last_pos: Optional[Tuple[float, float]] = None

    distance_covered: float = 0.0

    # Momentum rating (Elo-like)

    momentum_rating: float = 1000.0

    momentum_uncertainty: float = 200.0  # Bayesian-like uncertainty proxy


@dataclass

class MatchState:

    match_id: str

    start_time: float

    end_time: Optional[float] = None

    players: Dict[str, PlayerState] = field(default_factory=dict)

    # For synergy graph (assist chains / proximity / interactions)

    assist_graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    proximity_graph: nx.Graph = field(default_factory=nx.Graph)

    # Store positions over time for spectral/trajectory-based scoring

    trajectories: Dict[str, List[Tuple[float, float, float]]] = field(default_factory=lambda: defaultdict(list))

    # player -> list of (t, x, y)

    # store raw events (bounded memory)

    event_buffer: List[Dict[str, Any]] = field(default_factory=list)


class GameAnalyticsTracker:

    """

    Real-time tracking system:

    - ingest_event(event_json)

    - get_live_leaderboard()

    - finalize_match() -> metrics outputs + CSV export

    """

    def __init__(self, export_dir: str = ".", buffer_limit: int = 200000):

        self.export_dir = export_dir

        self.buffer_limit = buffer_limit

        self.matches: Dict[str, MatchState] = {}

    # ----------------------------

    # Public API

    # ----------------------------

    def start_match(self, match_id: str, timestamp: Optional[float] = None):

        ts = time.time() if timestamp is None else float(timestamp)

        self.matches[match_id] = MatchState(match_id=match_id, start_time=ts)

    def ingest_event(self, event: Dict[str, Any]):

        # Basic validation

        if "match_id" not in event or "event" not in event or "player_id" not in event:

            raise ValueError("Event missing required keys: match_id, event, player_id")

        match_id = str(event["match_id"])

        ts = float(event.get("timestamp", time.time()))

        etype = str(event["event"])

        pid = str(event["player_id"])

        if match_id not in self.matches:

            # auto-start match if not started

            self.start_match(match_id, timestamp=ts)

        ms = self.matches[match_id]

        ps = ms.players.setdefault(pid, PlayerState())

        # capture team if provided

        if "team_id" in event and event["team_id"] is not None:

            ps.team_id = str(event["team_id"])

        # raw buffer (for CSV/paper)

        ms.event_buffer.append(dict(event))

        if len(ms.event_buffer) > self.buffer_limit:

            ms.event_buffer.pop(0)

        # Update by event type

        if etype == "spawn":

            ps.spawn_time = ts

            ps.connected = True

        elif etype == "disconnect":

            ps.connected = False

            # close survival segment if alive

            if ps.spawn_time is not None:

                ps.total_survival_time += max(0.0, ts - ps.spawn_time)

                ps.spawn_time = None

        elif etype == "reconnect":

            ps.connected = True

            # treat as new life segment

            ps.spawn_time = ts if ps.spawn_time is None else ps.spawn_time

        elif etype == "shot":

            ps.shots += 1

        elif etype == "hit":

            ps.hits += 1

            ps.shots += 1  # if your engine logs hit without shot, keep consistent

            target_id = event.get("target_id")

            if target_id is not None:

                ms.proximity_graph.add_edge(pid, str(target_id), weight=ms.proximity_graph.get_edge_data(pid, str(target_id), {}).get("weight", 0) + 1)

        elif etype == "miss":

            ps.misses += 1

            ps.shots += 1

        elif etype == "damage":

            amt = float(event.get("amount", 0.0))

            ps.damage_dealt += max(0.0, amt)

            tgt = event.get("target_id")

            if tgt is not None:

                tgt = str(tgt)

                ms.players.setdefault(tgt, PlayerState()).damage_taken += max(0.0, amt)

        elif etype == "heal":

            amt = float(event.get("amount", 0.0))

            ps.healing_done += max(0.0, amt)

            ps.support_actions += 1

        elif etype == "assist":

            ps.assists += 1

            assister = pid

            killer = str(event.get("killer_id", pid))

            target = str(event.get("target_id", "unknown"))

            # assist chain graph: assister -> killer

            ms.assist_graph.add_edge(assister, killer, weight=ms.assist_graph.get_edge_data(assister, killer, {}).get("weight", 0) + 1)

            ms.assist_graph.add_edge(assister, target, weight=ms.assist_graph.get_edge_data(assister, target, {}).get("weight", 0) + 1)

        elif etype == "kill":

            ps.kills += 1

            target = event.get("target_id")

            if target is not None:

                tgt = str(target)

                ms.players.setdefault(tgt, PlayerState()).deaths += 1

                # momentum update per kill/death (Elo-like)

                self._update_momentum_kill(ms, killer=pid, victim=tgt)

                # interaction edge for synergy (kill links)

                ms.assist_graph.add_edge(pid, tgt, weight=ms.assist_graph.get_edge_data(pid, tgt, {}).get("weight", 0) + 1)

        elif etype == "death":

            ps.deaths += 1

        elif etype == "objective":

            ps.objective_captures += 1

        elif etype == "move":

            # movement: allow (x,y) or (dx,dy)

            x = event.get("x")

            y = event.get("y")

            if x is not None and y is not None:

                x, y = float(x), float(y)

                self._update_position(ms, pid, ts, x, y)

            else:

                dx = float(event.get("dx", 0.0))

                dy = float(event.get("dy", 0.0))

                if ps.last_pos is None:

                    self._update_position(ms, pid, ts, dx, dy)  # treat as absolute if first

                else:

                    nxp = ps.last_pos[0] + dx

                    nyp = ps.last_pos[1] + dy

                    self._update_position(ms, pid, ts, nxp, nyp)

        elif etype == "end":

            self.finalize_match(match_id, end_timestamp=ts)

        # else: unknown events are kept in buffer but ignored analytically

    def get_live_leaderboard(self, match_id: str, top_n: int = 10) -> pd.DataFrame:

        ms = self.matches[match_id]

        rows = []

        match_len = self._match_length(ms)

        for pid, ps in ms.players.items():

            prec, rec, f1 = self._standard_ml_metrics(ps)

            rows.append({

                "player_id": pid,

                "team_id": ps.team_id,

                "kills": ps.kills,

                "deaths": ps.deaths,

                "assists": ps.assists,

                "kd": ps.kills / max(1, ps.deaths),

                "precision": prec,

                "recall": rec,

                "f1": f1,

                "momentum_rating": ps.momentum_rating,

                "movement_efficiency": self._movement_efficiency(ps, match_len),

            })

        df = pd.DataFrame(rows).sort_values(["f1", "momentum_rating"], ascending=False).head(top_n)

        return df

    def finalize_match(self, match_id: str, end_timestamp: Optional[float] = None) -> Dict[str, Any]:

        ms = self.matches[match_id]

        if ms.end_time is None:

            ms.end_time = float(end_timestamp if end_timestamp is not None else time.time())

        # Close survival segments

        for ps in ms.players.values():

            if ps.spawn_time is not None:

                ps.total_survival_time += max(0.0, ms.end_time - ps.spawn_time)

                ps.spawn_time = None

        # Compute outputs

        player_df = self._compute_player_table(ms)

        team_df = self._compute_team_table(ms, player_df)

        novel = self._compute_novel_metrics(ms, player_df)

        # Export CSVs

        raw_df = pd.DataFrame(ms.event_buffer)

        raw_path = f"{self.export_dir}/events_{match_id}.csv"

        p_path = f"{self.export_dir}/players_{match_id}.csv"

        t_path = f"{self.export_dir}/teams_{match_id}.csv"

        raw_df.to_csv(raw_path, index=False)

        player_df.to_csv(p_path, index=False)

        team_df.to_csv(t_path, index=False)

        return {

            "match_id": match_id,

            "match_length": self._match_length(ms),

            "player_table": player_df,

            "team_table": team_df,

            "novel_metrics": novel,

            "csv": {"events": raw_path, "players": p_path, "teams": t_path},

        }

    # ----------------------------

    # Internal helpers

    # ----------------------------

    def _match_length(self, ms: MatchState) -> float:

        endt = ms.end_time if ms.end_time is not None else time.time()

        return max(1e-6, endt - ms.start_time)

    def _update_position(self, ms: MatchState, pid: str, ts: float, x: float, y: float):

        ps = ms.players.setdefault(pid, PlayerState())

        if ps.last_pos is not None:

            dx = x - ps.last_pos[0]

            dy = y - ps.last_pos[1]

            ps.distance_covered += math.sqrt(dx * dx + dy * dy)

        ps.last_pos = (x, y)

        ms.trajectories[pid].append((ts, x, y))

    def _standard_ml_metrics(self, ps: PlayerState) -> Tuple[float, float, float]:

        # Precision: kill efficiency = kills / (kills + misses)

        # Recall: threat neutralization proxy = kills / (kills + deaths)

        precision = ps.kills / max(1, ps.kills + ps.misses)

        recall = ps.kills / max(1, ps.kills + ps.deaths)

        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

        return precision, recall, f1

    def _movement_efficiency(self, ps: PlayerState, match_len: float) -> float:

        # Normalize by match length to handle different duration

        return ps.distance_covered / max(1e-6, match_len)

    def _update_momentum_kill(self, ms: MatchState, killer: str, victim: str):

        k = ms.players[killer]

        v = ms.players[victim]

        # Elo-like expected score

        def expected(ra, rb):

            return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

        ea = expected(k.momentum_rating, v.momentum_rating)

        eb = expected(v.momentum_rating, k.momentum_rating)

        # K-factor shrinks with more certainty (Bayesian-ish proxy)

        k_factor_k = 24 * (k.momentum_uncertainty / 200.0)

        k_factor_v = 24 * (v.momentum_uncertainty / 200.0)

        # killer "wins"

        k.momentum_rating += k_factor_k * (1.0 - ea)

        v.momentum_rating += k_factor_v * (0.0 - eb)

        # uncertainty shrinks gradually with interactions

        k.momentum_uncertainty = max(50.0, k.momentum_uncertainty * 0.995)

        v.momentum_uncertainty = max(50.0, v.momentum_uncertainty * 0.995)

    def _compute_player_table(self, ms: MatchState) -> pd.DataFrame:

        match_len = self._match_length(ms)

        rows = []

        for pid, ps in ms.players.items():

            prec, rec, f1 = self._standard_ml_metrics(ps)

            # Positioning score (simple, replace with map-specific logic):

            # lower average distance from team centroid => better cohesion/positioning

            pos_score = self._positioning_score(ms, pid)

            rows.append({

                "match_id": ms.match_id,

                "player_id": pid,

                "team_id": ps.team_id,

                "connected": ps.connected,

                "kills": ps.kills,

                "deaths": ps.deaths,

                "assists": ps.assists,

                "kd": ps.kills / max(1, ps.deaths),

                "damage_dealt": ps.damage_dealt,

                "damage_taken": ps.damage_taken,

                "healing_done": ps.healing_done,

                "support_actions": ps.support_actions,

                "objective_captures": ps.objective_captures,

                "survival_time": ps.total_survival_time,

                "distance_covered": ps.distance_covered,

                "movement_efficiency": self._movement_efficiency(ps, match_len),

                "positioning_score": pos_score,

                "precision": prec,

                "recall": rec,

                "f1": f1,

                "momentum_rating": ps.momentum_rating,

                "momentum_uncertainty": ps.momentum_uncertainty,

            })

        df = pd.DataFrame(rows)

        # Normalize for imbalanced teams + match length

        # (paper-friendly, comparable across matches)

        df["kills_per_min"] = df["kills"] / max(1e-6, match_len / 60.0)

        df["damage_per_min"] = df["damage_dealt"] / max(1e-6, match_len / 60.0)

        return df

    def _positioning_score(self, ms: MatchState, pid: str) -> float:

        ps = ms.players[pid]

        if ps.team_id is None:

            return 0.0

        traj = ms.trajectories.get(pid, [])

        if len(traj) < 2:

            return 0.0

        # For each timestamp, compute distance to team centroid at that time (approx using latest positions)

        # Cheap approximation: compare final positions only

        team_mates = [p for p, st in ms.players.items() if st.team_id == ps.team_id and ms.players[p].last_pos is not None]

        if len(team_mates) <= 1:

            return 0.0

        pts = np.array([ms.players[p].last_pos for p in team_mates], dtype=float)

        centroid = pts.mean(axis=0)

        myp = np.array(ps.last_pos, dtype=float)

        dist = float(np.linalg.norm(myp - centroid))

        # Convert to score: closer => higher (bounded)

        return 1.0 / (1.0 + dist)

    def _compute_team_table(self, ms: MatchState, player_df: pd.DataFrame) -> pd.DataFrame:

        # Handle imbalanced teams: normalize per active player

        team_groups = player_df.groupby("team_id", dropna=False)

        rows = []

        for team_id, g in team_groups:

            active = max(1, len(g))

            rows.append({

                "match_id": ms.match_id,

                "team_id": team_id,

                "players": active,

                "kills": int(g["kills"].sum()),

                "deaths": int(g["deaths"].sum()),

                "assists": int(g["assists"].sum()),

                "kd": float(g["kills"].sum() / max(1, g["deaths"].sum())),

                "damage_dealt": float(g["damage_dealt"].sum()),

                "healing_done": float(g["healing_done"].sum()),

                "objective_captures": int(g["objective_captures"].sum()),

                "avg_f1": float(g["f1"].mean()),

                "avg_momentum": float(g["momentum_rating"].mean()),

                "avg_positioning": float(g["positioning_score"].mean()),

                "kills_per_player": float(g["kills"].sum() / active),

                "damage_per_player": float(g["damage_dealt"].sum() / active),

            })

        return pd.DataFrame(rows)

    # ----------------------------

    # Novel differentiators

    # ----------------------------

    def _compute_novel_metrics(self, ms: MatchState, player_df: pd.DataFrame) -> Dict[str, Any]:

        out: Dict[str, Any] = {}

        # 1) Threat F1-Score:

        # "Predict/Eliminate high-threat opponents (top-KD players)"

        # We'll label high-threat players = top 30% KD

        tmp = player_df.copy()

        if len(tmp) >= 4:

            thr = tmp["kd"].quantile(0.70)

            tmp["is_threat"] = (tmp["kd"] >= thr).astype(int)

            # "Predicted threat" via simple embedding proxy:

            # Use [kills_per_min, damage_per_min, positioning_score] as embedding-like features

            feats = tmp[["kills_per_min", "damage_per_min", "positioning_score"]].fillna(0.0).to_numpy()

            feats = StandardScaler().fit_transform(feats)

            # prediction heuristic: high L2 norm => threat (proxy for learned embedding magnitude)

            norms = np.linalg.norm(feats, axis=1)

            pred = (norms >= np.median(norms)).astype(int)

            out["threat_f1_score"] = float(f1_score(tmp["is_threat"], pred))

        else:

            out["threat_f1_score"] = 0.0

        # 2) Momentum True Score:

        # Return distribution summary (mean, std, top/bottom)

        ratings = player_df[["player_id", "momentum_rating", "momentum_uncertainty"]].copy()

        out["momentum_true_score"] = {

            "mean": float(ratings["momentum_rating"].mean()),

            "std": float(ratings["momentum_rating"].std(ddof=0)),

            "top3": ratings.sort_values("momentum_rating", ascending=False).head(3).to_dict("records"),

            "bottom3": ratings.sort_values("momentum_rating", ascending=True).head(3).to_dict("records"),

        }

        # 3) Team Synergy Index:

        # Build graph from assist chains + proximity edges; spectral clustering proxy score = algebraic connectivity-ish

        out["team_synergy_index"] = self._team_synergy_index(ms)

        # 4) Anomaly Kill Score:

        # IsolationForest on kill vectors: [kills, deaths, assists, damage_dealt, objective_captures]

        out["anomaly_kill_score"] = self._anomaly_kill_scores(player_df)

        return out

    def _team_synergy_index(self, ms: MatchState) -> Dict[str, float]:

        # For each team, build an interaction graph among teammates:

        # edges from assist_graph (assist chains) + proximity_graph (frequent interactions)

        res: Dict[str, float] = {}

        teams = sorted({ps.team_id for ps in ms.players.values() if ps.team_id is not None})

        for team_id in teams:

            members = [pid for pid, ps in ms.players.items() if ps.team_id == team_id]

            if len(members) < 2:

                res[team_id] = 0.0

                continue

            G = nx.Graph()

            G.add_nodes_from(members)

            # add assist edges within team

            for u, v, data in ms.assist_graph.edges(data=True):

                if u in members and v in members:

                    w = float(data.get("weight", 1.0))

                    G.add_edge(u, v, weight=G.get_edge_data(u, v, {}).get("weight", 0.0) + w)

            # add proximity edges (also within team if you log them)

            for u, v, data in ms.proximity_graph.edges(data=True):

                if u in members and v in members:

                    w = float(data.get("weight", 1.0))

                    G.add_edge(u, v, weight=G.get_edge_data(u, v, {}).get("weight", 0.0) + w)

            if G.number_of_edges() == 0:

                res[team_id] = 0.0

                continue

            # Spectral-ish synergy score:

            # Use normalized Laplacian eigenvalues; lower average eigenvalue => more clustered/cohesive

            L = nx.normalized_laplacian_matrix(G, weight="weight").astype(float)

            eig = np.linalg.eigvals(L.A)

            eig = np.real(eig)

            # Convert into synergy score (bounded): more connectivity => higher score

            # Use (1 - mean(eig)) but clamp

            score = float(max(0.0, min(1.0, 1.0 - float(np.mean(eig)))))

            res[team_id] = score

        return res

    def _anomaly_kill_scores(self, player_df: pd.DataFrame) -> Dict[str, float]:

        cols = ["kills", "deaths", "assists", "damage_dealt", "objective_captures"]

        X = player_df[cols].fillna(0.0).to_numpy()

        if len(player_df) < 5:

            return {r["player_id"]: 0.0 for r in player_df[["player_id"]].to_dict("records")}

        Xs = StandardScaler().fit_transform(X)

        iso = IsolationForest(contamination=0.2, random_state=42)

        iso.fit(Xs)

        # decision_function: higher = more normal; lower = more anomalous

        scores = iso.decision_function(Xs)

        # invert & normalize to "anomaly score" where higher means more anomalous

        an = -scores

        an = (an - an.min()) / (an.max() - an.min() + 1e-9)

        return {pid: float(s) for pid, s in zip(player_df["player_id"].tolist(), an)}

def simulate_three_matches():

    tracker = GameAnalyticsTracker(export_dir="exports")

    players = [f"P{i}" for i in range(10)]

    teams = {p: ("A" if i < 5 else "B") for i, p in enumerate(players)}

    for m in range(1, 4):

        match_id = f"m{m}"

        tracker.start_match(match_id, timestamp=0.0)

        # spawn everyone

        for p in players:

            tracker.ingest_event({"match_id": match_id, "timestamp": 0.0, "event": "spawn", "player_id": p, "team_id": teams[p]})

        # simulate events

        t = 0.0

        for step in range(200):

            t += 1.0

            # movement for everyone

            for p in players:

                tracker.ingest_event({

                    "match_id": match_id,

                    "timestamp": t,

                    "event": "move",

                    "player_id": p,

                    "team_id": teams[p],

                    "dx": float(np.random.randn() * 0.8),

                    "dy": float(np.random.randn() * 0.8),

                })

            # create variance:

            # Player P0: high F1 (lots of kills, low misses) but low synergy (few assists)

            # Player P6: moderate F1 but high synergy (many assists + positioning)

            attacker = "P0" if random.random() < 0.35 else random.choice(players)

            victim = random.choice([p for p in players if teams[p] != teams[attacker]])

            # shots / hits / misses

            if attacker == "P0":

                # accurate killer

                tracker.ingest_event({"match_id": match_id, "timestamp": t, "event": "hit", "player_id": attacker, "target_id": victim, "team_id": teams[attacker]})

                tracker.ingest_event({"match_id": match_id, "timestamp": t, "event": "kill", "player_id": attacker, "target_id": victim, "team_id": teams[attacker]})

            else:

                # mixed accuracy

                if random.random() < 0.45:

                    tracker.ingest_event({"match_id": match_id, "timestamp": t, "event": "hit", "player_id": attacker, "target_id": victim, "team_id": teams[attacker]})

                    tracker.ingest_event({"match_id": match_id, "timestamp": t, "event": "kill", "player_id": attacker, "target_id": victim, "team_id": teams[attacker]})

                else:

                    tracker.ingest_event({"match_id": match_id, "timestamp": t, "event": "miss", "player_id": attacker, "target_id": victim, "team_id": teams[attacker]})

            # assists: make P6 a synergy-heavy support player

            if random.random() < 0.30:

                tracker.ingest_event({

                    "match_id": match_id, "timestamp": t, "event": "assist",

                    "player_id": "P6", "killer_id": attacker, "target_id": victim,

                    "team_id": teams["P6"]

                })

            # heals: support actions

            if random.random() < 0.12:

                tracker.ingest_event({"match_id": match_id, "timestamp": t, "event": "heal", "player_id": "P6", "team_id": teams["P6"], "amount": 25})

            # objectives sometimes

            if random.random() < 0.06:

                tracker.ingest_event({"match_id": match_id, "timestamp": t, "event": "objective", "player_id": random.choice(players), "team_id": teams[random.choice(players)]})

            # disconnection edge case

            if step == 120 and m == 2:

                tracker.ingest_event({"match_id": match_id, "timestamp": t, "event": "disconnect", "player_id": "P3", "team_id": teams["P3"]})

        # finalize match

        out = tracker.finalize_match(match_id, end_timestamp=t)

        print(f"\n=== Match {match_id} ===")

        print(out["player_table"].sort_values("f1", ascending=False)[["player_id","team_id","kills","deaths","misses","precision","recall","f1","momentum_rating","positioning_score"]].head(8))

        print("Novel metrics:", out["novel_metrics"])

        # Live leaderboard example

        lb = tracker.get_live_leaderboard(match_id, top_n=5)

        print("\nTop-5 live leaderboard snapshot:")

        print(lb)

if __name__ == "__main__":

    simulate_three_matches()

import json

import random

import math

import pandas as pd

import numpy as np

import networkx as nx

from collections import defaultdict

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.ensemble import IsolationForest

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# =========================================================

# CONFIG

# =========================================================

NUM_PLAYERS = 10

MATCHES = 3

EVENTS_PER_MATCH = 120

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

# =========================================================

# EVENT SIMULATION

# =========================================================

def simulate_match(match_id):

    events = []

    teams = {i: i % 2 for i in range(NUM_PLAYERS)}  # 2 teams

    alive = {i: True for i in range(NUM_PLAYERS)}

    positions = {i: np.random.rand(2) * 100 for i in range(NUM_PLAYERS)}

    for t in range(EVENTS_PER_MATCH):

        attacker = random.randint(0, NUM_PLAYERS - 1)

        target = random.randint(0, NUM_PLAYERS - 1)

        # movement

        positions[attacker] += np.random.randn(2)

        if attacker != target and teams[attacker] != teams[target]:

            hit_prob = 0.6 if attacker in [0, 1] else 0.4

            if random.random() < hit_prob:

                events.append({

                    "match": match_id,

                    "player_id": attacker,

                    "event": "kill",

                    "target_id": target,

                    "timestamp": t,

                    "x": positions[attacker][0],

                    "y": positions[attacker][1]

                })

                alive[target] = False

            else:

                events.append({

                    "match": match_id,

                    "player_id": attacker,

                    "event": "miss",

                    "target_id": target,

                    "timestamp": t,

                    "x": positions[attacker][0],

                    "y": positions[attacker][1]

                })

        else:

            events.append({

                "match": match_id,

                "player_id": attacker,

                "event": "move",

                "timestamp": t,

                "x": positions[attacker][0],

                "y": positions[attacker][1]

            })

    return events

# =========================================================

# EVENT INGESTION

# =========================================================

def ingest_events():

    all_events = []

    for m in range(MATCHES):

        all_events.extend(simulate_match(m))

    return pd.DataFrame(all_events)

# =========================================================

# BASIC STATS

# =========================================================

def compute_basic_stats(df):

    stats = defaultdict(lambda: defaultdict(int))

    for _, r in df.iterrows():

        pid = r["player_id"]

        if r["event"] == "kill":

            stats[pid]["kills"] += 1

            stats[r["target_id"]]["deaths"] += 1

        elif r["event"] == "miss":

            stats[pid]["misses"] += 1

    rows = []

    for pid, s in stats.items():

        kills = s["kills"]

        deaths = max(1, s["deaths"])

        misses = s["misses"]

        precision = kills / max(1, kills + misses)

        recall = kills / max(1, kills + deaths)

        f1 = 2 * precision * recall / max(1e-6, precision + recall)

        rows.append({

            "player_id": pid,

            "kills": kills,

            "deaths": deaths,

            "misses": misses,

            "precision": precision,

            "recall": recall,

            "f1_score": f1

        })

    return pd.DataFrame(rows)

# =========================================================

# NOVEL METRIC 1: THREAT F1-SCORE

# =========================================================

def threat_f1_score(stats_df):

    # top 30% players by K/D are high threat

    stats_df["kd"] = stats_df["kills"] / stats_df["deaths"]

    threshold = stats_df["kd"].quantile(0.7)

    stats_df["is_threat"] = stats_df["kd"] >= threshold

    y_true = stats_df["is_threat"].astype(int)

    y_pred = (stats_df["kills"] > stats_df["kills"].median()).astype(int)

    return f1_score(y_true, y_pred)

# =========================================================

# NOVEL METRIC 2: MOMENTUM TRUE SCORE (ELO-LIKE)

# =========================================================

def momentum_true_score(df):

    ratings = defaultdict(lambda: 1000)

    for _, r in df.iterrows():

        if r["event"] == "kill":

            a, b = r["player_id"], r["target_id"]

            ratings[a] += 5

            ratings[b] -= 5

    return ratings

# =========================================================

# NOVEL METRIC 3: TEAM SYNERGY INDEX (GRAPH)

# =========================================================

def team_synergy(df):

    G = nx.Graph()

    for _, r in df.iterrows():

        if r["event"] == "kill":

            G.add_edge(r["player_id"], r["target_id"])

    if G.number_of_nodes() < 2:

        return 0.0

    laplacian = nx.normalized_laplacian_matrix(G).todense()

    eigvals = np.linalg.eigvals(laplacian)

    return float(np.mean(eigvals.real))

# =========================================================

# NOVEL METRIC 4: ANOMALY KILL SCORE

# =========================================================

def anomaly_kill_score(df):

    kill_vectors = []

    for pid, g in df[df["event"] == "kill"].groupby("player_id"):

        kill_vectors.append([pid, len(g)])

    if len(kill_vectors) < 2:

        return {}

    X = np.array([v[1:] for v in kill_vectors])

    X = StandardScaler().fit_transform(X)

    iso = IsolationForest(contamination=0.2, random_state=42)

    scores = iso.fit_predict(X)

    return {kill_vectors[i][0]: scores[i] for i in range(len(scores))}

# =========================================================

# DASHBOARD

# =========================================================

def plot_leaderboard(stats_df):

    stats_df.sort_values("f1_score", ascending=False).plot(

        x="player_id", y="f1_score", kind="bar", title="F1 Score Leaderboard"

    )

    plt.tight_layout()

    plt.show()

# =========================================================

# MAIN

# =========================================================

if __name__ == "__main__":

    df = ingest_events()

    df.to_csv("raw_events.csv", index=False)

    stats = compute_basic_stats(df)

    stats.to_csv("player_stats.csv", index=False)

    threat_f1 = threat_f1_score(stats)

    momentum = momentum_true_score(df)

    synergy = team_synergy(df)

    anomaly = anomaly_kill_score(df)

    print("\n=== CORE METRICS ===")

    print(stats.sort_values("f1_score", ascending=False))

    print("\nThreat F1-Score:", threat_f1)

    print("Team Synergy Index:", synergy)

    print("Momentum Ratings:", dict(momentum))

    print("Anomaly Kill Scores (−1 = anomaly):", anomaly)

    plot_leaderboard(stats)

 