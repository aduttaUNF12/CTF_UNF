# Framework Block Audit: Are the 12 Blocks Implemented?

This document checks each framework block against the current codebase (`all_blocks.py` + `sim_main.py`).

---

## 1. ENVIRONMENT INITIALIZATION

**Framework asks for:** Load environment; spawn N robots (N > 10); reset map, goals, flags, obstacles; initialize robot sensors, communication buffers; start logging.

| Item | Status | Notes |
|------|--------|--------|
| Load environment | ✅ | PyQuaticusEnv in `sim_main.py`; `block1_environment_initialization` creates initial `GlobalState` and is assigned to `env.global_state`. |
| Spawn N robots (N > 10) | ⚠️ | Config has `num_robots=12`, but `sim_main` uses `team_size` (1–6) so total agents = 2×team_size (2–12). N > 10 only if user picks 6. |
| Reset map, goals, flags | ✅ | Done by `env.reset()` in PyQuaticus; Block 1 does not reset—env does. |
| Obstacles | ✅ | PyQuaticus world has boundaries; Block 8 grid uses boundary buffer. |
| Robot sensors / comm buffers | ⚠️ | No explicit “sensors” or “communication buffers” struct; `LocalObservation` has `messages_in` (currently unused). |
| Logging system | ✅ | `gs.history` holds actions, rewards, human_constraints, llm_summaries; Block 9 appends to it when used. |

**Gap:** Block 1 in `all_blocks` builds a generic robot dict; real spawn/reset is in PyQuaticus. No explicit sensor/comm buffer init. For N > 10, require team_size ≥ 6 or cap from config.

---

## 2. STATE COLLECTION (RAW MULTI-ROBOT OBSERVATIONS)

**Framework asks for:** Each Ri collects local oi: nearby map/objects/enemies, teammate positions, local hazards; no full observability; all observations gathered centrally.

| Item | Status | Notes |
|------|--------|--------|
| Per-robot local observation | ✅ | `block2_state_collection` builds `LocalObservation` per robot (position, nearby, teammates, hazards, messages_in). |
| Nearby map/objects/enemies | ✅ | In `sim_main`, we pass `env.state_to_obs(..., normalize=False)` into Block 2; `nearby` includes env-driven `wall_near` and `enemy_spotted` (from opponent distances). Falls back to placeholder random only if env_obs not provided. |
| Teammate positions | ✅ | `teammates = [(tid, gs.robots[tid]["position"]) for ...]` from current `gs.robots`. |
| Local hazards/danger zones | ✅ | In `sim_main`, Block 2 derives hazards from env obs: `red_zone` (on enemy side), and `wall_close` when wall distance is below threshold. Falls back to placeholder random only if env_obs not provided. |
| No full observability | ✅ | Each oi is local (own position + subset of others + placeholder nearby/hazards). |
| Central gathering | ✅ | All observations returned in one list by Block 2. |

**Gap:** `nearby`/`hazards` now come from env walls/opponents + on_side. Still missing richer “objects/obstacles” beyond walls, and any explicit minefield/keepout hazards besides red_zone/wall_close.

---

## 3. GLOBAL STATE ENCODING (CTDE)

**Framework asks for:** Combine all observations into S_t; include robot states 1..N, environment map, historical actions & rewards, previous human constraints, previous LLM suggestions; input for strategy generation.

| Item | Status | Notes |
|------|--------|--------|
| Combine into global S_t | ✅ | `block3_global_state_encoding` takes `gs` + `local_obs`, updates `gs.map_summary`. |
| Robot states 1..N | ✅ | In `gs.robots` (position, has_flag, hp, team after Block 7). |
| Environment map | ✅ | `gs.map_summary` (hazard count, enemy signals, history counts). |
| Historical actions & rewards | ✅ | `gs.history["actions"]`, `gs.history["rewards"]`. |
| Previous human constraints | ✅ | `gs.history["human_constraints"]`. |
| Previous LLM suggestions | ✅ | `gs.history["llm_summaries"]`. |
| Input for strategy generation | ✅ | Block 4 uses `gs` (and optionally ctf_state). |

**Gap:** None for the listed items. Optional: richer map encoding (e.g. occupancy or sectors).

---

## 4. LLM STATE SUMMARIZATION

**Framework asks for:** Convert S_t to natural language; produce candidate strategies with reasoning (“Team A defend…”, “Team B flank right…”, “Team C explore…”); generate uncertainties, warnings, risk predictions.

| Item | Status | Notes |
|------|--------|--------|
| Natural language summary | ✅ | Rule-based summary + optional LLM; “Frontline weak…”, “Carriers …”, etc. |
| Candidate strategies | ✅ | 2–3 strategies (stubs or LLM); e.g. “Team A defend; Team B flank right; Team C explore north.” |
| Uncertainties / warnings / risks | ✅ | Each `StrategyCandidate` has `risks` and `uncertainties`; printed in Block 5. |

**Gap:** None. Stubs satisfy the spec; LLM path exists when API is configured.

---

## 5. HUMAN STRATEGY INTERVENTION

**Framework asks for:** Human reads LLM summary; approves, modifies, merges, or rejects strategies; adds priorities, safety constraints, mission goals; output = final HUMAN+LLM plan.

| Item | Status | Notes |
|------|--------|--------|
| Human reads summary | ✅ | Block 5 prints summary and candidate strategies. |
| Approve / modify / reject | ✅ | User chooses 1–N or types custom; that becomes `approved_strategies`. |
| Priorities | ✅ | Fixed in plan: `priorities=["Defense > Capture"]`. |
| Safety constraints | ✅ | Default `safety_constraints=["Avoid red_zone"]` (no longer prompted). |
| Mission goals | ✅ | `mission_goals=["Secure perimeter", "Capture opponent flag"]`. |
| Output = final plan | ✅ | `HumanPlan` returned and reused until replan. |

**Gap:** Priorities and mission goals are fixed; could be made editable by human if desired.

---

## 6. HIGH-LEVEL RL MANAGER (HRL META-POLICY)

**Framework asks for:** Inputs: Human+LLM plan, encoded S_t, robot team performance history. Decides: (A) team splits, (B) subgoals (hold region, capture flag, patrol, scout).

| Item | Status | Notes |
|------|--------|--------|
| Input: Human+LLM plan | ✅ | `human_plan` passed to `block6_high_level_rl_manager`. |
| Input: Encoded S_t | ✅ | `gs` (includes map_summary, robots, history). |
| Input: Robot team performance history | ⚠️ | Not passed explicitly; only via `gs.history` (actions, rewards). Could be summarized and passed as extra arg. |
| (A) Which robot groups/teams | ✅ | Defend / Flank / Attack splits by strategy index or keyword parsing. |
| (B) Subgoals (hold, capture, patrol, scout) | ✅ | Subgoals: Defend our flag, Flank right, Attack red zone; roles map to Block 8 goals. |

**Gap:** “Robot team performance history” is only implicit in `gs.history`; no explicit performance metric (e.g. per-team reward or success rate) yet.

---

## 7. SUBGOAL DISPATCHING TO ROBOT TEAMS

**Framework asks for:** Subgoal g_k assigned to team Tk; each subgoal encoded into vector; each robot receives its team’s subgoal with local meaning.

| Item | Status | Notes |
|------|--------|--------|
| Tk = {robot i1, i2, …} | ✅ | `hrl.teams` e.g. Defend=[R0,R1], Flank=[R2,R3], Attack=[R4,R5]. |
| Subgoal g_k per team | ✅ | `hrl.subgoals[team_id]` (Defend, Flank, Attack). |
| Encoded into vector | ✅ | `Subgoal.vector` from `encode_subgoal(description)`. |
| Each Ri receives team subgoal | ✅ | Block 7 sets `gs.robots[rid]["team"] = team_id`; Block 8 looks up `hrl.subgoals[team_id]`. |

**Gap:** None.

---

## 8. LOW-LEVEL MULTI-AGENT RL EXECUTION (MAPPO / MASAC)

**Framework asks for:** Each Ri uses policy πi to select primitive action (movement, attack/tag/collect, communicate); inputs: oi, encoded g_k, past teammate messages; outputs: action ai.

| Item | Status | Notes |
|------|--------|--------|
| Primitive action (movement) | ✅ | Block 8 outputs direction (N, S, E, W, etc.) or HOLD; converted to env action in sim_main. |
| Attack / tag / collect | ⚠️ | No explicit “attack/tag” action; tagging is env mechanic (collision). Goal for carrier = return home. |
| Communicate | ⚠️ | `messages_in` in LocalObservation not used; no message output in actions. |
| Inputs: oi, g_k | ✅ | Block 8 uses `local_obs` (oi) and `hrl.subgoals[team_id]` (g_k). |
| Inputs: Past teammate messages | ❌ | Not used. |
| Policy | ⚠️ | A* + fallback direction (placeholder for MAPPO/MASAC); no learned policy yet. |

**Gap:** Low-level is rule-based (A* + subgoal goal); no MAPPO/MASAC training. No communication actions or message history.

---

## 9. ENVIRONMENT TRANSITION

**Framework asks for:** Apply all actions; world → S_(t+1); compute reward (task progress, goal completion, damage/collisions, comm overhead, safety); update logs.

| Item | Status | Notes |
|------|--------|--------|
| Apply all actions | ✅ | In sim_main: `env.step(actions)` applies blue (from Block 8) + red (heuristic). |
| World → S_(t+1) | ✅ | PyQuaticus advances state. |
| Reward (task, goal, damage, safety) | ⚠️ | PyQuaticus computes its own rewards; Block 9 in all_blocks is a stub (not used in sim_main) with simple “moved” reward. |
| Update logs | ⚠️ | Block 9 would append to gs.history; in sim_main, env handles state; gs.history is not updated each step for actions/rewards. |

**Gap:** sim_main does not call Block 9; transition and reward are entirely in the env. To align with framework, either call Block 9 to log actions/rewards from env, or document that “env performs Block 9.”

---

## 10. CHECK FOR STRATEGIC CHANGES

**Framework asks for:** Is plan failing? Team stuck or overwhelmed? New threats? If YES → return to Human+LLM (4–5); else continue.

| Item | Status | Notes |
|------|--------|--------|
| Plan failing | ✅ | `block10_check_strategic_changes`: score_red increased since last replan. |
| Team stuck / overwhelmed | ❌ | Removed earlier; no “stuck” check. |
| New threats | ✅ | red_has_blue_flag just became True. |
| Periodic replan | ✅ | Every REPLAN_INTERVAL_STEPS (e.g. 60). |
| Return to 4–5 | ✅ | sim_main sets `human_plan = None` when Block 10 returns True. |

**Gap:** “Team stuck or overwhelmed” is not implemented; could add back a simple displacement or progress check.

---

## 11. LEARNING & POLICY UPDATES

**Framework asks for:** After episode or periodically: high-level (PPO/A2C on subgoal outcomes), low-level (MAPPO/MASAC CTDE), LLM optional (no training).

| Item | Status | Notes |
|------|--------|--------|
| High-level meta-RL | ❌ | `block11_learning_and_updates` is a stub; no PPO/A2C. |
| Low-level MARL | ❌ | No MAPPO/MASAC training; Block 8 is A* only. |
| LLM | ✅ | Not trained (as per framework). |
| When | ⚠️ | Block 11 exists but is not called in sim_main. |

**Gap:** Blocks 11 is placeholder only; no real learning. sim_main does not call it.

---

## 12. TERMINATION CHECK

**Framework asks for:** Win/loss? Task finished? Human manually ends?

| Item | Status | Notes |
|------|--------|--------|
| Win/loss | ✅ | `block12_termination_check(gs, cfg, env)` checks `env.dones['__all__']` and ends the main loop with `env.message` (e.g. “Blue Wins! Red Loses”). |
| Task finished | ✅ | `block12_termination_check` also stops at `gs.t >= cfg.max_steps_per_episode` (max steps). |
| Human manually ends | ✅ | ESC or window close in sim_main. |
| Loop termination | ✅ | sim_main breaks after env.step when Block 12 returns done. |

**Gap:** None for termination; win/loss and max steps are wired in.

---

## Summary Table

| Block | Mostly aligned? | Main gaps |
|-------|-----------------|-----------|
| 1 | ✅ | N > 10 only for team_size=6; no explicit sensor/comm init. |
| 2 | ✅ | Env-derived walls/opponents/on_side are used; still no comm/messages and no extra hazard types beyond wall/red_zone. |
| 3 | ✅ | None. |
| 4 | ✅ | None. |
| 5 | ✅ | Priorities/goals fixed. |
| 6 | ✅ | Team performance history only implicit in gs.history. |
| 7 | ✅ | None. |
| 8 | ⚠️ | A* placeholder, no MAPPO/MASAC; no comm/tag actions. |
| 9 | ⚠️ | Not used in sim_main; env does transition. |
| 10 | ✅ | No “team stuck” check. |
| 11 | ❌ | Stub only; not called in sim_main. |
| 12 | ✅ | Wired into sim_main using env.dones/env.message + max steps. |

Overall, Blocks 1–7, 10 and 12 match the framework conceptually; Block 8 is a rule-based placeholder for MARL; Blocks 9 and 11 are either delegated to the env or not wired into sim_main.
