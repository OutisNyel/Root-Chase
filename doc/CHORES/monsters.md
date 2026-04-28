---
title: Monsters Observation Samples (is_in_view validation)
doc_type: chore
doc_id: chores-monsters-is-in-view-sample
updated: 2026-04-20
keywords:
  - monsters
  - frame_state.monsters
  - MonsterState
  - is_in_view
  - monster_id
  - hero_relative_direction
  - hero_l2_distance
  - monster_interval
  - pos.x
  - pos.z
  - speed
  - train_test
  - agent_diy
search_aliases:
  - monster observation samples
  - is_in_view samples
  - monsters field output
---

# Monsters Observation Samples (is_in_view validation)

## Search Index
- Document ID: `chores-monsters-is-in-view-sample`
- Primary keywords: `monsters`, `is_in_view`, `frame_state.monsters`, `MonsterState`
- Secondary keywords: `monster_id`, `monster_interval`, `hero_relative_direction`, `hero_l2_distance`, `speed`, `pos`
- Scope: `train_test` / `agent_diy` observation verification
- Related code: `code/agent_diy/feature/preprocessor.py`

## Summary
- Total parsed samples: **415**
- `is_in_view=1`: **407** (98.07%)
- `is_in_view=0`: **8** (1.93%)
- In this sample set, when `is_in_view=0`:
  - `pos = {x:-1, z:-1}`: **8/8**
  - `speed = -1`: **8/8**

## Field Notes
| Field | Notes | Observed pattern |
|---|---|---|
| `hero_l2_distance` | Distance bucket to hero | all `0` in this sample set |
| `hero_relative_direction` | Relative direction from monster to hero (0-8) | values `0..8` observed |
| `monster_id` | Monster entity id | all `14` |
| `monster_interval` | Configured second-monster interval | all `300` |
| `pos.x`, `pos.z` | Monster position | visible: map coords; invisible: often `-1,-1` |
| `speed` | Monster speed | visible: `1`; invisible: often `-1` |
| `is_in_view` | Visibility flag | `1` or `0` |

## Distribution Snapshot
- `monster_id`: `{14: 415}`
- `monster_interval`: `{300: 415}`
- `speed`: `{1: 407, -1: 8}`
- `hero_relative_direction`:
  - `0`: 1
  - `1`: 26
  - `2`: 117
  - `3`: 27
  - `4`: 58
  - `5`: 19
  - `6`: 79
  - `7`: 44
  - `8`: 44

## Typical Records
### Visible (`is_in_view=1`)
```json
{
  "hero_l2_distance": 0,
  "hero_relative_direction": 6,
  "monster_id": 14,
  "monster_interval": 300,
  "pos": {
    "x": 19,
    "z": 89
  },
  "speed": 1,
  "is_in_view": 1
}
```

### Not visible (`is_in_view=0`)
```json
{
  "hero_l2_distance": 0,
  "hero_relative_direction": 2,
  "monster_id": 14,
  "monster_interval": 300,
  "pos": {
    "x": -1,
    "z": -1
  },
  "speed": -1,
  "is_in_view": 0
}
```

## Full Cleaned Records
<details>
<summary>Expand to view 415 monster records</summary>

```json
{"idx": 1, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 19, "z": 89}, "speed": 1, "is_in_view": 1}
{"idx": 2, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 19, "z": 88}, "speed": 1, "is_in_view": 1}
{"idx": 3, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 60, "z": 37}, "speed": 1, "is_in_view": 1}
{"idx": 4, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": -1, "z": -1}, "speed": -1, "is_in_view": 0}
{"idx": 5, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 59, "z": 38}, "speed": 1, "is_in_view": 1}
{"idx": 6, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 58, "z": 38}, "speed": 1, "is_in_view": 1}
{"idx": 7, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 58, "z": 39}, "speed": 1, "is_in_view": 1}
{"idx": 8, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 57, "z": 39}, "speed": 1, "is_in_view": 1}
{"idx": 9, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 57, "z": 40}, "speed": 1, "is_in_view": 1}
{"idx": 10, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 56, "z": 40}, "speed": 1, "is_in_view": 1}
{"idx": 11, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 56, "z": 41}, "speed": 1, "is_in_view": 1}
{"idx": 12, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 55, "z": 41}, "speed": 1, "is_in_view": 1}
{"idx": 13, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 55, "z": 42}, "speed": 1, "is_in_view": 1}
{"idx": 14, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 55, "z": 43}, "speed": 1, "is_in_view": 1}
{"idx": 15, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 54, "z": 43}, "speed": 1, "is_in_view": 1}
{"idx": 16, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 54, "z": 44}, "speed": 1, "is_in_view": 1}
{"idx": 17, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 54, "z": 45}, "speed": 1, "is_in_view": 1}
{"idx": 18, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 55, "z": 45}, "speed": 1, "is_in_view": 1}
{"idx": 19, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 55, "z": 46}, "speed": 1, "is_in_view": 1}
{"idx": 20, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 54, "z": 46}, "speed": 1, "is_in_view": 1}
{"idx": 21, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 54, "z": 47}, "speed": 1, "is_in_view": 1}
{"idx": 22, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 53, "z": 47}, "speed": 1, "is_in_view": 1}
{"idx": 23, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 53, "z": 48}, "speed": 1, "is_in_view": 1}
{"idx": 24, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 53, "z": 49}, "speed": 1, "is_in_view": 1}
{"idx": 25, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 85, "z": 110}, "speed": 1, "is_in_view": 1}
{"idx": 26, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": -1, "z": -1}, "speed": -1, "is_in_view": 0}
{"idx": 27, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": -1, "z": -1}, "speed": -1, "is_in_view": 0}
{"idx": 28, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": -1, "z": -1}, "speed": -1, "is_in_view": 0}
{"idx": 29, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 110, "z": 97}, "speed": 1, "is_in_view": 1}
{"idx": 30, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 110, "z": 96}, "speed": 1, "is_in_view": 1}
{"idx": 31, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 110, "z": 95}, "speed": 1, "is_in_view": 1}
{"idx": 32, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 109, "z": 95}, "speed": 1, "is_in_view": 1}
{"idx": 33, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 109, "z": 94}, "speed": 1, "is_in_view": 1}
{"idx": 34, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 109, "z": 93}, "speed": 1, "is_in_view": 1}
{"idx": 35, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 109, "z": 92}, "speed": 1, "is_in_view": 1}
{"idx": 36, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 109, "z": 91}, "speed": 1, "is_in_view": 1}
{"idx": 37, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 108, "z": 91}, "speed": 1, "is_in_view": 1}
{"idx": 38, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 108, "z": 90}, "speed": 1, "is_in_view": 1}
{"idx": 39, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 107, "z": 90}, "speed": 1, "is_in_view": 1}
{"idx": 40, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 107, "z": 89}, "speed": 1, "is_in_view": 1}
{"idx": 41, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 106, "z": 89}, "speed": 1, "is_in_view": 1}
{"idx": 42, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 106, "z": 88}, "speed": 1, "is_in_view": 1}
{"idx": 43, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 106, "z": 87}, "speed": 1, "is_in_view": 1}
{"idx": 44, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 105, "z": 87}, "speed": 1, "is_in_view": 1}
{"idx": 45, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 105, "z": 86}, "speed": 1, "is_in_view": 1}
{"idx": 46, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 104, "z": 86}, "speed": 1, "is_in_view": 1}
{"idx": 47, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 104, "z": 85}, "speed": 1, "is_in_view": 1}
{"idx": 48, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 77, "z": 79}, "speed": 1, "is_in_view": 1}
{"idx": 49, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 78, "z": 79}, "speed": 1, "is_in_view": 1}
{"idx": 50, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": -1, "z": -1}, "speed": -1, "is_in_view": 0}
{"idx": 51, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 79, "z": 78}, "speed": 1, "is_in_view": 1}
{"idx": 52, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 80, "z": 77}, "speed": 1, "is_in_view": 1}
{"idx": 53, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 81, "z": 77}, "speed": 1, "is_in_view": 1}
{"idx": 54, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 82, "z": 77}, "speed": 1, "is_in_view": 1}
{"idx": 55, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 83, "z": 77}, "speed": 1, "is_in_view": 1}
{"idx": 56, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 83, "z": 76}, "speed": 1, "is_in_view": 1}
{"idx": 57, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 84, "z": 76}, "speed": 1, "is_in_view": 1}
{"idx": 58, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 85, "z": 76}, "speed": 1, "is_in_view": 1}
{"idx": 59, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 85, "z": 75}, "speed": 1, "is_in_view": 1}
{"idx": 60, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 86, "z": 75}, "speed": 1, "is_in_view": 1}
{"idx": 61, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 87, "z": 75}, "speed": 1, "is_in_view": 1}
{"idx": 62, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 87, "z": 74}, "speed": 1, "is_in_view": 1}
{"idx": 63, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 88, "z": 74}, "speed": 1, "is_in_view": 1}
{"idx": 64, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 88, "z": 73}, "speed": 1, "is_in_view": 1}
{"idx": 65, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 89, "z": 73}, "speed": 1, "is_in_view": 1}
{"idx": 66, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 90, "z": 73}, "speed": 1, "is_in_view": 1}
{"idx": 67, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 90, "z": 72}, "speed": 1, "is_in_view": 1}
{"idx": 68, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 90, "z": 71}, "speed": 1, "is_in_view": 1}
{"idx": 69, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 91, "z": 71}, "speed": 1, "is_in_view": 1}
{"idx": 70, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 91, "z": 70}, "speed": 1, "is_in_view": 1}
{"idx": 71, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 92, "z": 70}, "speed": 1, "is_in_view": 1}
{"idx": 72, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 45, "z": 32}, "speed": 1, "is_in_view": 1}
{"idx": 73, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 46, "z": 32}, "speed": 1, "is_in_view": 1}
{"idx": 74, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 46, "z": 31}, "speed": 1, "is_in_view": 1}
{"idx": 75, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 45, "z": 31}, "speed": 1, "is_in_view": 1}
{"idx": 76, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 86, "z": 96}, "speed": 1, "is_in_view": 1}
{"idx": 77, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 86, "z": 95}, "speed": 1, "is_in_view": 1}
{"idx": 78, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 87, "z": 95}, "speed": 1, "is_in_view": 1}
{"idx": 79, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 87, "z": 94}, "speed": 1, "is_in_view": 1}
{"idx": 80, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 87, "z": 93}, "speed": 1, "is_in_view": 1}
{"idx": 81, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 87, "z": 92}, "speed": 1, "is_in_view": 1}
{"idx": 82, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 86, "z": 92}, "speed": 1, "is_in_view": 1}
{"idx": 83, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 86, "z": 91}, "speed": 1, "is_in_view": 1}
{"idx": 84, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 87, "z": 91}, "speed": 1, "is_in_view": 1}
{"idx": 85, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 87, "z": 90}, "speed": 1, "is_in_view": 1}
{"idx": 86, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 88, "z": 90}, "speed": 1, "is_in_view": 1}
{"idx": 87, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 43, "z": 72}, "speed": 1, "is_in_view": 1}
{"idx": 88, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 42, "z": 72}, "speed": 1, "is_in_view": 1}
{"idx": 89, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 37, "z": 75}, "speed": 1, "is_in_view": 1}
{"idx": 90, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 36, "z": 76}, "speed": 1, "is_in_view": 1}
{"idx": 91, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": -1, "z": -1}, "speed": -1, "is_in_view": 0}
{"idx": 92, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 32, "z": 79}, "speed": 1, "is_in_view": 1}
{"idx": 93, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 31, "z": 79}, "speed": 1, "is_in_view": 1}
{"idx": 94, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 30, "z": 79}, "speed": 1, "is_in_view": 1}
{"idx": 95, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 29, "z": 79}, "speed": 1, "is_in_view": 1}
{"idx": 96, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 29, "z": 80}, "speed": 1, "is_in_view": 1}
{"idx": 97, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 28, "z": 80}, "speed": 1, "is_in_view": 1}
{"idx": 98, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 27, "z": 80}, "speed": 1, "is_in_view": 1}
{"idx": 99, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 26, "z": 80}, "speed": 1, "is_in_view": 1}
{"idx": 100, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 26, "z": 81}, "speed": 1, "is_in_view": 1}
{"idx": 101, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 25, "z": 81}, "speed": 1, "is_in_view": 1}
{"idx": 102, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 24, "z": 81}, "speed": 1, "is_in_view": 1}
{"idx": 103, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 24, "z": 80}, "speed": 1, "is_in_view": 1}
{"idx": 104, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 23, "z": 80}, "speed": 1, "is_in_view": 1}
{"idx": 105, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 22, "z": 80}, "speed": 1, "is_in_view": 1}
{"idx": 106, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 21, "z": 80}, "speed": 1, "is_in_view": 1}
{"idx": 107, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 21, "z": 81}, "speed": 1, "is_in_view": 1}
{"idx": 108, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 42, "z": 63}, "speed": 1, "is_in_view": 1}
{"idx": 109, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": -1, "z": -1}, "speed": -1, "is_in_view": 0}
{"idx": 110, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 43, "z": 69}, "speed": 1, "is_in_view": 1}
{"idx": 111, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 43, "z": 70}, "speed": 1, "is_in_view": 1}
{"idx": 112, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 43, "z": 71}, "speed": 1, "is_in_view": 1}
{"idx": 113, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 44, "z": 71}, "speed": 1, "is_in_view": 1}
{"idx": 114, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 45, "z": 71}, "speed": 1, "is_in_view": 1}
{"idx": 115, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 46, "z": 71}, "speed": 1, "is_in_view": 1}
{"idx": 116, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 47, "z": 71}, "speed": 1, "is_in_view": 1}
{"idx": 117, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 48, "z": 71}, "speed": 1, "is_in_view": 1}
{"idx": 118, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 49, "z": 71}, "speed": 1, "is_in_view": 1}
{"idx": 119, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 50, "z": 71}, "speed": 1, "is_in_view": 1}
{"idx": 120, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 51, "z": 71}, "speed": 1, "is_in_view": 1}
{"idx": 121, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 51, "z": 70}, "speed": 1, "is_in_view": 1}
{"idx": 122, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 52, "z": 70}, "speed": 1, "is_in_view": 1}
{"idx": 123, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 52, "z": 69}, "speed": 1, "is_in_view": 1}
{"idx": 124, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 53, "z": 69}, "speed": 1, "is_in_view": 1}
{"idx": 125, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 53, "z": 68}, "speed": 1, "is_in_view": 1}
{"idx": 126, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 54, "z": 68}, "speed": 1, "is_in_view": 1}
{"idx": 127, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 54, "z": 67}, "speed": 1, "is_in_view": 1}
{"idx": 128, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 55, "z": 67}, "speed": 1, "is_in_view": 1}
{"idx": 129, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 55, "z": 66}, "speed": 1, "is_in_view": 1}
{"idx": 130, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 56, "z": 66}, "speed": 1, "is_in_view": 1}
{"idx": 131, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 56, "z": 65}, "speed": 1, "is_in_view": 1}
{"idx": 132, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 55, "z": 65}, "speed": 1, "is_in_view": 1}
{"idx": 133, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 106, "z": 74}, "speed": 1, "is_in_view": 1}
{"idx": 134, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 106, "z": 75}, "speed": 1, "is_in_view": 1}
{"idx": 135, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 105, "z": 75}, "speed": 1, "is_in_view": 1}
{"idx": 136, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 104, "z": 76}, "speed": 1, "is_in_view": 1}
{"idx": 137, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 103, "z": 76}, "speed": 1, "is_in_view": 1}
{"idx": 138, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 103, "z": 77}, "speed": 1, "is_in_view": 1}
{"idx": 139, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 102, "z": 77}, "speed": 1, "is_in_view": 1}
{"idx": 140, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 102, "z": 76}, "speed": 1, "is_in_view": 1}
{"idx": 141, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 101, "z": 76}, "speed": 1, "is_in_view": 1}
{"idx": 142, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 101, "z": 75}, "speed": 1, "is_in_view": 1}
{"idx": 143, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 100, "z": 75}, "speed": 1, "is_in_view": 1}
{"idx": 144, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 100, "z": 74}, "speed": 1, "is_in_view": 1}
{"idx": 145, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 99, "z": 74}, "speed": 1, "is_in_view": 1}
{"idx": 146, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 98, "z": 74}, "speed": 1, "is_in_view": 1}
{"idx": 147, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 97, "z": 74}, "speed": 1, "is_in_view": 1}
{"idx": 148, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 96, "z": 74}, "speed": 1, "is_in_view": 1}
{"idx": 149, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 95, "z": 74}, "speed": 1, "is_in_view": 1}
{"idx": 150, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 61, "z": 105}, "speed": 1, "is_in_view": 1}
{"idx": 151, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 61, "z": 104}, "speed": 1, "is_in_view": 1}
{"idx": 152, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 62, "z": 104}, "speed": 1, "is_in_view": 1}
{"idx": 153, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 62, "z": 103}, "speed": 1, "is_in_view": 1}
{"idx": 154, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 63, "z": 103}, "speed": 1, "is_in_view": 1}
{"idx": 155, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 63, "z": 102}, "speed": 1, "is_in_view": 1}
{"idx": 156, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 64, "z": 102}, "speed": 1, "is_in_view": 1}
{"idx": 157, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 64, "z": 101}, "speed": 1, "is_in_view": 1}
{"idx": 158, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 65, "z": 101}, "speed": 1, "is_in_view": 1}
{"idx": 159, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 65, "z": 100}, "speed": 1, "is_in_view": 1}
{"idx": 160, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 66, "z": 100}, "speed": 1, "is_in_view": 1}
{"idx": 161, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 66, "z": 99}, "speed": 1, "is_in_view": 1}
{"idx": 162, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 67, "z": 99}, "speed": 1, "is_in_view": 1}
{"idx": 163, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 67, "z": 98}, "speed": 1, "is_in_view": 1}
{"idx": 164, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 68, "z": 98}, "speed": 1, "is_in_view": 1}
{"idx": 165, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 69, "z": 98}, "speed": 1, "is_in_view": 1}
{"idx": 166, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 69, "z": 97}, "speed": 1, "is_in_view": 1}
{"idx": 167, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 16, "z": 24}, "speed": 1, "is_in_view": 1}
{"idx": 168, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 17, "z": 24}, "speed": 1, "is_in_view": 1}
{"idx": 169, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 17, "z": 25}, "speed": 1, "is_in_view": 1}
{"idx": 170, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 18, "z": 25}, "speed": 1, "is_in_view": 1}
{"idx": 171, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 18, "z": 26}, "speed": 1, "is_in_view": 1}
{"idx": 172, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 18, "z": 27}, "speed": 1, "is_in_view": 1}
{"idx": 173, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 18, "z": 28}, "speed": 1, "is_in_view": 1}
{"idx": 174, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 19, "z": 28}, "speed": 1, "is_in_view": 1}
{"idx": 175, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 19, "z": 29}, "speed": 1, "is_in_view": 1}
{"idx": 176, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 19, "z": 30}, "speed": 1, "is_in_view": 1}
{"idx": 177, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 20, "z": 30}, "speed": 1, "is_in_view": 1}
{"idx": 178, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 20, "z": 31}, "speed": 1, "is_in_view": 1}
{"idx": 179, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 21, "z": 31}, "speed": 1, "is_in_view": 1}
{"idx": 180, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 21, "z": 32}, "speed": 1, "is_in_view": 1}
{"idx": 181, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 72, "z": 100}, "speed": 1, "is_in_view": 1}
{"idx": 182, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 72, "z": 101}, "speed": 1, "is_in_view": 1}
{"idx": 183, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 72, "z": 102}, "speed": 1, "is_in_view": 1}
{"idx": 184, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 71, "z": 102}, "speed": 1, "is_in_view": 1}
{"idx": 185, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 71, "z": 103}, "speed": 1, "is_in_view": 1}
{"idx": 186, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 71, "z": 104}, "speed": 1, "is_in_view": 1}
{"idx": 187, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 71, "z": 105}, "speed": 1, "is_in_view": 1}
{"idx": 188, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 71, "z": 106}, "speed": 1, "is_in_view": 1}
{"idx": 189, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 70, "z": 106}, "speed": 1, "is_in_view": 1}
{"idx": 190, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 70, "z": 107}, "speed": 1, "is_in_view": 1}
{"idx": 191, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 70, "z": 108}, "speed": 1, "is_in_view": 1}
{"idx": 192, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 71, "z": 108}, "speed": 1, "is_in_view": 1}
{"idx": 193, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 71, "z": 109}, "speed": 1, "is_in_view": 1}
{"idx": 194, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 54, "z": 22}, "speed": 1, "is_in_view": 1}
{"idx": 195, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 55, "z": 22}, "speed": 1, "is_in_view": 1}
{"idx": 196, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 56, "z": 22}, "speed": 1, "is_in_view": 1}
{"idx": 197, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 56, "z": 21}, "speed": 1, "is_in_view": 1}
{"idx": 198, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 57, "z": 21}, "speed": 1, "is_in_view": 1}
{"idx": 199, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 57, "z": 20}, "speed": 1, "is_in_view": 1}
{"idx": 200, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 58, "z": 20}, "speed": 1, "is_in_view": 1}
{"idx": 201, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 59, "z": 20}, "speed": 1, "is_in_view": 1}
{"idx": 202, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 59, "z": 19}, "speed": 1, "is_in_view": 1}
{"idx": 203, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 60, "z": 19}, "speed": 1, "is_in_view": 1}
{"idx": 204, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 61, "z": 19}, "speed": 1, "is_in_view": 1}
{"idx": 205, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 99, "z": 83}, "speed": 1, "is_in_view": 1}
{"idx": 206, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 99, "z": 84}, "speed": 1, "is_in_view": 1}
{"idx": 207, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 98, "z": 86}, "speed": 1, "is_in_view": 1}
{"idx": 208, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 98, "z": 87}, "speed": 1, "is_in_view": 1}
{"idx": 209, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 98, "z": 88}, "speed": 1, "is_in_view": 1}
{"idx": 210, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 97, "z": 88}, "speed": 1, "is_in_view": 1}
{"idx": 211, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 97, "z": 89}, "speed": 1, "is_in_view": 1}
{"idx": 212, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 97, "z": 90}, "speed": 1, "is_in_view": 1}
{"idx": 213, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 96, "z": 90}, "speed": 1, "is_in_view": 1}
{"idx": 214, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 96, "z": 91}, "speed": 1, "is_in_view": 1}
{"idx": 215, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 95, "z": 91}, "speed": 1, "is_in_view": 1}
{"idx": 216, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 95, "z": 92}, "speed": 1, "is_in_view": 1}
{"idx": 217, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 94, "z": 92}, "speed": 1, "is_in_view": 1}
{"idx": 218, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 94, "z": 93}, "speed": 1, "is_in_view": 1}
{"idx": 219, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 93, "z": 93}, "speed": 1, "is_in_view": 1}
{"idx": 220, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 93, "z": 94}, "speed": 1, "is_in_view": 1}
{"idx": 221, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 92, "z": 94}, "speed": 1, "is_in_view": 1}
{"idx": 222, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 92, "z": 95}, "speed": 1, "is_in_view": 1}
{"idx": 223, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 91, "z": 95}, "speed": 1, "is_in_view": 1}
{"idx": 224, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 91, "z": 96}, "speed": 1, "is_in_view": 1}
{"idx": 225, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 90, "z": 96}, "speed": 1, "is_in_view": 1}
{"idx": 226, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 90, "z": 97}, "speed": 1, "is_in_view": 1}
{"idx": 227, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 89, "z": 97}, "speed": 1, "is_in_view": 1}
{"idx": 228, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 89, "z": 98}, "speed": 1, "is_in_view": 1}
{"idx": 229, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 42, "z": 32}, "speed": 1, "is_in_view": 1}
{"idx": 230, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 41, "z": 29}, "speed": 1, "is_in_view": 1}
{"idx": 231, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 41, "z": 28}, "speed": 1, "is_in_view": 1}
{"idx": 232, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 42, "z": 28}, "speed": 1, "is_in_view": 1}
{"idx": 233, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 42, "z": 27}, "speed": 1, "is_in_view": 1}
{"idx": 234, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 41, "z": 27}, "speed": 1, "is_in_view": 1}
{"idx": 235, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 41, "z": 26}, "speed": 1, "is_in_view": 1}
{"idx": 236, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 41, "z": 25}, "speed": 1, "is_in_view": 1}
{"idx": 237, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 40, "z": 25}, "speed": 1, "is_in_view": 1}
{"idx": 238, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 40, "z": 24}, "speed": 1, "is_in_view": 1}
{"idx": 239, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 41, "z": 24}, "speed": 1, "is_in_view": 1}
{"idx": 240, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 41, "z": 23}, "speed": 1, "is_in_view": 1}
{"idx": 241, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 41, "z": 22}, "speed": 1, "is_in_view": 1}
{"idx": 242, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 42, "z": 22}, "speed": 1, "is_in_view": 1}
{"idx": 243, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 42, "z": 21}, "speed": 1, "is_in_view": 1}
{"idx": 244, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 49, "z": 102}, "speed": 1, "is_in_view": 1}
{"idx": 245, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 50, "z": 101}, "speed": 1, "is_in_view": 1}
{"idx": 246, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 50, "z": 100}, "speed": 1, "is_in_view": 1}
{"idx": 247, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 51, "z": 100}, "speed": 1, "is_in_view": 1}
{"idx": 248, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 51, "z": 99}, "speed": 1, "is_in_view": 1}
{"idx": 249, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 52, "z": 99}, "speed": 1, "is_in_view": 1}
{"idx": 250, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 52, "z": 98}, "speed": 1, "is_in_view": 1}
{"idx": 251, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 53, "z": 98}, "speed": 1, "is_in_view": 1}
{"idx": 252, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 53, "z": 97}, "speed": 1, "is_in_view": 1}
{"idx": 253, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 54, "z": 97}, "speed": 1, "is_in_view": 1}
{"idx": 254, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 54, "z": 96}, "speed": 1, "is_in_view": 1}
{"idx": 255, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 54, "z": 95}, "speed": 1, "is_in_view": 1}
{"idx": 256, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 55, "z": 95}, "speed": 1, "is_in_view": 1}
{"idx": 257, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 55, "z": 94}, "speed": 1, "is_in_view": 1}
{"idx": 258, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 56, "z": 94}, "speed": 1, "is_in_view": 1}
{"idx": 259, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 56, "z": 93}, "speed": 1, "is_in_view": 1}
{"idx": 260, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 55, "z": 93}, "speed": 1, "is_in_view": 1}
{"idx": 261, "hero_l2_distance": 0, "hero_relative_direction": 7, "monster_id": 14, "monster_interval": 300, "pos": {"x": 55, "z": 92}, "speed": 1, "is_in_view": 1}
{"idx": 262, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 114, "z": 111}, "speed": 1, "is_in_view": 1}
{"idx": 263, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 113, "z": 111}, "speed": 1, "is_in_view": 1}
{"idx": 264, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 113, "z": 110}, "speed": 1, "is_in_view": 1}
{"idx": 265, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 112, "z": 110}, "speed": 1, "is_in_view": 1}
{"idx": 266, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 112, "z": 111}, "speed": 1, "is_in_view": 1}
{"idx": 267, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 111, "z": 111}, "speed": 1, "is_in_view": 1}
{"idx": 268, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 111, "z": 110}, "speed": 1, "is_in_view": 1}
{"idx": 269, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 110, "z": 110}, "speed": 1, "is_in_view": 1}
{"idx": 270, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 109, "z": 110}, "speed": 1, "is_in_view": 1}
{"idx": 271, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 109, "z": 111}, "speed": 1, "is_in_view": 1}
{"idx": 272, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 108, "z": 111}, "speed": 1, "is_in_view": 1}
{"idx": 273, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 92, "z": 93}, "speed": 1, "is_in_view": 1}
{"idx": 274, "hero_l2_distance": 0, "hero_relative_direction": 0, "monster_id": 14, "monster_interval": 300, "pos": {"x": 91, "z": 93}, "speed": 1, "is_in_view": 1}
{"idx": 275, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 48, "z": 105}, "speed": 1, "is_in_view": 1}
{"idx": 276, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 48, "z": 104}, "speed": 1, "is_in_view": 1}
{"idx": 277, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 47, "z": 104}, "speed": 1, "is_in_view": 1}
{"idx": 278, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 47, "z": 103}, "speed": 1, "is_in_view": 1}
{"idx": 279, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 47, "z": 102}, "speed": 1, "is_in_view": 1}
{"idx": 280, "hero_l2_distance": 0, "hero_relative_direction": 8, "monster_id": 14, "monster_interval": 300, "pos": {"x": 47, "z": 101}, "speed": 1, "is_in_view": 1}
{"idx": 281, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 48, "z": 101}, "speed": 1, "is_in_view": 1}
{"idx": 282, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 48, "z": 100}, "speed": 1, "is_in_view": 1}
{"idx": 283, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 49, "z": 100}, "speed": 1, "is_in_view": 1}
{"idx": 284, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 49, "z": 99}, "speed": 1, "is_in_view": 1}
{"idx": 285, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 50, "z": 99}, "speed": 1, "is_in_view": 1}
{"idx": 286, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 21, "z": 80}, "speed": 1, "is_in_view": 1}
{"idx": 287, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 21, "z": 81}, "speed": 1, "is_in_view": 1}
{"idx": 288, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": -1, "z": -1}, "speed": -1, "is_in_view": 0}
{"idx": 289, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 22, "z": 82}, "speed": 1, "is_in_view": 1}
{"idx": 290, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 23, "z": 82}, "speed": 1, "is_in_view": 1}
{"idx": 291, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 23, "z": 83}, "speed": 1, "is_in_view": 1}
{"idx": 292, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 24, "z": 83}, "speed": 1, "is_in_view": 1}
{"idx": 293, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 24, "z": 84}, "speed": 1, "is_in_view": 1}
{"idx": 294, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 25, "z": 84}, "speed": 1, "is_in_view": 1}
{"idx": 295, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 25, "z": 85}, "speed": 1, "is_in_view": 1}
{"idx": 296, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 25, "z": 86}, "speed": 1, "is_in_view": 1}
{"idx": 297, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 25, "z": 87}, "speed": 1, "is_in_view": 1}
{"idx": 298, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 25, "z": 88}, "speed": 1, "is_in_view": 1}
{"idx": 299, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 24, "z": 88}, "speed": 1, "is_in_view": 1}
{"idx": 300, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 24, "z": 89}, "speed": 1, "is_in_view": 1}
{"idx": 301, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 23, "z": 89}, "speed": 1, "is_in_view": 1}
{"idx": 302, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 23, "z": 90}, "speed": 1, "is_in_view": 1}
{"idx": 303, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 22, "z": 90}, "speed": 1, "is_in_view": 1}
{"idx": 304, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 22, "z": 91}, "speed": 1, "is_in_view": 1}
{"idx": 305, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 21, "z": 91}, "speed": 1, "is_in_view": 1}
{"idx": 306, "hero_l2_distance": 0, "hero_relative_direction": 1, "monster_id": 14, "monster_interval": 300, "pos": {"x": 21, "z": 92}, "speed": 1, "is_in_view": 1}
{"idx": 307, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 101, "z": 18}, "speed": 1, "is_in_view": 1}
{"idx": 308, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 103, "z": 18}, "speed": 1, "is_in_view": 1}
{"idx": 309, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 104, "z": 18}, "speed": 1, "is_in_view": 1}
{"idx": 310, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 104, "z": 19}, "speed": 1, "is_in_view": 1}
{"idx": 311, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 105, "z": 19}, "speed": 1, "is_in_view": 1}
{"idx": 312, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 106, "z": 19}, "speed": 1, "is_in_view": 1}
{"idx": 313, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 106, "z": 20}, "speed": 1, "is_in_view": 1}
{"idx": 314, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 107, "z": 20}, "speed": 1, "is_in_view": 1}
{"idx": 315, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 108, "z": 20}, "speed": 1, "is_in_view": 1}
{"idx": 316, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 108, "z": 19}, "speed": 1, "is_in_view": 1}
{"idx": 317, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 109, "z": 19}, "speed": 1, "is_in_view": 1}
{"idx": 318, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 110, "z": 19}, "speed": 1, "is_in_view": 1}
{"idx": 319, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 111, "z": 19}, "speed": 1, "is_in_view": 1}
{"idx": 320, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 93, "z": 60}, "speed": 1, "is_in_view": 1}
{"idx": 321, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 93, "z": 61}, "speed": 1, "is_in_view": 1}
{"idx": 322, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 91, "z": 70}, "speed": 1, "is_in_view": 1}
{"idx": 323, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 91, "z": 71}, "speed": 1, "is_in_view": 1}
{"idx": 324, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 90, "z": 71}, "speed": 1, "is_in_view": 1}
{"idx": 325, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 90, "z": 72}, "speed": 1, "is_in_view": 1}
{"idx": 326, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 89, "z": 72}, "speed": 1, "is_in_view": 1}
{"idx": 327, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 89, "z": 73}, "speed": 1, "is_in_view": 1}
{"idx": 328, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 88, "z": 73}, "speed": 1, "is_in_view": 1}
{"idx": 329, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 88, "z": 74}, "speed": 1, "is_in_view": 1}
{"idx": 330, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 87, "z": 74}, "speed": 1, "is_in_view": 1}
{"idx": 331, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 87, "z": 75}, "speed": 1, "is_in_view": 1}
{"idx": 332, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 88, "z": 75}, "speed": 1, "is_in_view": 1}
{"idx": 333, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 88, "z": 76}, "speed": 1, "is_in_view": 1}
{"idx": 334, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 89, "z": 76}, "speed": 1, "is_in_view": 1}
{"idx": 335, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 89, "z": 77}, "speed": 1, "is_in_view": 1}
{"idx": 336, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 89, "z": 78}, "speed": 1, "is_in_view": 1}
{"idx": 337, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 88, "z": 78}, "speed": 1, "is_in_view": 1}
{"idx": 338, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 88, "z": 79}, "speed": 1, "is_in_view": 1}
{"idx": 339, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 87, "z": 79}, "speed": 1, "is_in_view": 1}
{"idx": 340, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 87, "z": 80}, "speed": 1, "is_in_view": 1}
{"idx": 341, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 87, "z": 81}, "speed": 1, "is_in_view": 1}
{"idx": 342, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 88, "z": 81}, "speed": 1, "is_in_view": 1}
{"idx": 343, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 88, "z": 82}, "speed": 1, "is_in_view": 1}
{"idx": 344, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 90, "z": 43}, "speed": 1, "is_in_view": 1}
{"idx": 345, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 89, "z": 44}, "speed": 1, "is_in_view": 1}
{"idx": 346, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 86, "z": 45}, "speed": 1, "is_in_view": 1}
{"idx": 347, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 85, "z": 45}, "speed": 1, "is_in_view": 1}
{"idx": 348, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 85, "z": 46}, "speed": 1, "is_in_view": 1}
{"idx": 349, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 84, "z": 46}, "speed": 1, "is_in_view": 1}
{"idx": 350, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 84, "z": 47}, "speed": 1, "is_in_view": 1}
{"idx": 351, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 84, "z": 48}, "speed": 1, "is_in_view": 1}
{"idx": 352, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 84, "z": 49}, "speed": 1, "is_in_view": 1}
{"idx": 353, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 84, "z": 50}, "speed": 1, "is_in_view": 1}
{"idx": 354, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 84, "z": 51}, "speed": 1, "is_in_view": 1}
{"idx": 355, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 85, "z": 51}, "speed": 1, "is_in_view": 1}
{"idx": 356, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 85, "z": 52}, "speed": 1, "is_in_view": 1}
{"idx": 357, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 86, "z": 52}, "speed": 1, "is_in_view": 1}
{"idx": 358, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 86, "z": 53}, "speed": 1, "is_in_view": 1}
{"idx": 359, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 87, "z": 53}, "speed": 1, "is_in_view": 1}
{"idx": 360, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 87, "z": 54}, "speed": 1, "is_in_view": 1}
{"idx": 361, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 88, "z": 54}, "speed": 1, "is_in_view": 1}
{"idx": 362, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 88, "z": 53}, "speed": 1, "is_in_view": 1}
{"idx": 363, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 89, "z": 53}, "speed": 1, "is_in_view": 1}
{"idx": 364, "hero_l2_distance": 0, "hero_relative_direction": 5, "monster_id": 14, "monster_interval": 300, "pos": {"x": 89, "z": 54}, "speed": 1, "is_in_view": 1}
{"idx": 365, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 90, "z": 54}, "speed": 1, "is_in_view": 1}
{"idx": 366, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 90, "z": 53}, "speed": 1, "is_in_view": 1}
{"idx": 367, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 91, "z": 53}, "speed": 1, "is_in_view": 1}
{"idx": 368, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 43, "z": 66}, "speed": 1, "is_in_view": 1}
{"idx": 369, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 47, "z": 62}, "speed": 1, "is_in_view": 1}
{"idx": 370, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 48, "z": 62}, "speed": 1, "is_in_view": 1}
{"idx": 371, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 48, "z": 61}, "speed": 1, "is_in_view": 1}
{"idx": 372, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 49, "z": 61}, "speed": 1, "is_in_view": 1}
{"idx": 373, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 50, "z": 61}, "speed": 1, "is_in_view": 1}
{"idx": 374, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 51, "z": 61}, "speed": 1, "is_in_view": 1}
{"idx": 375, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 52, "z": 61}, "speed": 1, "is_in_view": 1}
{"idx": 376, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 55, "z": 61}, "speed": 1, "is_in_view": 1}
{"idx": 377, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 56, "z": 61}, "speed": 1, "is_in_view": 1}
{"idx": 378, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 57, "z": 61}, "speed": 1, "is_in_view": 1}
{"idx": 379, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 58, "z": 61}, "speed": 1, "is_in_view": 1}
{"idx": 380, "hero_l2_distance": 0, "hero_relative_direction": 4, "monster_id": 14, "monster_interval": 300, "pos": {"x": 58, "z": 62}, "speed": 1, "is_in_view": 1}
{"idx": 381, "hero_l2_distance": 0, "hero_relative_direction": 3, "monster_id": 14, "monster_interval": 300, "pos": {"x": 59, "z": 62}, "speed": 1, "is_in_view": 1}
{"idx": 382, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 60, "z": 62}, "speed": 1, "is_in_view": 1}
{"idx": 383, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 60, "z": 63}, "speed": 1, "is_in_view": 1}
{"idx": 384, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 61, "z": 63}, "speed": 1, "is_in_view": 1}
{"idx": 385, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 61, "z": 64}, "speed": 1, "is_in_view": 1}
{"idx": 386, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 62, "z": 64}, "speed": 1, "is_in_view": 1}
{"idx": 387, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 62, "z": 65}, "speed": 1, "is_in_view": 1}
{"idx": 388, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 63, "z": 65}, "speed": 1, "is_in_view": 1}
{"idx": 389, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 63, "z": 66}, "speed": 1, "is_in_view": 1}
{"idx": 390, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 64, "z": 66}, "speed": 1, "is_in_view": 1}
{"idx": 391, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 64, "z": 67}, "speed": 1, "is_in_view": 1}
{"idx": 392, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 64, "z": 68}, "speed": 1, "is_in_view": 1}
{"idx": 393, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 64, "z": 69}, "speed": 1, "is_in_view": 1}
{"idx": 394, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 64, "z": 70}, "speed": 1, "is_in_view": 1}
{"idx": 395, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 63, "z": 70}, "speed": 1, "is_in_view": 1}
{"idx": 396, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 63, "z": 71}, "speed": 1, "is_in_view": 1}
{"idx": 397, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 62, "z": 71}, "speed": 1, "is_in_view": 1}
{"idx": 398, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 62, "z": 72}, "speed": 1, "is_in_view": 1}
{"idx": 399, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 61, "z": 72}, "speed": 1, "is_in_view": 1}
{"idx": 400, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 61, "z": 73}, "speed": 1, "is_in_view": 1}
{"idx": 401, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 60, "z": 73}, "speed": 1, "is_in_view": 1}
{"idx": 402, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 60, "z": 74}, "speed": 1, "is_in_view": 1}
{"idx": 403, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 59, "z": 74}, "speed": 1, "is_in_view": 1}
{"idx": 404, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 59, "z": 75}, "speed": 1, "is_in_view": 1}
{"idx": 405, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 58, "z": 75}, "speed": 1, "is_in_view": 1}
{"idx": 406, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 58, "z": 76}, "speed": 1, "is_in_view": 1}
{"idx": 407, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 57, "z": 76}, "speed": 1, "is_in_view": 1}
{"idx": 408, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 57, "z": 77}, "speed": 1, "is_in_view": 1}
{"idx": 409, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 56, "z": 77}, "speed": 1, "is_in_view": 1}
{"idx": 410, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 56, "z": 78}, "speed": 1, "is_in_view": 1}
{"idx": 411, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 55, "z": 78}, "speed": 1, "is_in_view": 1}
{"idx": 412, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 55, "z": 79}, "speed": 1, "is_in_view": 1}
{"idx": 413, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 54, "z": 79}, "speed": 1, "is_in_view": 1}
{"idx": 414, "hero_l2_distance": 0, "hero_relative_direction": 2, "monster_id": 14, "monster_interval": 300, "pos": {"x": 54, "z": 80}, "speed": 1, "is_in_view": 1}
{"idx": 415, "hero_l2_distance": 0, "hero_relative_direction": 6, "monster_id": 14, "monster_interval": 300, "pos": {"x": 38, "z": 81}, "speed": 1, "is_in_view": 1}
```

</details>
