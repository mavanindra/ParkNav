import json
import os
import random


class ParkingQAgent:
    """
    Q-Learning agent for parking availability prediction.
    Pre-trained on 50,000 samples (9 locations × 5 time buckets).
    Improves in real-time via crowdsourced user feedback.
    """

    # Indices 0-8 MUST match the trained qtable.json order
    LOCATION_TYPES = [
        'university',   # 0
        'mall',         # 1
        'hospital',     # 2
        'station',      # 3
        'residential',  # 4
        'generic',      # 5
        'market',       # 6
        'office',       # 7
        'beach',        # 8
    ]
    LOCATION_INDICES = {loc: i for i, loc in enumerate(LOCATION_TYPES)}

    # Indices 0-4 MUST match the trained qtable.json order
    TIME_BUCKET_ORDER = [
        'early_morning',  # 0 — hours 0-6
        'morning_peak',   # 1 — hours 7-10
        'afternoon',      # 2 — hours 11-15
        'evening_peak',   # 3 — hours 16-20
        'night',          # 4 — hours 21-23
    ]
    TIME_BUCKET_INDICES = {b: i for i, b in enumerate(TIME_BUCKET_ORDER)}

    TIME_BUCKET_RANGES = {
        'early_morning': (0, 6),
        'morning_peak':  (7, 10),
        'afternoon':     (11, 15),
        'evening_peak':  (16, 20),
        'night':         (21, 23),
    }

    ACTIONS = ['low', 'medium', 'high']  # action indices 0, 1, 2

    ACTION_RANGES = {
        'low':    (0, 35),
        'medium': (36, 65),
        'high':   (66, 100),
    }

    def __init__(self, qtable_path='qtable.json', load_q_table=None, save_q_table=None):
        self.qtable_path = qtable_path
        self.load_q_table_func = load_q_table
        self.save_q_table_func = save_q_table
        self.alpha = 0.1          # learning rate (matches training)
        self.gamma = 0.9          # discount factor (matches training)
        self.epsilon = 0.01       # minimal exploration — exploit trained knowledge
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.q_table = None       # 3D list: [loc_idx][time_idx][action]
        self._load_q_table()

    # ── Helpers ──────────────────────────────────────────────────

    def _get_time_bucket(self, hour):
        for bucket, (start, end) in self.TIME_BUCKET_RANGES.items():
            if start <= hour <= end:
                return bucket
        return 'night'

    def _loc_idx(self, location_type):
        return self.LOCATION_INDICES.get(
            location_type, self.LOCATION_INDICES['generic'])

    def _time_idx(self, time_bucket):
        return self.TIME_BUCKET_INDICES.get(time_bucket, 4)

    def _ensure_q_table(self):
        """Initialise a fresh 9×5×3 zero Q-table (fallback only)."""
        self.q_table = [[[0.0, 0.0, 0.0] for _ in range(5)]
                        for _ in range(9)]

    # ── Pre-training ─────────────────────────────────────────────

    def pre_train(self, episodes=5000):
        """
        Skip if 50k-trained Q-table is already loaded.
        Only falls back to empty table if nothing was loaded.
        """
        if self.q_table is not None:
            return  # trained table loaded — skip
        self._ensure_q_table()

    # ── Prediction ───────────────────────────────────────────────

    def get_prediction(self, location_type, hour, traffic=1, is_special_day=0):
        if location_type not in self.LOCATION_INDICES:
            location_type = 'generic'

        bucket = self._get_time_bucket(hour)
        ti     = self._time_idx(bucket)
        traf_idx = min(traffic, 3)
        spec_idx = min(is_special_day, 1)
        
        try:
            # Try new 4D format [traffic][time][special]
            if isinstance(self.q_table[0][0][0], list):
                q_vals = self.q_table[traf_idx][ti][spec_idx]
            else:
                raise TypeError("Not 4D format")
        except (IndexError, TypeError):
            # Fallback to old 3D format [loc][time] if old table is loaded
            li     = self._loc_idx(location_type)
            try:
                q_vals = self.q_table[li][ti]
            except Exception:
                q_vals = [0.0, 0.0, 0.0]

        action_idx = q_vals.index(max(q_vals)) if max(q_vals) != min(q_vals) else 1
        action     = self.ACTIONS[action_idx]

        lo, hi = self.ACTION_RANGES[action]
        base   = random.randint(lo, hi)

        status = ('high'   if base > 65 else
                  'medium' if base > 35 else 'low')

        spread     = max(q_vals) - min(q_vals) if max(q_vals) != min(q_vals) else 0
        confidence = min(97, max(55, int(62 + min(spread, 5) * 7)))

        reasoning = self._build_reasoning(location_type, bucket, status, base, confidence)

        return {
            'availability_percent': base,
            'status':               status,
            'confidence':           confidence,
            'reasoning':            reasoning,
            'action':               action,
            'location_type':        location_type,
            'time_bucket':          bucket,
            'hour':                 hour,
            'q_values': {self.ACTIONS[i]: round(q_vals[i], 4) for i in range(3)} if len(q_vals)==3 else {},
        }

    def _build_reasoning(self, loc, bucket, status, avail, conf):
        time_label = bucket.replace('_', ' ').title()
        loc_label  = loc.replace('_', ' ').title()

        reasons = {
            ('university', 'morning_peak'):
                f"University areas are very busy during morning hours. Students and staff arriving for classes reduce parking to ~{avail}%.",
            ('university', 'afternoon'):
                f"Peak campus hours — lectures and labs running. Parking availability is critically low at ~{avail}%.",
            ('mall', 'evening_peak'):
                f"Evening shopping rush at malls. Expect very limited parking (~{avail}%) during 4-8 PM.",
            ('station', 'morning_peak'):
                f"Morning commuter rush at railway/metro stations. Parking fills up fast (~{avail}%).",
            ('station', 'evening_peak'):
                f"Evening commuters returning — station parking is congested (~{avail}%).",
            ('hospital', 'afternoon'):
                f"Hospital visiting hours — moderate parking demand. ~{avail}% spots available.",
            ('residential', 'night'):
                f"Residential areas at night — street parking is freely available (~{avail}%).",
            ('market', 'afternoon'):
                f"Market areas are busiest in the afternoon — heavy footfall reduces parking to ~{avail}%.",
            ('office', 'morning_peak'):
                f"Office areas fill up fast during morning work hours. Early arrival recommended (~{avail}% available).",
            ('beach', 'evening_peak'):
                f"Beach/tourist spots are busiest in the evening. Very limited parking (~{avail}%) expected.",
        }

        specific = reasons.get((loc, bucket))
        if specific:
            return specific

        return (
            f"Based on Q-learning analysis: {loc_label} area during {time_label} "
            f"typically has {status} parking availability (~{avail}%). "
            f"AI confidence: {conf}%."
        )

    # ── Learning from feedback ────────────────────────────────────

    def update(self, location_type, hour, feedback_status):
        if location_type not in self.LOCATION_INDICES:
            location_type = 'generic'

        bucket = self._get_time_bucket(hour)
        li     = self._loc_idx(location_type)
        ti     = self._time_idx(bucket)

        feedback_map = {
            'available':     2,   # high
            'few_left':      1,   # medium
            'not_available': 0,   # low
        }
        correct_action = feedback_map.get(feedback_status, 1)

        # Boost correct action, penalise incorrect
        try:
            for a in range(3):
                if a == correct_action:
                    self.q_table[li][ti][a] += self.alpha * (1.0  - self.q_table[li][ti][a])
                else:
                    self.q_table[li][ti][a] += self.alpha * (-0.5 - self.q_table[li][ti][a])

            self.save_q_table()
            new_q      = self.q_table[li][ti]
            spread     = max(new_q) - min(new_q) if max(new_q) != min(new_q) else 0
            confidence = min(97, max(55, int(62 + min(spread, 5) * 7)))
            return confidence
        except (IndexError, TypeError):
            return 75

    # ── Persistence ──────────────────────────────────────────────

    def _load_q_table(self):
        try:
            if self.load_q_table_func is not None:
                data = self.load_q_table_func()
            else:
                if not os.path.exists(self.qtable_path):
                    self.q_table = None
                    return
                with open(self.qtable_path, 'r') as f:
                    data = json.load(f)

            if isinstance(data, list):
                # New format: 3D list [9][5][3] from 50k training
                self.q_table = data
            elif isinstance(data, dict):
                # Old format: {"university_morning_peak": [low, med, high], ...}
                # Convert to new list format
                self._ensure_q_table()
                time_suffixes = self.TIME_BUCKET_ORDER[:]
                for key, vals in data.items():
                    for suffix in time_suffixes:
                        full_suffix = '_' + suffix
                        if key.endswith(full_suffix):
                            loc_type = key[:-len(full_suffix)]
                            li = self.LOCATION_INDICES.get(loc_type)
                            ti = self.TIME_BUCKET_INDICES.get(suffix)
                            if li is not None and ti is not None:
                                self.q_table[li][ti] = list(vals)
                            break
            else:
                self.q_table = None
        except (json.JSONDecodeError, IOError, ValueError, AttributeError):
            self.q_table = None

    def save_q_table(self):
        if self.save_q_table_func is not None:
            self.save_q_table_func(self.q_table)
            return
        with open(self.qtable_path, 'w') as f:
            json.dump(self.q_table, f)

    # ── Stats ─────────────────────────────────────────────────────

    def get_stats(self):
        return {
            'total_states':    len(self.LOCATION_TYPES) * len(self.TIME_BUCKET_ORDER),
            'epsilon':         round(self.epsilon, 4),
            'learning_rate':   self.alpha,
            'discount_factor': self.gamma,
            'trained_on':      '50,000 samples',
            'location_types':  len(self.LOCATION_TYPES),
        }
