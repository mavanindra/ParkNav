import datetime
import json
import os
import urllib.error
import urllib.parse
import urllib.request


class ParkNavStorage:
    """Persistence layer with local JSON fallback and optional Supabase storage."""

    PATH_KEYS = {
        'parking_reports.json': 'parking_reports',
        'known_parking_spots.json': 'known_parking_spots',
        'qtable.json': 'qtable',
    }

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.supabase_url = os.environ.get('SUPABASE_URL', '').rstrip('/')
        self.supabase_key = (
            os.environ.get('SUPABASE_SERVICE_ROLE_KEY', '')
            or os.environ.get('SUPABASE_ANON_KEY', '')
        )
        self.use_supabase = bool(self.supabase_url and self.supabase_key)

    def load_json(self, path, default=None):
        key = self._key_for_path(path)
        if default is None and key != 'qtable':
            default = []
        if self.use_supabase and key:
            data = self._load_supabase_value(key)
            if data is not None:
                return data
        return self._load_file(path, default)

    def save_json(self, path, data):
        key = self._key_for_path(path)
        if self.use_supabase and key:
            self._save_supabase_value(key, data)
            return
        self._save_file(path, data)

    def _key_for_path(self, path):
        return self.PATH_KEYS.get(os.path.basename(path))

    def _load_file(self, path, default=None):
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return default

    def _save_file(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _headers(self):
        headers = {
            'apikey': self.supabase_key,
            'Content-Type': 'application/json',
        }
        # Supabase secret keys (sb_secret_...) are not JWTs and should not be
        # sent as Bearer tokens. Legacy service_role keys still expect it.
        if not self.supabase_key.startswith('sb_'):
            headers['Authorization'] = f'Bearer {self.supabase_key}'
        return headers

    def _state_url(self, key=None, upsert=False):
        base = f'{self.supabase_url}/rest/v1/parknav_state'
        if upsert:
            return f'{base}?on_conflict=key'
        query = urllib.parse.urlencode({
            'select': 'value',
            'key': f'eq.{key}',
        })
        return f'{base}?{query}'

    def _load_supabase_value(self, key):
        req = urllib.request.Request(self._state_url(key), headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                rows = json.loads(resp.read().decode('utf-8'))
            if rows:
                return rows[0].get('value')
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f'[ParkNav AI] Supabase read failed for {key}; using local fallback: {e}')
        return None

    def _save_supabase_value(self, key, value):
        payload = json.dumps({
            'key': key,
            'value': value,
            'updated_at': datetime.datetime.now(datetime.UTC).isoformat(),
        }).encode('utf-8')
        headers = self._headers()
        headers['Prefer'] = 'resolution=merge-duplicates,return=minimal'
        req = urllib.request.Request(
            self._state_url(upsert=True),
            data=payload,
            headers=headers,
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                return
        except (urllib.error.URLError, TimeoutError) as e:
            print(f'[ParkNav AI] Supabase write failed for {key}: {e}')
            raise
