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

        # Check ALL possible env var names (Vercel uses NEXT_PUBLIC_ prefix)
        self.supabase_url = (
            os.environ.get('SUPABASE_URL', '')
            or os.environ.get('NEXT_PUBLIC_SUPABASE_URL', '')
        ).rstrip('/')

        self.supabase_key = (
            os.environ.get('SUPABASE_SERVICE_ROLE_KEY', '')
            or os.environ.get('SUPABASE_ANON_KEY', '')
            or os.environ.get('NEXT_PUBLIC_SUPABASE_ANON_KEY', '')
            or os.environ.get('NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY', '')
        )

        self.use_supabase = bool(self.supabase_url and self.supabase_key)
        if self.use_supabase:
            print(f'[ParkNav AI] Supabase enabled: {self.supabase_url} (key: {self.supabase_key[:12]}...)')
        else:
            print(f'[ParkNav AI] Supabase NOT configured. Using local file storage.')

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
            try:
                self._save_supabase_value(key, data)
                return
            except Exception as e:
                print(f'[ParkNav AI] Supabase save failed for {key}, falling back to local: {e}')
                # Fall through to local save
        self._save_file(path, data)

    def _key_for_path(self, path):
        return self.PATH_KEYS.get(os.path.basename(path))

    def _load_file(self, path, default=None):
        # Try the original path first, then /tmp fallback (for Vercel)
        for p in [path, os.path.join('/tmp', os.path.basename(path))]:
            if os.path.exists(p):
                try:
                    with open(p, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass
        return default

    def _save_file(self, path, data):
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except (IOError, OSError):
            # Vercel has read-only filesystem, use /tmp as fallback
            tmp_path = os.path.join('/tmp', os.path.basename(path))
            try:
                with open(tmp_path, 'w') as f:
                    json.dump(data, f, indent=2)
            except (IOError, OSError) as e:
                print(f'[ParkNav AI] Cannot write to {path} or {tmp_path}: {e}')

    def _headers(self):
        headers = {
            'apikey': self.supabase_key,
            'Content-Type': 'application/json',
            # ALWAYS send Authorization header - PostgREST requires it
            'Authorization': f'Bearer {self.supabase_key}',
        }
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
            with urllib.request.urlopen(req, timeout=5) as resp:
                rows = json.loads(resp.read().decode('utf-8'))
            if rows:
                return rows[0].get('value')
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            print(f'[ParkNav AI] Supabase read failed for {key}; using local fallback: {e}')
        return None

    def _save_supabase_value(self, key, value):
        payload = json.dumps({
            'key': key,
            'value': value,
            'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }).encode('utf-8')

        headers = self._headers()
        headers['Prefer'] = 'resolution=merge-duplicates,return=minimal'

        url = self._state_url(upsert=True)
        req = urllib.request.Request(
            url,
            data=payload,
            headers=headers,
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8')
            print(f'[ParkNav AI] Supabase HTTP Error {e.code} for {key}: {body}')
            raise Exception(f"Supabase Error {e.code}: {body}")
        except (urllib.error.URLError, TimeoutError) as e:
            print(f'[ParkNav AI] Supabase connection failed for {key}: {e}')
            raise Exception(f"Connection to Supabase failed: {e}")
