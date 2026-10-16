"""Read-only monthly momentum screening; only Python's standard library."""
import datetime as dt
import gzip
import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
INSTRUMENT_URL = 'https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz'
HISTORY_URL = 'https://api.upstox.com/v3/historical-candle/{key}/months/1/{today}/2000-01-01'
_scan_lock = threading.Lock()
_instruments = None


class ScreenerError(ValueError):
    pass


def now_ist():
    return dt.datetime.now(IST)


def status():
    return {'configured': bool(os.environ.get('UPSTOX_ACCESS_TOKEN', '').strip()),
            'provider': 'Upstox', 'read_only': True, 'live_quotes': False,
            'rsi_period': 14, 'time_zone': 'Asia/Kolkata'}


def wilder_rsi(closes, period=14):
    """Wilder seed = first 14 changes; flat RSI convention is 50."""
    result = [None] * len(closes)
    if len(closes) <= period:
        return result
    changes = [b - a for a, b in zip(closes, closes[1:])]
    gain = sum(max(c, 0) for c in changes[:period]) / period
    loss = sum(max(-c, 0) for c in changes[:period]) / period

    def value():
        return 50.0 if gain == loss == 0 else (100.0 if loss == 0 else 100 - 100 / (1 + gain / loss))

    result[period] = value()
    for i in range(period + 1, len(closes)):
        change = changes[i - 1]
        gain = (gain * (period - 1) + max(change, 0)) / period
        loss = (loss * (period - 1) + max(-change, 0)) / period
        result[i] = value()
    return result


def classify(previous, current):
    if previous <= 70 < current:
        return 'fresh'
    if current > 70:
        return 'strong'
    if current < 69.3:
        return 'exit'
    return 'neutral'


def month_number(value):
    return value.year * 12 + value.month - 1


def month_label(number):
    year, month = divmod(number, 12)
    return f'{year:04d}-{month + 1:02d}'


def validate_candles(candles, now=None):
    now = now or now_ist()
    current_month = month_number(now.astimezone(IST))
    if not isinstance(candles, list) or not candles:
        raise ScreenerError('No historical candles returned.')
    indexed = {}
    for candle in candles:
        try:
            if not isinstance(candle, list) or len(candle) < 6:
                raise ValueError()
            timestamp = dt.datetime.fromisoformat(str(candle[0]).replace('Z', '+00:00'))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=IST)
            month = month_number(timestamp.astimezone(IST))
            if any(isinstance(v, bool) for v in candle[1:5]):
                raise ValueError()
            open_, high, low, close = map(float, candle[1:5])
            if not all(math.isfinite(v) and v > 0 for v in (open_, high, low, close)):
                raise ValueError()
            if low > min(open_, close) or high < max(open_, close) or low > high:
                raise ValueError()
        except (ValueError, TypeError, OverflowError):
            raise ScreenerError('Invalid candle timestamp or OHLC data.') from None
        if month > current_month:
            raise ScreenerError('Future candle month returned.')
        if month in indexed:
            raise ScreenerError('Duplicate candle month returned.')
        indexed[month] = {'month': month_label(month), 'number': month,
                          'open': open_, 'high': high, 'low': low, 'close': close}
    ordered = [indexed[m] for m in sorted(indexed)]
    if any(b['number'] != a['number'] + 1 for a, b in zip(ordered, ordered[1:])):
        raise ScreenerError('Missing month in historical candles; RSI cannot be trusted.')
    completed = [c for c in ordered if c['number'] < current_month]
    if not completed or completed[-1]['number'] != current_month - 1:
        raise ScreenerError('Stale history: latest completed month is missing.')
    if len(completed) < 16:
        raise ScreenerError('At least 16 completed monthly closes are required.')
    return completed, ordered


def analyze(symbol, name, candles, now=None):
    completed, all_candles = validate_candles(candles, now)
    rsis = wilder_rsi([c['close'] for c in completed])
    current, previous = rsis[-1], rsis[-2]
    crossings = [i for i in range(15, len(rsis)) if rsis[i - 1] <= 70 < rsis[i]]
    crossing = crossings[-1] if crossings else None
    signal_candle = completed[crossing] if crossing is not None else None
    warning = ['Corporate-action adjustment is unverified. Breakout uses monthly highs, not a live quote.']
    if len(completed) < 60:
        warning.append('Fewer than 60 completed months; RSI seed sensitivity may be material.')
    return {'symbol': symbol, 'name': name, 'close': completed[-1]['close'],
            'month': completed[-1]['month'], 'rsi': current, 'previous_rsi': previous,
            'signal': classify(previous, current),
            'signal_month': signal_candle['month'] if signal_candle else None,
            'signal_high': signal_candle['high'] if signal_candle else None,
            'previous_low': completed[-2]['low'],
            'breakout': bool(signal_candle and any(c['number'] > signal_candle['number'] and
                              c['high'] > signal_candle['high'] for c in all_candles)),
            'history': [{'month': c['month'], 'close': c['close'], 'rsi': r}
                        for c, r in zip(completed, rsis)], 'warning': ' '.join(warning)}


def fetch_json(url, token=None, compressed=False):
    headers = {'Accept': 'application/json', 'User-Agent': 'LocalMomentumScreener/1.0'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    # Never follow upstream redirects carrying an Authorization header.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
            payload = response.read(32 * 1024 * 1024 + 1)
        if len(payload) > 32 * 1024 * 1024:
            raise ScreenerError('Provider response exceeds size limit.')
        if compressed:
            import io
            with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
                payload = stream.read(64 * 1024 * 1024 + 1)
            if len(payload) > 64 * 1024 * 1024:
                raise ScreenerError('Instrument directory exceeds size limit.')
        return json.loads(payload)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ScreenerError('Upstox authentication failed; check the server environment token.') from None
        if exc.code == 429:
            raise ScreenerError('Upstox rate limit reached; retry later.') from None
        raise ScreenerError(f'Provider HTTP error ({exc.code}).') from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        raise ScreenerError('Provider request failed or returned invalid data.') from None


def instruments():
    global _instruments
    if _instruments is None:
        raw = fetch_json(INSTRUMENT_URL, compressed=True)
        if not isinstance(raw, list):
            raise ScreenerError('Invalid instrument directory response.')
        found = {}
        for item in raw:
            if isinstance(item, dict) and item.get('segment') == 'NSE_EQ' and item.get('instrument_type') == 'EQ':
                symbol = item.get('trading_symbol')
                key = item.get('instrument_key')
                if isinstance(symbol, str) and isinstance(key, str) and key.startswith('NSE_EQ|'):
                    found[symbol.upper()] = (key, str(item.get('name', symbol)))
        if not found:
            raise ScreenerError('No NSE equity instruments found in provider directory.')
        _instruments = found
    return _instruments


def demo_rows(now):
    """Fictional histories with derived signals, never market prices."""
    end = month_number(now.astimezone(IST)) - 1
    rows = []
    for kind, symbol in [('fresh', 'DEMO-SPARK'), ('strong', 'DEMO-CLIMB'),
                         ('neutral', 'DEMO-PAUSE'), ('exit', 'DEMO-COOL')]:
        closes = [100.0]
        for i in range(1, 72):
            closes.append(closes[-1] + (1.5 if i % 2 else -1.0))
        if kind == 'strong':
            closes = [100.0 + i * 2 for i in range(72)]
        elif kind in ('fresh', 'neutral'):
            # Solve the last change for target RSI using the actual calculator.
            target = 76 if kind == 'fresh' else 69.6
            low, high = 0.0, 100.0
            for _ in range(70):
                change = (low + high) / 2
                if wilder_rsi(closes[:-1] + [closes[-2] + change])[-1] < target:
                    low = change
                else:
                    high = change
            closes[-1] = closes[-2] + (low + high) / 2
        else:
            closes[-1] = closes[-2] - 6
        candles = []
        for i, close in enumerate(closes):
            label = month_label(end - len(closes) + 1 + i)
            candles.append([label + '-01T00:00:00+05:30', close, close + 2, close - 2, close, 1000, 0])
        rows.append(analyze(symbol, 'Fictional ' + symbol[5:].title(), candles, now))
    return rows


def scan(payload):
    if not isinstance(payload, dict) or set(payload) - {'mode', 'symbols'}:
        raise ScreenerError('Only mode and symbols fields are accepted.')
    mode = payload.get('mode')
    if mode not in ('demo', 'live'):
        raise ScreenerError('Mode must be demo or live.')
    text = payload.get('symbols', '')
    if not isinstance(text, str) or len(text) > 20000:
        raise ScreenerError('Symbols must be a comma-separated string.')
    symbols = list(dict.fromkeys(s.strip().upper() for s in text.split(',') if s.strip()))
    if len(symbols) > 500:
        raise ScreenerError('A maximum of 500 symbols is allowed.')
    if any(len(s) > 64 or any(not (c.isascii() and (c.isalnum() or c in '&-_.')) for c in s) for s in symbols):
        raise ScreenerError('Invalid stock symbol.')
    if not _scan_lock.acquire(blocking=False):
        raise ScreenerError('A scan is already running. Wait for it to finish.')
    try:
        now = now_ist()
        result = {'mode': mode, 'as_of': now.isoformat(), 'rows': [], 'errors': [],
                  'warnings': ['Exit is an RSI threshold alert, not a portfolio instruction.',
                               'Corporate-action adjustment is unverified; no all-time-high assessment is made.',
                               'Breakout uses subsequent monthly highs, including any current provisional candle; no live quote.']}
        if mode == 'demo':
            result['rows'] = demo_rows(now)
            result['warnings'].insert(0, 'Synthetic fictional demo data; supplied symbols are ignored.')
            return result
        token = os.environ.get('UPSTOX_ACCESS_TOKEN', '').strip()
        if not token:
            raise ScreenerError('Live mode requires UPSTOX_ACCESS_TOKEN in the server environment.')
        if not symbols:
            raise ScreenerError('Provide at least one NSE equity symbol.')
        directory = instruments()
        last_request = 0.0
        for symbol in symbols:
            if symbol not in directory:
                result['errors'].append({'symbol': symbol, 'message': 'NSE equity symbol not found.'})
                continue
            key, name = directory[symbol]
            try:
                time.sleep(max(0, 0.35 - (time.monotonic() - last_request)))
                last_request = time.monotonic()
                url = HISTORY_URL.format(key=urllib.parse.quote(key, safe=''), today=now.date().isoformat())
                response = fetch_json(url, token)
                if not isinstance(response, dict) or not isinstance(response.get('data'), dict):
                    raise ScreenerError('Invalid history response from Upstox.')
                result['rows'].append(analyze(symbol, name, response['data'].get('candles'), now))
            except ScreenerError as exc:
                result['errors'].append({'symbol': symbol, 'message': str(exc)})
        return result
    finally:
        _scan_lock.release()
