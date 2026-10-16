import datetime as dt
import http.client
import json
import math
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import screener
import server

NOW = dt.datetime(2026, 10, 10, tzinfo=screener.IST)


def candles(count=72):
    end = screener.month_number(NOW) - 1
    return [[screener.month_label(end - count + i + 1) + '-01T00:00:00+05:30',
             100 + i, 102 + i, 98 + i, 100 + i, 1000, 0] for i in range(count)]


class CalculationTests(unittest.TestCase):
    def test_wilder_reference(self):
        # Published Wilder example: first RSI 70.4641, then smoothed 66.2496.
        values = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
                  45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00]
        actual = screener.wilder_rsi(values)
        self.assertIsNone(actual[13])
        self.assertAlmostEqual(actual[14], 70.464135, places=5)
        self.assertAlmostEqual(actual[15], 66.249619, places=5)

    def test_flat_up_down(self):
        for values, expected in [([20] * 30, 50), (list(range(1, 31)), 100),
                                 (list(range(31, 1, -1)), 0)]:
            self.assertEqual(screener.wilder_rsi(values)[-1], expected)

    def test_threshold_boundaries(self):
        for prev, curr, expected in [(70, 70.00001, 'fresh'), (71, 71, 'strong'),
                                     (65, 70, 'neutral'), (70, 69.3, 'neutral'),
                                     (70, 69.2999, 'exit')]:
            self.assertEqual(screener.classify(prev, curr), expected)

    def test_sorted_complete_months_and_ist_boundary(self):
        data = candles()
        # This UTC instant is October in IST and must not enter RSI.
        data.append(['2026-09-30T19:00:00Z', 900, 910, 890, 900, 1])
        row = screener.analyze('TEST', 'Test', list(reversed(data)), NOW)
        self.assertEqual(row['month'], '2026-09')
        self.assertEqual(row['close'], 171)
        self.assertEqual(len(row['history']), 72)

    def test_reject_stale_gap_duplicate_invalid_short(self):
        invalid = candles()
        invalid[-1][4] = math.nan
        cases = [candles()[:-1], candles()[:10] + candles()[11:],
                 candles() + [candles()[-1]], invalid, candles(15)]
        for data in cases:
            with self.subTest(data_length=len(data)), self.assertRaises(screener.ScreenerError):
                screener.analyze('TEST', 'Test', data, NOW)
        row = screener.analyze('TEST', 'Test', candles(16), NOW)
        self.assertIn('Fewer than 60', row['warning'])

    def test_demo_derived_mix(self):
        rows = screener.demo_rows(NOW)
        self.assertEqual({r['signal'] for r in rows}, {'fresh', 'strong', 'neutral', 'exit'})
        for row in rows:
            self.assertAlmostEqual(screener.wilder_rsi([c['close'] for c in row['history']])[-1], row['rsi'])
            self.assertTrue(row['symbol'].startswith('DEMO-'))
        self.assertFalse(rows[0]['breakout'])  # Crossing candle itself cannot be breakout.

    def test_breakout_after_crossing_in_provisional_month(self):
        row = screener.demo_rows(NOW)[0]
        data = [[h['month'] + '-01T00:00:00+05:30', h['close'], h['close'] + 2,
                 h['close'] - 2, h['close'], 1000] for h in row['history']]
        high = row['signal_high']
        data.append(['2026-10-01T00:00:00+05:30', high, high + 1, high - 1, high, 1000])
        result = screener.analyze('TEST', 'Test', data, NOW)
        self.assertTrue(result['breakout'])
        self.assertEqual(result['rsi'], row['rsi'])


class ProviderTests(unittest.TestCase):
    def test_missing_token_no_fallback(self):
        with patch.dict('os.environ', {}, clear=True), self.assertRaises(screener.ScreenerError):
            screener.scan({'mode': 'live', 'symbols': 'TCS'})

    def test_upstream_failure_is_explicit_and_sanitized(self):
        opener = unittest.mock.Mock()
        opener.open.side_effect = urllib.error.HTTPError('secret-url', 401, 'secret-token', {}, None)
        with patch('urllib.request.build_opener', return_value=opener):
            with self.assertRaisesRegex(screener.ScreenerError, 'authentication failed') as error:
                screener.fetch_json('https://api.upstox.com/test', 'secret-token')
            self.assertNotIn('secret', str(error.exception))

    def test_live_error_and_url_contract(self):
        with patch.dict('os.environ', {'UPSTOX_ACCESS_TOKEN': 'private'}), \
                patch.object(screener, 'now_ist', return_value=NOW), \
                patch.object(screener, 'instruments', return_value={'TCS': ('NSE_EQ|INE1', 'TCS')}), \
                patch.object(screener, 'fetch_json', return_value={'data': {'candles': candles()}}) as fetch:
            result = screener.scan({'mode': 'live', 'symbols': 'TCS,UNKNOWN'})
        self.assertEqual(len(result['rows']), 1)
        self.assertEqual(result['errors'][0]['symbol'], 'UNKNOWN')
        self.assertIn('NSE_EQ%7CINE1/months/1/2026-10-10/2000-01-01', fetch.call_args.args[0])
        self.assertNotIn('private', json.dumps(result))

    def test_live_provider_failure_stays_live(self):
        with patch.dict('os.environ', {'UPSTOX_ACCESS_TOKEN': 'private'}), \
                patch.object(screener, 'instruments', return_value={'TCS': ('NSE_EQ|INE1', 'TCS')}), \
                patch.object(screener, 'fetch_json', side_effect=screener.ScreenerError('Provider HTTP error (503).')):
            result = screener.scan({'mode': 'live', 'symbols': 'TCS'})
        self.assertEqual(result['mode'], 'live')
        self.assertEqual(result['rows'], [])
        self.assertEqual(result['errors'], [{'symbol': 'TCS', 'message': 'Provider HTTP error (503).'}])

    def test_limit_unexpected_fields_and_lock(self):
        for payload in [{'mode': 'demo', 'token': 'no'},
                        {'mode': 'live', 'symbols': ','.join('S' + str(i) for i in range(501))}]:
            with self.assertRaises(screener.ScreenerError):
                screener.scan(payload)
        with screener._scan_lock, self.assertRaisesRegex(screener.ScreenerError, 'already running'):
            screener.scan({'mode': 'demo'})


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.httpd.server_port

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join()

    def request(self, method='GET', path='/api/status', body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        code, data, headers = response.status, response.read(), dict(response.getheaders())
        conn.close()
        return code, data, headers

    def test_status_and_path_allowlist(self):
        code, data, headers = self.request()
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(data)['provider'], 'Upstox')
        self.assertNotIn('Access-Control-Allow-Origin', headers)
        for path in ['/server.py', '/../server.py', '/%2e%2e/server.py']:
            self.assertEqual(self.request(path=path)[0], 404)

    def test_host_origin_type_and_body_limits(self):
        self.assertEqual(self.request(headers={'Host': 'evil.example'})[0], 403)
        body = '{"mode":"demo"}'
        headers = {'Content-Type': 'application/json'}
        self.assertEqual(self.request('POST', '/api/scan', body, headers)[0], 403)
        headers['Origin'] = 'https://evil.example'
        self.assertEqual(self.request('POST', '/api/scan', body, headers)[0], 403)
        headers['Origin'] = f'http://127.0.0.1:{self.port}'
        headers['Content-Type'] = 'text/plain'
        self.assertEqual(self.request('POST', '/api/scan', body, headers)[0], 415)
        headers['Content-Type'] = 'application/json'
        self.assertEqual(self.request('POST', '/api/scan', 'x' * 32769, headers)[0], 413)
        self.assertEqual(self.request('POST', '/api/scan', '{', headers)[0], 400)
        code, data, _ = self.request('POST', '/api/scan', body, headers)
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(data)['mode'], 'demo')


if __name__ == '__main__':
    unittest.main()
