'use strict';
const assert = require('node:assert/strict');
const { SIGNAL_GROUPS, signalGroup, groupRows, paginateRows } = require('../public/app.js');

const rows = [
  { symbol: 'A', signal: 'fresh' }, { symbol: 'B', signal: 'strong' },
  { symbol: 'C', signal: 'neutral' }, { symbol: 'D', signal: 'exit' },
  { symbol: 'E', signal: 'unexpected' },
];
assert.deepEqual(SIGNAL_GROUPS, ['fresh', 'strong', 'neutral', 'exit']);
assert.equal(signalGroup(rows[4]), 'neutral');
assert.deepEqual(Object.fromEntries(Object.entries(groupRows(rows)).map(([key, value]) => [key, value.map((row) => row.symbol)])), {
  fresh: ['A'], strong: ['B'], neutral: ['C', 'E'], exit: ['D'],
});
assert.deepEqual(paginateRows(Array.from({ length: 26 }, (_, i) => i), 0), { rows: Array.from({ length: 25 }, (_, i) => i), page: 0, pages: 2, total: 26 });
assert.deepEqual(paginateRows(Array.from({ length: 26 }, (_, i) => i), 99), { rows: [25], page: 1, pages: 2, total: 26 });
console.log('grouped tables helpers: ok');
