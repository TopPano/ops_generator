import test from 'ava';
import types from '../lib/types';

test('cvToStd', t => {
  t.is(types.cvToStd('8U'), 'uint8_t');
});
