import { describe, it, expect } from 'vitest';
import { transferPairs } from './transfers';
import type { Player } from './types';
describe('transfer bundles', () => {
  it('pairs positions even when sales and purchases have different orders', () => {
    const players = new Map([[1, {position: 1}], [2, {position: 2}], [3, {position: 2}], [4, {position: 1}]]) as Map<number, Player>;
    const pairs = transferPairs({sell: [{player_id: 1, units: 50}, {player_id: 2, units: 60}], buy: [{player_id: 3, units: 45}, {player_id: 4, units: 40}]}, players);
    expect(pairs.map(p => [p.sale.player_id, p.buy.player_id])).toEqual([[2, 3], [1, 4]]);
  });
});
