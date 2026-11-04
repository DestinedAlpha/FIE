import type { Ledger, Player } from './types';

// A transfer bundle has no intrinsic list pairing. Match positions explicitly,
// including archived ledgers produced before the backend normalized their order.
export function transferPairs(row: Pick<Ledger, 'sell' | 'buy'>, players: Map<number, Player>) {
  const remaining = [...row.sell];
  return row.buy.map(buy => {
    const index = remaining.findIndex(sale =>
      players.get(sale.player_id)?.position === players.get(buy.player_id)?.position);
    if (index < 0) throw new Error('Invalid transfer bundle: squad positions do not balance');
    return { sale: remaining.splice(index, 1)[0], buy };
  });
}
