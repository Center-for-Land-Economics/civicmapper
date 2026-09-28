export function numOrNull(v: any): number | null {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

export function fmt(n: any, digits = 2): string {
  const x = Number(n);
  if (!Number.isFinite(x)) return String(n ?? '—');
  if (Math.abs(x) >= 1) return x.toLocaleString(undefined, { maximumFractionDigits: digits });
  if (x === 0) return '0';
  return x.toLocaleString(undefined, { maximumSignificantDigits: 3 });
}

// One-entry memo of the last array sorted by percentile(). Callers ask for many percentiles of
// the SAME value array in a row (quantileBreaks → k−1 calls, the legend's p1/p99/p99.9, one per
// colour-break handle), and each call used to copy + comparator-sort the whole array — ~10 full
// sorts of 80k values per metric switch on the larger GeoParquet cities. Keyed by identity, and
// re-validated by length + three sampled elements so an array mutated in place is re-sorted.
let _memoSrc: number[] | null = null;
let _memoLen = -1;
let _memoProbe: [number, number, number] = [NaN, NaN, NaN];
let _memoSorted: Float64Array | null = null;

function sortedValues(vals: number[]): Float64Array {
  const n = vals.length;
  const probe: [number, number, number] = [vals[0], vals[n >> 1], vals[n - 1]];
  if (
    _memoSrc === vals && _memoLen === n && _memoSorted &&
    Object.is(_memoProbe[0], probe[0]) && Object.is(_memoProbe[1], probe[1]) && Object.is(_memoProbe[2], probe[2])
  ) {
    return _memoSorted;
  }
  // Typed-array sort is numeric (no comparator callback) and several times faster than
  // Array#sort((x, y) => x - y) for the same finite values.
  const sorted = Float64Array.from(vals).sort();
  _memoSrc = vals; _memoLen = n; _memoProbe = probe; _memoSorted = sorted;
  return sorted;
}

export function percentile(vals: number[], p: number): number {
  if (!vals.length) return NaN;
  const a = sortedValues(vals);
  const idx = (p / 100) * (a.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return a[lo];
  const t = idx - lo;
  return a[lo] + (a[hi] - a[lo]) * t;
}

export function quantileBreaks(values: number[], k: number, lowPct = 1, highPct = 99): number[] {
  const ks = Math.max(2, Math.min(k, 12));
  const out: number[] = [];
  for (let i = 1; i < ks; i++) {
    const p = lowPct + (highPct - lowPct) * (i / ks);
    const q = percentile(values, p);
    if (Number.isFinite(q)) out.push(q);
  }
  out.sort((a,b)=>a-b);
  return out.filter((v, i) => i === 0 || v > out[i-1]);
}
