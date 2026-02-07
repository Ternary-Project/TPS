"""
TPS Module Test Script
======================
Tests DeltaTernary compression and pattern detection on market data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    from TPS import DeltaTernary
except ImportError as e:
    print(f"❌ Failed to import TPS: {e}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "threshold": 0.005,
    "csv_file": "btcusd.csv",
    "max_plot_points": 500,
    "output_dir": ".",
    "dpi": 100,
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(filepath: str) -> tuple:
    """
    Load and validate price data.
    
    Returns:
        (prices, dates) tuple
    
    Raises:
        FileNotFoundError, ValueError on errors
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    
    # Auto-detect columns
    date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
    price_col = next((c for c in df.columns if 'close' in c or 'price' in c), None)
    
    if not date_col:
        raise ValueError(f"No date column found. Available: {list(df.columns)}")
    if not price_col:
        raise ValueError(f"No price column found. Available: {list(df.columns)}")
    
    # Parse dates
    try:
        df['_date'] = pd.to_datetime(df[date_col])
    except:
        df['_date'] = pd.to_datetime(df[date_col], unit='s')
    
    df = df.sort_values('_date').reset_index(drop=True)
    
    prices = df[price_col].ffill().bfill().values.astype(np.float64)
    dates = df['_date'].values
    
    # Validate
    if not np.isfinite(prices).all():
        raise ValueError("Data contains NaN or Inf values")
    
    return prices, dates


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def get_pattern_color(name: str) -> str:
    """Return color based on pattern type."""
    name_lower = name.lower()
    if 'vol' in name_lower or 'squeeze' in name_lower:
        return '#3498db'  # Blue
    elif 'algo' in name_lower:
        return '#27ae60'  # Green
    elif 'bart' in name_lower:
        return '#e67e22'  # Orange
    else:
        return '#e74c3c'  # Red


def create_pattern_chart(
    dates: np.ndarray,
    prices: np.ndarray,
    matches: list,
    pattern_name: str,
    pattern_seq: str,
    output_dir: str,
    max_points: int = 500,
    dpi: int = 100
) -> str:
    """Create and save pattern detection chart."""
    
    pat_len = len(pattern_seq)
    total = len(matches)
    
    # Limit points
    if total > max_points:
        matches_to_plot = matches[-max_points:]
        suffix = f" (Last {max_points})"
    else:
        matches_to_plot = matches
        suffix = ""
    
    # Calculate signal positions
    # Pattern at position m → signal at end of pattern (m + pat_len)
    signal_indices = []
    for m in matches_to_plot:
        idx = m + pat_len
        if idx < len(prices):
            signal_indices.append(idx)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    
    ax.plot(dates, prices, 
            label='Close Price', 
            color='#2c3e50', 
            linewidth=0.8, 
            alpha=0.7)
    
    if signal_indices:
        color = get_pattern_color(pattern_name)
        ax.scatter(
            dates[signal_indices], 
            prices[signal_indices],
            color=color, 
            s=40, 
            zorder=5, 
            label=f'{pattern_name} Signal',
            alpha=0.7,
            edgecolors='white',
            linewidths=0.5
        )
    
    ax.set_title(f"Ternary Detection: {pattern_name} | Total: {total}{suffix}", fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    fig.autofmt_xdate()
    
    # Save
    safe_name = pattern_name.replace(" ", "_").lower()
    filepath = Path(output_dir) / f"chart_{safe_name}.png"
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    
    return str(filepath)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> Optional[Dict]:
    """
    Run TPS module test.
    
    Returns:
        Dictionary with results, or None on failure.
    """
    print(f"{'═' * 60}")
    print(f"  TPS CORE TESTER")
    print(f"{'═' * 60}")
    
    # ─── Load Data ───
    print(f"\n📂 Loading: {CONFIG['csv_file']}")
    try:
        prices, dates = load_data(CONFIG['csv_file'])
        print(f"   ✓ Loaded {len(prices):,} candles")
        print(f"   ✓ Range: ${prices.min():,.2f} - ${prices.max():,.2f}")
    except FileNotFoundError as e:
        print(f"   ❌ {e}")
        return None
    except ValueError as e:
        print(f"   ❌ {e}")
        return None
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return None
    
    # ─── Initialize Engine ───
    engine = DeltaTernary(threshold=CONFIG['threshold'])
    print(f"\n⚙️  Engine: {engine}")
    
    # ─── Compress ───
    print("\n📦 Compressing...")
    t0 = time.perf_counter()
    compressed, orig_len = engine.compress(prices)
    t1 = time.perf_counter()
    
    if orig_len == 0:
        print("   ❌ Compression failed (data may be constant, NaN, or too short)")
        return None
    
    comp_time = (t1 - t0) * 1000
    comp_kb = len(compressed) / 1024
    raw_kb = prices.nbytes / 1024
    ratio = raw_kb / comp_kb
    
    print(f"   ✓ Time: {comp_time:.2f} ms")
    print(f"   ✓ Size: {comp_kb:.2f} KB ({ratio:.1f}× compression)")
    
    # ─── Decode Once ───
    print("\n🔓 Decoding to search string...")
    t0 = time.perf_counter()
    stream_str = engine.to_string(compressed, orig_len)
    t1 = time.perf_counter()
    print(f"   ✓ Time: {(t1-t0)*1000:.2f} ms")
    print(f"   ✓ Length: {len(stream_str):,} characters")
    
    # ─── Trit Distribution ───
    trits = engine.unpack_trits(compressed, orig_len)
    unique, counts = np.unique(trits, return_counts=True)
    print("\n📊 Trit Distribution:")
    for t, c in zip(unique, counts):
        symbol = {-1: 'D (Down)', 0: '- (Flat)', 1: 'U (Up)'}[t]
        pct = c / len(trits) * 100
        print(f"   {symbol:12}: {c:>8,} ({pct:5.1f}%)")
    
    # ─── Search Patterns ───
    patterns = engine.get_trading_patterns()
    print(f"\n🔍 Scanning for {len(patterns)} patterns...")
    print("-" * 60)
    
    results = {}
    
    for name, seq in patterns.items():
        t0 = time.perf_counter()
        
        # Fast manual search
        matches = []
        pos = stream_str.find(seq)
        while pos != -1:
            matches.append(pos)
            pos = stream_str.find(seq, pos + 1)
        
        t1 = time.perf_counter()
        search_time = (t1 - t0) * 1000
        count = len(matches)
        
        results[name] = {
            "count": count,
            "search_time_ms": search_time,
            "pattern": seq
        }
        
        if count == 0:
            print(f"   [SKIP] {name:<18} (0 found)")
            continue
        
        # Create chart
        chart_path = create_pattern_chart(
            dates, prices, matches,
            name, seq,
            CONFIG['output_dir'],
            CONFIG['max_plot_points'],
            CONFIG['dpi']
        )
        
        results[name]["chart"] = chart_path
        print(f"   [PLOT] {name:<18} | Found: {count:<6} | Time: {search_time:.4f}ms")
    
    # ─── Summary ───
    print("-" * 60)
    print("\n📋 SUMMARY:")
    total_found = sum(r["count"] for r in results.values())
    print(f"   Total patterns detected: {total_found:,}")
    
    for name, data in results.items():
        if data["count"] > 0:
            print(f"   • {name}: {data['count']:,}")
    
    print(f"\n✅ Done! Charts saved to '{CONFIG['output_dir']}'")
    
    return results


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)