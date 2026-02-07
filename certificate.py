import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from TPS import DeltaTernary

# --- CONFIGURATION ---
CSV_FILE = 'btcusd.csv'
THRESHOLD = 0.005
TEST_SIZE = 4056

def verify_integrity():
    print(f"--- CERTIFYING REAL DATA ({CSV_FILE}) ---")
    
    # 1. Load Data
    try:
        header_df = pd.read_csv(CSV_FILE, nrows=0)
        clean_cols = {c.strip().lower(): c for c in header_df.columns}
        
        target_col = None
        for candidate in ['close', 'price', 'last', 'adj close']:
            if candidate in clean_cols:
                target_col = clean_cols[candidate]
                break
        
        if not target_col:
            print(f"❌ Error: No price column in: {list(header_df.columns)}")
            return None

        print(f"   ✓ Found price column: '{target_col}'")

        df = pd.read_csv(CSV_FILE, usecols=[target_col], nrows=TEST_SIZE * 2)
        prices = df[target_col].ffill().values[:TEST_SIZE].astype(float)
        
        if len(prices) < TEST_SIZE:
            print(f"⚠️ Warning: Requested {TEST_SIZE}, found {len(prices)}")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

    # 2. Compress
    engine = DeltaTernary(threshold=THRESHOLD)
    compressed, length = engine.compress(prices)
    
    if length == 0:
        print("❌ Error: Compression failed")
        return None
    
    # 3. Decompress
    reconstructed = engine.decompress(compressed, length, start_price=prices[0])
    
    if len(reconstructed) == 0:
        print("❌ Error: Decompression returned empty array")
        return None

    # 4. Align lengths
    n = min(len(prices), len(reconstructed))
    clean_orig = prices[:n]
    clean_recon = reconstructed[:n]

    # ═══════════════════════════════════════════════════════════════
    #  METRICS
    # ═══════════════════════════════════════════════════════════════
    
    # A. Shape Correlation
    correlation = np.corrcoef(clean_orig, clean_recon)[0, 1]
    
    # B. RMSE
    rmse = np.sqrt(np.mean((clean_orig - clean_recon) ** 2))
    
    # C. Drift Analysis
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_errors = np.abs((clean_orig - clean_recon) / clean_orig)
        relative_errors = np.nan_to_num(relative_errors)
    
    max_drift_pct = np.max(relative_errors) * 100
    avg_drift_pct = np.mean(relative_errors) * 100

    # D. DIRECTION ACCURACY (Most important for ternary!)
    orig_direction = np.sign(np.diff(clean_orig))
    recon_direction = np.sign(np.diff(clean_recon))
    direction_accuracy = np.mean(orig_direction == recon_direction) * 100

    # E. Compression Ratio
    raw_size = clean_orig.nbytes
    comp_size = len(compressed)
    ratio = raw_size / comp_size if comp_size > 0 else 0

    # ═══════════════════════════════════════════════════════════════
    #  REPORT
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n{'Metric':<20} | {'Value':<15} | {'Status'}")
    print("=" * 55)
    print(f"{'Data Points':<20} | {n:<15,} | ")
    print(f"{'Comp. Ratio':<20} | {ratio:<15.2f}x | ✓")
    print(f"{'Correlation':<20} | {correlation:<15.5f} | {'✓' if correlation > 0.95 else '⚠️'}")
    print(f"{'Direction Accuracy':<20} | {direction_accuracy:<14.2f}% | {'✓' if direction_accuracy > 99 else '⚠️'}")
    print(f"{'RMSE':<20} | {rmse:<15.4f} | (expected)")
    print(f"{'Avg Drift':<20} | {avg_drift_pct:<14.2f}% | (expected)")
    print(f"{'Max Drift':<20} | {max_drift_pct:<14.2f}% | (expected)")
    print("=" * 55)

    # ═══════════════════════════════════════════════════════════════
    #  VERDICT
    # ═══════════════════════════════════════════════════════════════
    
    # Primary check: Direction accuracy (this is what ternary preserves!)
    if direction_accuracy >= 99.0 and correlation > 0.95:
        print("✅ PASS: Excellent fidelity")
        verdict = "PASS"
    elif direction_accuracy >= 95.0 and correlation > 0.90:
        print("⚠️ WARN: Good fidelity with expected drift")
        verdict = "WARN"
    else:
        print("❌ FAIL: Compression quality issue")
        verdict = "FAIL"

    # ═══════════════════════════════════════════════════════════════
    #  PLOT
    # ═══════════════════════════════════════════════════════════════
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150, 
                              gridspec_kw={'height_ratios': [3, 1]})
    
    # Top: Price comparison
    ax1 = axes[0]
    ax1.plot(clean_orig, color='black', alpha=0.6, linewidth=1, label='Original')
    ax1.plot(clean_recon, color='#e74c3c', linestyle='--', alpha=0.7, linewidth=0.8, 
             label='Reconstructed')
    ax1.set_title(
        f"Integrity Certificate: {ratio:.1f}× | r={correlation:.4f} | Dir={direction_accuracy:.1f}%",
        fontsize=12, fontweight='bold'
    )
    ax1.set_ylabel("Price")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2)
    
    # Bottom: Drift over time
    ax2 = axes[1]
    drift_pct = (clean_recon - clean_orig) / clean_orig * 100
    ax2.fill_between(range(len(drift_pct)), 0, drift_pct, 
                     where=(drift_pct > 0), color='green', alpha=0.3, label='Over')
    ax2.fill_between(range(len(drift_pct)), 0, drift_pct,
                     where=(drift_pct < 0), color='red', alpha=0.3, label='Under')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel("Candles")
    ax2.set_ylabel("Drift (%)")
    ax2.set_title("Cumulative Drift", fontsize=10)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    filename = "certificate_final.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Certificate saved: '{filename}'")
    
    return {
        "correlation": correlation,
        "direction_accuracy": direction_accuracy,
        "compression_ratio": ratio,
        "rmse": rmse,
        "max_drift_pct": max_drift_pct,
        "verdict": verdict
    }


if __name__ == "__main__":
    result = verify_integrity()
    sys.exit(0 if result else 1)