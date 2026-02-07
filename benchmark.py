import pandas as pd
import numpy as np
import time
import sys
from TPS import DeltaTernary

# --- CONFIGURATION ---
THRESHOLD = 0.005
PATTERN_STR = "UUUD"  # Pattern: Up, Up, Up, Down
CSV_FILE = "btcusd.csv"
ITERATIONS = 50

def main():
    print(f"--- BENCHMARKING HFT ENGINE ({CSV_FILE}) ---")
    
    # ==========================================
    # PART 1: LOAD DATA
    # ==========================================
    try:
        df = pd.read_csv(CSV_FILE)
        df.columns = df.columns.str.strip().str.lower()
        
        # Robust column detection
        col = next((c for c in ['close', 'price'] if c in df.columns), None)
        if not col: 
            raise ValueError(f"No price column found. Available: {list(df.columns)}")
            
        prices = df[col].ffill().values.astype(np.float64)
        print(f"[1] Loaded {len(prices):,} candles.\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    # ==========================================
    # PART 2: STORAGE EFFICIENCY
    # ==========================================
    print("[2] Running Storage Benchmark...")

    raw_bytes = prices.nbytes
    raw_kb = raw_bytes / 1024

    engine = DeltaTernary(threshold=THRESHOLD)
    compressed_data, orig_len = engine.compress(prices)
    tern_kb = len(compressed_data) / 1024
    
    ratio_size = raw_kb / tern_kb

    print(f"    Raw Size:       {raw_kb:>10.2f} KB")
    print(f"    Ternary Size:   {tern_kb:>10.2f} KB")
    print(f"    Compression:    {ratio_size:>10.1f}x\n")

    # ==========================================
    # PART 2.5: VERIFICATION (CRITICAL FIX)
    # ==========================================
    print("[3] Verifying Logic Consistency...")

    # A. Float Search Logic
    # Delta calculation
    # Note: np.diff returns length N-1. 
    # To align indices for 'UUUD', we need slices of length (N-1)-3 = N-4
    deltas = np.diff(prices) / prices[:-1]
    d = deltas
    
    # Logic for "UUUD":
    # d[i] > T, d[i+1] > T, d[i+2] > T, d[i+3] < -T
    mask = (d[:-3] > THRESHOLD) & \
           (d[1:-2] > THRESHOLD) & \
           (d[2:-1] > THRESHOLD) & \
           (d[3:] < -THRESHOLD)
    
    float_count = np.sum(mask)

    # B. Ternary Search Logic
    trits = engine.unpack_trits(compressed_data, orig_len)
    t_chars = [engine.char_map[t] for t in trits]
    t_string = "".join(t_chars)
    
    tern_count = 0
    pos = t_string.find(PATTERN_STR)
    while pos != -1:
        tern_count += 1
        # Overlapping search: advance by 1
        pos = t_string.find(PATTERN_STR, pos + 1)

    if float_count == tern_count:
        print(f"    ✅ SUCCESS: Both algorithms found exactly {float_count} matches.")
    else:
        print(f"    ❌ WARNING: Mismatch! Float={float_count}, Ternary={tern_count}")
        print("       (Check threshold strictness or floating point precision)\n")

    # ==========================================
    # PART 3: ALGORITHMIC SEARCH SPEED
    # ==========================================
    print(f"\n[4] Running Algorithmic Search Speed ({ITERATIONS} runs)...")
    print("    (Measuring pure search logic, excluding decode overhead)\n")

    # Benchmark Loop
    float_times = []
    ternary_times = []

    for _ in range(ITERATIONS):
        # Float search
        start = time.perf_counter()
        # Re-compute mask to measure CPU cost
        _ = (d[:-3] > THRESHOLD) & (d[1:-2] > THRESHOLD) & \
            (d[2:-1] > THRESHOLD) & (d[3:] < -THRESHOLD)
        float_times.append((time.perf_counter() - start) * 1000)

        # Ternary search
        start = time.perf_counter()
        pos = t_string.find(PATTERN_STR)
        while pos != -1:
            pos = t_string.find(PATTERN_STR, pos + 1)
        ternary_times.append((time.perf_counter() - start) * 1000)

    float_avg = np.mean(float_times)
    tern_avg = np.mean(ternary_times)
    float_std = np.std(float_times)
    tern_std = np.std(ternary_times)
    
    # Prevent divide by zero if fast machine gets 0.0ms
    if tern_avg == 0: tern_avg = 0.0001
    ratio_speed = float_avg / tern_avg

    # ==========================================
    # PART 4: DECODE OVERHEAD
    # ==========================================
    print("[5] Measuring Python Decoding Overhead (One-time cost)...\n")
    
    decode_times = []
    for _ in range(10):  # Run 10 times for stability
        start_decode = time.perf_counter()
        _trits = engine.unpack_trits(compressed_data, orig_len)
        _chars = [engine.char_map[t] for t in _trits]
        _str = "".join(_chars)
        decode_times.append((time.perf_counter() - start_decode) * 1000)
    
    decode_time_ms = np.mean(decode_times)
    decode_std = np.std(decode_times)
    
    # Calculate exact break-even point
    # Cost_Float = N * Float_Avg
    # Cost_Ternary = Decode_Overhead + (N * Ternary_Avg)
    # N * (Float_Avg - Ternary_Avg) = Decode_Overhead
    
    speed_diff = float_avg - tern_avg
    if speed_diff > 0:
        breakeven = decode_time_ms / speed_diff
    else:
        breakeven = float('inf') # Ternary is slower (unlikely)

    # ==========================================
    # OUTPUT: PUBLICATION-READY TABLES
    # ==========================================
    print("=" * 65)
    print("            TABLE 1: ALGORITHMIC PERFORMANCE")
    print("          (Once data is loaded in memory)")
    print("=" * 65)
    print(f"{'Metric':<20} | {'Float64':<15} | {'Ternary':<15} | {'Improvement'}")
    print("-" * 65)
    print(f"{'Storage (KB)':<20} | {raw_kb:<15.2f} | {tern_kb:<15.2f} | {ratio_size:.1f}x")
    print(f"{'Search (ms)':<20} | {float_avg:<15.4f} | {tern_avg:<15.4f} | {ratio_speed:.1f}x")
    print(f"{'Search Std Dev':<20} | {float_std:<15.4f} | {tern_std:<15.4f} | ±{tern_std:.2f}ms")
    print("=" * 65)

    print("\n" + "=" * 65)
    print("          TABLE 2: IMPLEMENTATION OVERHEAD")
    print("          (Python-specific Decoding Cost)")
    print("=" * 65)
    print(f"{'Metric':<20} | {'Value':<15} | {'Note'}")
    print("-" * 65)
    print(f"{'Decode Time':<20} | {decode_time_ms:<15.2f} | ±{decode_std:.2f}ms")
    
    be_str = f"{int(round(breakeven)):,}" if breakeven != float('inf') else "N/A"
    print(f"{'Break-even':<20} | {be_str:<15} | searches")
    print("=" * 65)
    
    print("\n📊 INTERPRETATION:")
    if breakeven != float('inf'):
        print(f"  • For <{be_str} queries: Use Float64 (faster setup)")
        print(f"  • For >{be_str} queries: Use Ternary (faster execution)")
        
        speedup_100 = (float_avg*100)/(decode_time_ms + tern_avg*100)
        speedup_1000 = (float_avg*1000)/(decode_time_ms + tern_avg*1000)
        
        print(f"  • At 100 queries: Ternary is {speedup_100:.2f}x effective speed")
        print(f"  • At 1000 queries: Ternary is {speedup_1000:.2f}x effective speed")
    
    print("\n  NOTE: Decode overhead is Python-specific.")
    print("    C/Rust implementation would reduce this by ~100x.")

if __name__ == "__main__":
    main()