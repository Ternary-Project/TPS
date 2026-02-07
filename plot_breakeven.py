import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_breakeven(decode_ms=2401.73, float_ms=53.2936, ternary_ms=4.2886):
    """
    Generates break-even analysis chart comparing Float64 vs Ternary search.
    Values default to the latest benchmark run on 7.35M Bitcoin candles.
    """
    # --- 1. SAFETY CHECKS ---
    speed_diff = float_ms - ternary_ms
    if speed_diff <= 0:
        print("Error: Ternary search must be faster than Float64 to find a break-even point.")
        return None

    # --- 2. DATA PREPARATION ---
    # Query range for visualization (1 to 100 searches)
    searches = np.arange(1, 101)
    
    # Calculate cumulative execution time
    time_float = searches * float_ms
    time_ternary = decode_ms + (searches * ternary_ms)
    
    # Calculate exact intersection point
    intersect_x = decode_ms / speed_diff
    intersect_y = intersect_x * float_ms
    
    # --- 3. PLOTTING ---
    plt.figure(figsize=(12, 7), dpi=300)
    
    # Plot trend lines
    plt.plot(searches, time_float, 
             label='Standard Float64 (No Compression)', 
             color='#e74c3c', linewidth=3, linestyle='--', alpha=0.9)
    
    plt.plot(searches, time_ternary, 
             label='Ternary Search (Compressed)', 
             color='#27ae60', linewidth=3, alpha=0.9)
    
    # Vertical Guide Line (Visual Aid)
    plt.axvline(intersect_x, color='black', linestyle=':', alpha=0.5, zorder=1)

    # Mark intersection point
    plt.scatter([intersect_x], [intersect_y], 
                color='black', s=100, zorder=5, marker='o', edgecolors='white')
    
    # Annotate break-even point
    plt.annotate(
        f'Break-even ≈ {intersect_x:.0f} searches', 
        xy=(intersect_x, intersect_y), 
        xytext=(intersect_x + 8, intersect_y * 0.8),
        fontsize=11, fontweight='bold',
        arrowprops=dict(facecolor='black', shrink=0.08, width=2, headwidth=8)
    )
    
    # Label the decode overhead
    plt.text(
        2, decode_ms * 0.92,
        f"Python Decode Overhead = {decode_ms:.0f} ms", 
        fontsize=10, fontstyle='italic',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#27ae60", alpha=0.8)
    )
    
    # Shade efficiency gain zone
    plt.fill_between(searches, time_float, time_ternary, 
                     where=(searches > intersect_x), 
                     color='#27ae60', alpha=0.15, 
                     label='Efficiency Gain Zone')
    
    # --- 4. STYLING & EXPORT ---
    plt.title("Performance Break-Even Analysis: Ternary vs Standard Search", 
              fontsize=16, pad=20, fontweight='bold')
    plt.xlabel("Number of Search Queries", fontsize=12)
    plt.ylabel("Total Execution Time (ms)", fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Save and Close (Memory Safe)
    filename = "chart_breakeven_analysis.png"
    plt.savefig(filename)
    plt.close() # Important: Frees memory
    
    # --- 5. REPORTING ---
    speedup_100 = time_float[-1] / time_ternary[-1]
    
    print(f"✅ Chart saved as '{filename}'")
    print(f"📊 Exact break-even: {intersect_x:.2f} searches")
    print(f"📈 Speedup at 100 queries: {speedup_100:.2f}x")
    
    return {
        "break_even_searches": intersect_x,
        "speedup_at_100": speedup_100
    }

if __name__ == "__main__":
    # You can run it with defaults, or pass new numbers from a fresh benchmark run
    plot_breakeven()