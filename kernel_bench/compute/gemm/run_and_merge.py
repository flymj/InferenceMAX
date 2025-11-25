import subprocess
import pandas as pd
import os
import sys
import argparse

def run_benchmark(dtype="fp16", m_arg="256:16384:256", n_arg="256:16384:256", k_arg="256:16384:256", output_csv="gemm_results.csv", ncu_csv="ncu_report.csv"):
    # 1. Build the ncu command
    # We filter by NVTX range "TensorCoreRange" to only profile the relevant loop
    # Metric: sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
    metric_name = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
    
    cmd = [
        "ncu",
        "--csv",
        "--quiet",
        "--metrics", metric_name,
        "--nvtx",
        "--nvtx-include", "TensorCoreRange",
        "--clock-control", "none", # Optional: disable clock control for faster run if needed, but might affect consistency
        "./gemm",
        "--dtype", dtype,
        "--M", m_arg,
        "--N", n_arg,
        "--K", k_arg,
        "--sweep",
        "--profile" # Tell the C++ app to run the NVTX-wrapped profiling loop
    ]
    
    print(f"Running command: {' '.join(cmd)} > {ncu_csv}")
    
    try:
        with open(ncu_csv, "w") as outfile:
            subprocess.run(cmd, check=True, stdout=outfile, stderr=subprocess.PIPE) # Capture stdout to file, stderr to pipe (or inherit)
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr.decode()}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'ncu' command not found. Please ensure Nsight Compute is installed and in your PATH.")
        sys.exit(1)

    return metric_name

def merge_results(gemm_csv, ncu_csv, merged_csv, metric_name):
    print("Merging results...")
    
    if not os.path.exists(gemm_csv):
        print(f"Error: {gemm_csv} not found.")
        return
    if not os.path.exists(ncu_csv):
        print(f"Error: {ncu_csv} not found.")
        return

    # Load DataFrames
    df_gemm = pd.read_csv(gemm_csv)
    
    # NCU output parsing:
    # The CSV data starts after the line "All target application created processes are terminated."
    try:
        with open(ncu_csv, 'r') as f:
            lines = f.readlines()
            
        start_index = -1
        for i, line in enumerate(lines):
            if "All target application created processes are terminated." in line:
                start_index = i + 1
                break
        
        if start_index != -1 and start_index < len(lines):
            # Check if the next line is empty, sometimes there's a blank line
            if not lines[start_index].strip():
                start_index += 1
            
            from io import StringIO
            csv_content = "".join(lines[start_index:])
            df_ncu = pd.read_csv(StringIO(csv_content))
        else:
            # Fallback: try reading directly if marker not found (maybe clean output)
            print("Warning: NCU marker not found, attempting to read file directly.")
            df_ncu = pd.read_csv(ncu_csv)

    except Exception as e:
        print(f"Error reading NCU CSV: {e}")
        return

    # Filter NCU results to ensure we have the metric
    # The column name in NCU CSV might be slightly different or contain units
    # Usually it matches the requested metric name exactly in the 'Metric Name' column 
    # and value in 'Metric Value' column, OR it's a pivot table.
    
    # We are looking for rows where "Metric Name" == metric_name
    if 'Metric Name' in df_ncu.columns:
        df_ncu_metric = df_ncu[df_ncu['Metric Name'] == metric_name].copy()
        # Convert value to numeric
        df_ncu_metric['Metric Value'] = pd.to_numeric(df_ncu_metric['Metric Value'].astype(str).str.replace(',', ''), errors='coerce')
    else:
        # Sometimes ncu outputs a pivoted CSV where the metric name is the column header
        if metric_name in df_ncu.columns:
             df_ncu_metric = df_ncu.copy()
             df_ncu_metric['Metric Value'] = df_ncu_metric[metric_name]
             # Add dummy Metric Name column for consistency if needed, or just proceed
        else:
            print(f"Error: Metric '{metric_name}' not found in NCU CSV columns: {df_ncu.columns.tolist()}")
            return

    # Identify NVTX Range column
    nvtx_col = None
    
    # Priority check for user-specified column pattern
    for col in df_ncu_metric.columns:
        if "Id:Domain:Start/Stop_Range" in col:
            nvtx_col = col
            break
            
    # Fallback to generic search
    if not nvtx_col:
        for col in df_ncu_metric.columns:
            if "NVTX" in col and "Range" in col:
                nvtx_col = col
                break
    
    metric_series = None
    
    if nvtx_col:
        print(f"Grouping by NVTX Range column: {nvtx_col}")
        
        def sum_and_warn(x):
            # Check if multiple kernels have significant metric value (> 0.01 to avoid noise)
            non_zeros = x[x > 0.01].count()
            if non_zeros > 1:
                print(f"Warning: Multiple kernels ({non_zeros}) have non-zero metric in a single NVTX range.")
            return x.sum()

        # Group by NVTX Range ID and sum. sort=False to attempt to preserve order of appearance
        # Note: NVTX Range IDs usually increment, so sorting by them is also fine/safer.
        # But let's assume the file order is the execution order.
        df_ncu_ranges = df_ncu_metric.groupby(nvtx_col, sort=False)['Metric Value'].agg(sum_and_warn).reset_index()
        metric_series = df_ncu_ranges['Metric Value']
    else:
        print("Warning: NVTX Range column not found. Falling back to row-based grouping (assuming 1 kernel per range).")
        metric_series = df_ncu_metric['Metric Value'].reset_index(drop=True)

    # We assume the order of execution is preserved.
    # gemm_generalized runs: Config1 (1x), Config2 (1x), ...
    # So we group the *Ranges* by chunks of 1.
    
    runs_per_config = 1
    n_gemm_rows = len(df_gemm)
    n_ranges = len(metric_series)
    
    expected_ranges = n_gemm_rows * runs_per_config
    
    if n_ranges != expected_ranges:
        print(f"Warning: Mismatch in counts. GEMM rows: {n_gemm_rows}, NVTX Ranges: {n_ranges}.")
        print(f"Expected {expected_ranges} ranges (1 run per GEMM config).")
        print("Attempting to merge by aligning as much as possible...")

    # Group by every 1 range (effectively identity)
    # Create a DataFrame for grouping
    df_agg = pd.DataFrame({'val': metric_series})
    df_agg['group_id'] = df_agg.index // runs_per_config
    
    df_final = df_agg.groupby('group_id')['val'].mean().reset_index()
    
    # Now merge
    if len(df_final) > len(df_gemm):
        df_final = df_final.iloc[:len(df_gemm)]
    
    df_gemm['measured_mfu'] = df_final['val']
    
    # Save
    df_gemm.to_csv(merged_csv, index=False)
    print(f"Successfully saved merged results to {merged_csv}")
    print("\nSample of merged data:")
    print(df_gemm[['Tile', 'TFLOPS', 'measured_mfu']].head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GEMM benchmark with Nsight Compute profiling")
    parser.add_argument("--dtype", type=str, default="fp16", help="Data type to benchmark (fp32, fp16, bf16, int8)")
    parser.add_argument("--M", type=str, default="256:4096:256", help="M dimension range (start:end:step)")
    parser.add_argument("--N", type=str, default="256:4096:256", help="N dimension range (start:end:step)")
    parser.add_argument("--K", type=str, default="256:4096:256", help="K dimension range (start:end:step)")
    args = parser.parse_args()

    gemm_csv = "gemm_results.csv"
    ncu_csv = "ncu_report.csv"
    merged_csv = "gemm_merged_results.csv"

    metric = run_benchmark(args.dtype, args.M, args.N, args.K, gemm_csv, ncu_csv)
    merge_results(gemm_csv, ncu_csv, merged_csv, metric)
