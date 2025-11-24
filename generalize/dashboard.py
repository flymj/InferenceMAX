import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

st.set_page_config(layout="wide", page_title="GEMM Benchmark Dashboard")

st.title("GEMM Benchmark Analysis Dashboard")

@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

# Parse command line arguments
# Usage: streamlit run dashboard.py -- <csv_path>
if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    data_path = "gemm_full_scan_results.csv"

st.sidebar.markdown(f"**Data Source:** `{data_path}`")

df = load_data(data_path)

if df is None:
    st.error(f"Data file `{data_path}` not found.")
    st.info("Usage: `streamlit run dashboard.py -- <path_to_csv>`")
else:
    # Sidebar filters
    st.sidebar.header("Filters")
    
    if 'dtype' in df.columns:
        dtypes = st.sidebar.multiselect("Select Data Types", df['dtype'].unique(), default=df['dtype'].unique())
    else:
        st.warning("Column 'dtype' not found in CSV. Showing all data.")
        dtypes = []

    # Sort data for meaningful index visualization
    if {'M', 'N', 'K'}.issubset(df.columns):
        df = df.sort_values(by=['dtype', 'M', 'N', 'K'])
    df = df.reset_index(drop=True)
    df['Trial_Index'] = df.index
    
    # Extract Stage from Tile name if not present
    # Format is usually "MxNxK_S{Stage}" or "MxNxK_S{Stage}_SK{SplitK}"
    if 'Stage' not in df.columns and 'Tile' in df.columns:
        df['Stage'] = df['Tile'].apply(lambda x: int(x.split('_S')[1].split('_')[0]) if '_S' in x else 0)

    # Re-filter after sorting and feature extraction
    if 'dtype' in df.columns:
        filtered_df = df[df['dtype'].isin(dtypes)]
    else:
        filtered_df = df

    # Key Metrics Overview
    st.header("Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if 'TFLOPS' in filtered_df.columns:
            st.metric("Max TFLOPS", f"{filtered_df['TFLOPS'].max():.2f}")
    with col2:
        if 'MFU' in filtered_df.columns:
            st.metric("Max MFU (%)", f"{filtered_df['MFU'].max():.2f}%")
    with col3:
        if 'Time_ms' in filtered_df.columns:
            st.metric("Best Latency (ms)", f"{filtered_df['Time_ms'].min():.4f}")
    with col4:
        st.metric("Total Runs", len(filtered_df))

    # TFLOPS Analysis
    if 'TFLOPS' in filtered_df.columns:
        st.subheader("TFLOPS Analysis")
        
        c1, c2 = st.columns(2)
        with c1:
            x_axis = st.selectbox("X-Axis Variable", ["Trial_Index", "ProblemSize", "M", "N", "K"], index=0)
        with c2:
            color_options = ["Tile", "dtype", "SplitK", "Stage"]
            valid_colors = [c for c in color_options if c in filtered_df.columns]
            color_by = st.selectbox("Color By", valid_colors, index=0)

        if "ProblemSize" not in filtered_df.columns and "M" in filtered_df.columns:
             filtered_df['ProblemSize'] = filtered_df['M'] * filtered_df['N']

        hover_cols = ["M", "N", "K", "Tile"]
        if "SplitK" in filtered_df.columns: hover_cols.append("SplitK")
        if "Stage" in filtered_df.columns: hover_cols.append("Stage")

        fig_tflops = px.scatter(
            filtered_df, 
            x=x_axis, 
            y="TFLOPS", 
            color=color_by if color_by in filtered_df.columns else None,
            hover_data=hover_cols,
            title=f"TFLOPS vs {x_axis}"
        )
        st.plotly_chart(fig_tflops, use_container_width=True)

    # Stage Impact Analysis
    if 'Stage' in filtered_df.columns and 'TFLOPS' in filtered_df.columns:
        st.subheader("Impact of Pipeline Stages")
        fig_stage = px.box(
            filtered_df,
            x="Stage",
            y="TFLOPS",
            color="Stage",
            points="all",
            hover_data=["M", "N", "K", "Tile"],
            title="TFLOPS Distribution by Pipeline Stage"
        )
        st.plotly_chart(fig_stage, use_container_width=True)

    # Tile Impact Analysis
    if 'Tile' in filtered_df.columns and 'TFLOPS' in filtered_df.columns:
        st.subheader("Impact of Tile Configuration")
        st.markdown("Distribution of TFLOPS for each Tile configuration. Points represent individual benchmark runs.")
        
        fig_tile = px.box(
            filtered_df, 
            x="Tile", 
            y="TFLOPS", 
            color="Tile", 
            points="all", # Show all points to reveal density and periodicity
            hover_data=["M", "N", "K"],
            title="TFLOPS Distribution by Tile"
        )
        st.plotly_chart(fig_tile, use_container_width=True)

    # Detailed Analysis by Dtype
    if dtypes:
        st.subheader("Detailed Analysis by Data Type")
        
        tabs = st.tabs(dtypes)
        
        for i, dtype in enumerate(dtypes):
            with tabs[i]:
                dtype_df = df[df['dtype'] == dtype]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Top 5 Configs for {dtype} (by TFLOPS)**")
                    cols_to_show = ['M', 'N', 'K', 'Tile', 'TFLOPS', 'MFU']
                    if 'SplitK' in dtype_df.columns: cols_to_show.append('SplitK')
                    if 'Stage' in dtype_df.columns: cols_to_show.append('Stage')
                    
                    top_configs = dtype_df.nlargest(5, 'TFLOPS')[cols_to_show]
                    st.dataframe(top_configs)
                
                with col2:
                    # Faceted view for this dtype
                    st.markdown("**TFLOPS vs Index (Faceted by Stage)**")
                    if 'Stage' in dtype_df.columns:
                        fig_facet = px.scatter(
                            dtype_df,
                            x="Trial_Index",
                            y="TFLOPS",
                            color="Tile",
                            facet_col="Stage",
                            title=f"Performance Trends per Stage ({dtype})"
                        )
                        st.plotly_chart(fig_facet, use_container_width=True)
                    else:
                        st.info("Stage information not available.")
                
                # Heatmap of TFLOPS for M vs N (fixed K)
                if 'K' in dtype_df.columns:
                    unique_ks = dtype_df['K'].unique()
                    selected_k = st.selectbox(f"Select K for Heatmap ({dtype})", unique_ks, key=f"k_{dtype}")
                    
                    heatmap_data = dtype_df[dtype_df['K'] == selected_k].pivot_table(
                        index='M', columns='N', values='TFLOPS', aggfunc='max'
                    )
                    
                    fig_heat = px.imshow(
                        heatmap_data, 
                        labels=dict(x="N", y="M", color="TFLOPS"),
                        title=f"TFLOPS Heatmap (K={selected_k})"
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

    # Raw Data
    with st.expander("View Raw Data"):
        st.dataframe(filtered_df)
