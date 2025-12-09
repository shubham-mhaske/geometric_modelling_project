#!/usr/bin/env python3
"""
Generate Final Report Figures
Creates high-quality visualizations for the CSCE 645 Final Report.

Author: Shubham Vikas Mhaske
"""

import os
import sys
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Output directory
OUTPUT_DIR = "outputs/figures/final_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# DATA FROM EVALUATION
# ============================================================================

RESULTS = {
    "Laplacian": {
        "vol_mean": -0.96, "vol_std": 0.88,
        "smooth_mean": 83.8, "smooth_std": 2.8,
        "ar_mean": 13.6, "ar_std": 3.5,
        "time_mean": 28
    },
    "Taubin": {
        "vol_mean": 0.06, "vol_std": 0.05,
        "smooth_mean": 63.8, "smooth_std": 4.9,
        "ar_mean": 12.1, "ar_std": 1.1,
        "time_mean": 38
    },
    "Geodesic Heat": {
        "vol_mean": -0.42, "vol_std": 0.40,
        "smooth_mean": 73.6, "smooth_std": 4.1,
        "ar_mean": 13.6, "ar_std": 1.8,
        "time_mean": 29
    },
    "Anisotropic Tensor": {
        "vol_mean": -0.01, "vol_std": 0.01,
        "smooth_mean": 15.7, "smooth_std": 1.3,
        "ar_mean": 4.5, "ar_std": 0.3,
        "time_mean": 88
    },
    "Info-Theoretic": {
        "vol_mean": 0.04, "vol_std": 0.03,
        "smooth_mean": 56.8, "smooth_std": 5.1,
        "ar_mean": 10.9, "ar_std": 0.9,
        "time_mean": 72
    }
}

COLORS = {
    "Laplacian": "#667eea",
    "Taubin": "#764ba2",
    "Geodesic Heat": "#38ef7d",
    "Anisotropic Tensor": "#4facfe",
    "Info-Theoretic": "#f093fb"
}

# ============================================================================
# FIGURE 1: ALGORITHM COMPARISON BAR CHART
# ============================================================================

def create_algorithm_comparison():
    """Create grouped bar chart comparing all algorithms."""
    algorithms = list(RESULTS.keys())
    vol_changes = [RESULTS[a]["vol_mean"] for a in algorithms]
    smoothness = [RESULTS[a]["smooth_mean"] for a in algorithms]
    ar_improve = [RESULTS[a]["ar_mean"] for a in algorithms]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Volume Change (%)", "Smoothness Improvement (%)", "Aspect Ratio Improvement (%)"),
        horizontal_spacing=0.08
    )
    
    # Volume change
    vol_colors = ['#f5576c' if v < -0.5 else '#38ef7d' for v in vol_changes]
    fig.add_trace(go.Bar(
        x=algorithms, y=vol_changes,
        marker_color=vol_colors,
        error_y=dict(type='data', array=[RESULTS[a]["vol_std"] for a in algorithms]),
        showlegend=False
    ), row=1, col=1)
    
    # Smoothness
    fig.add_trace(go.Bar(
        x=algorithms, y=smoothness,
        marker_color=[COLORS[a] for a in algorithms],
        error_y=dict(type='data', array=[RESULTS[a]["smooth_std"] for a in algorithms]),
        showlegend=False
    ), row=1, col=2)
    
    # Aspect ratio
    fig.add_trace(go.Bar(
        x=algorithms, y=ar_improve,
        marker_color=[COLORS[a] for a in algorithms],
        error_y=dict(type='data', array=[RESULTS[a]["ar_std"] for a in algorithms]),
        showlegend=False
    ), row=1, col=3)
    
    fig.update_layout(
        title=dict(text="Algorithm Performance Comparison (20 BraTS Samples)", font=dict(size=20)),
        height=500,
        width=1400,
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,240,250,0.5)',
        font=dict(family="Inter", size=12)
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(gridcolor='rgba(0,0,0,0.1)')
    
    # Add zero line for volume change
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    fig.write_html(f"{OUTPUT_DIR}/fig1_algorithm_comparison.html")
    fig.write_image(f"{OUTPUT_DIR}/fig1_algorithm_comparison.png", scale=2)
    print("✅ Created Figure 1: Algorithm Comparison")
    return fig


# ============================================================================
# FIGURE 2: VOLUME VS SMOOTHNESS TRADE-OFF
# ============================================================================

def create_tradeoff_scatter():
    """Create scatter plot showing volume-smoothness trade-off."""
    algorithms = list(RESULTS.keys())
    
    fig = go.Figure()
    
    for algo in algorithms:
        data = RESULTS[algo]
        fig.add_trace(go.Scatter(
            x=[abs(data["vol_mean"])],
            y=[data["smooth_mean"]],
            mode='markers+text',
            name=algo,
            text=[algo],
            textposition='top center',
            marker=dict(
                size=data["time_mean"] / 2,  # Size by processing time
                color=COLORS[algo],
                line=dict(color='white', width=2),
                opacity=0.8
            ),
            textfont=dict(size=11)
        ))
    
    # Add ideal region annotation
    fig.add_shape(
        type="rect",
        x0=0, y0=60, x1=0.1, y1=85,
        fillcolor="rgba(56,239,125,0.1)",
        line=dict(color="rgba(56,239,125,0.5)", dash="dash")
    )
    fig.add_annotation(
        x=0.05, y=72.5, text="Ideal Region",
        showarrow=False, font=dict(color="#38ef7d", size=10)
    )
    
    fig.update_layout(
        title=dict(text="Volume Preservation vs Smoothing Trade-off", font=dict(size=18)),
        xaxis_title="Absolute Volume Change (%)",
        yaxis_title="Smoothness Improvement (%)",
        height=600,
        width=900,
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,240,250,0.5)',
        font=dict(family="Inter"),
        showlegend=True,
        legend=dict(x=1.02, y=0.5)
    )
    
    fig.update_xaxes(gridcolor='rgba(0,0,0,0.1)', range=[-0.05, 1.1])
    fig.update_yaxes(gridcolor='rgba(0,0,0,0.1)', range=[0, 100])
    
    fig.write_html(f"{OUTPUT_DIR}/fig2_tradeoff_scatter.html")
    fig.write_image(f"{OUTPUT_DIR}/fig2_tradeoff_scatter.png", scale=2)
    print("✅ Created Figure 2: Trade-off Scatter")
    return fig


# ============================================================================
# FIGURE 3: RADAR CHART COMPARISON
# ============================================================================

def create_radar_chart():
    """Create radar chart comparing algorithm properties."""
    categories = ['Volume Preservation', 'Smoothness', 'Mesh Quality', 'Speed', 'Consistency']
    
    fig = go.Figure()
    
    # Normalize metrics to 0-100 scale
    def normalize_metrics(algo):
        data = RESULTS[algo]
        # Volume: lower absolute change is better (invert)
        vol_score = max(0, 100 - abs(data["vol_mean"]) * 100)
        # Smoothness: already 0-100
        smooth_score = data["smooth_mean"]
        # AR: lower is better for triangle quality (invert and scale)
        ar_score = max(0, 100 - data["ar_mean"] * 5)
        # Speed: faster is better (invert, scale)
        speed_score = max(0, 100 - data["time_mean"])
        # Consistency: lower std is better
        consistency = max(0, 100 - data["vol_std"] * 50)
        return [vol_score, smooth_score, ar_score, speed_score, consistency]
    
    # Convert hex colors to rgba for fill
    def hex_to_rgba(hex_color, alpha=0.2):
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'
    
    for algo in RESULTS.keys():
        scores = normalize_metrics(algo)
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name=algo,
            line=dict(color=COLORS[algo], width=2),
            fillcolor=hex_to_rgba(COLORS[algo], 0.2)
        ))
    
    fig.update_layout(
        title=dict(text="Multi-Metric Algorithm Comparison", font=dict(size=18)),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(0,0,0,0.1)'),
            angularaxis=dict(gridcolor='rgba(0,0,0,0.1)')
        ),
        height=600,
        width=800,
        paper_bgcolor='white',
        font=dict(family="Inter"),
        legend=dict(x=1.1, y=0.5)
    )
    
    fig.write_html(f"{OUTPUT_DIR}/fig3_radar_comparison.html")
    fig.write_image(f"{OUTPUT_DIR}/fig3_radar_comparison.png", scale=2)
    print("✅ Created Figure 3: Radar Chart")
    return fig


# ============================================================================
# FIGURE 4: PROCESSING TIME COMPARISON
# ============================================================================

def create_time_comparison():
    """Create bar chart of processing times."""
    algorithms = list(RESULTS.keys())
    times = [RESULTS[a]["time_mean"] for a in algorithms]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=algorithms,
        y=times,
        marker_color=[COLORS[a] for a in algorithms],
        text=[f"{t}ms" for t in times],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(text="Processing Time Comparison", font=dict(size=18)),
        xaxis_title="Algorithm",
        yaxis_title="Time (milliseconds)",
        height=450,
        width=700,
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,240,250,0.5)',
        font=dict(family="Inter"),
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(gridcolor='rgba(0,0,0,0.1)')
    
    # Add "real-time threshold" line
    fig.add_hline(y=100, line_dash="dash", line_color="green",
                  annotation_text="Real-time threshold (100ms)")
    
    fig.write_html(f"{OUTPUT_DIR}/fig4_processing_time.html")
    fig.write_image(f"{OUTPUT_DIR}/fig4_processing_time.png", scale=2)
    print("✅ Created Figure 4: Processing Time")
    return fig


# ============================================================================
# FIGURE 5: ALGORITHM NOVELTY CONTRIBUTIONS
# ============================================================================

def create_novelty_diagram():
    """Create diagram showing novel contributions."""
    
    fig = go.Figure()
    
    # Algorithm boxes
    algorithms_data = [
        {"name": "Geodesic Heat", "novelty": "Heat kernel curvature-adaptive", "x": 0.2, "color": COLORS["Geodesic Heat"]},
        {"name": "Anisotropic Tensor", "novelty": "Tangential diffusion", "x": 0.5, "color": COLORS["Anisotropic Tensor"]},
        {"name": "Info-Theoretic", "novelty": "Shannon entropy weighting", "x": 0.8, "color": COLORS["Info-Theoretic"]}
    ]
    
    for algo in algorithms_data:
        # Box
        fig.add_shape(
            type="rect",
            x0=algo["x"]-0.12, y0=0.4, x1=algo["x"]+0.12, y1=0.7,
            fillcolor=algo["color"],
            line=dict(color="white", width=2),
            opacity=0.8
        )
        # Name
        fig.add_annotation(
            x=algo["x"], y=0.6,
            text=f"<b>{algo['name']}</b>",
            showarrow=False,
            font=dict(size=14, color="white")
        )
        # Novelty
        fig.add_annotation(
            x=algo["x"], y=0.5,
            text=algo["novelty"],
            showarrow=False,
            font=dict(size=10, color="white")
        )
    
    # Title
    fig.add_annotation(
        x=0.5, y=0.9,
        text="<b>Novel Algorithm Contributions</b>",
        showarrow=False,
        font=dict(size=20, color="#333")
    )
    
    # Key improvements
    improvements = [
        "82% better volume preservation",
        "15% better smoothing than Taubin",
        "First entropy-based feature detection"
    ]
    for i, imp in enumerate(improvements):
        fig.add_annotation(
            x=0.2 + i*0.3, y=0.25,
            text=f"✓ {imp}",
            showarrow=False,
            font=dict(size=11, color="#38ef7d")
        )
    
    fig.update_layout(
        height=400,
        width=900,
        paper_bgcolor='white',
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    fig.write_html(f"{OUTPUT_DIR}/fig5_novelty_diagram.html")
    fig.write_image(f"{OUTPUT_DIR}/fig5_novelty_diagram.png", scale=2)
    print("✅ Created Figure 5: Novelty Diagram")
    return fig


# ============================================================================
# FIGURE 6: PIPELINE ARCHITECTURE
# ============================================================================

def create_pipeline_diagram():
    """Create pipeline architecture diagram."""
    
    fig = go.Figure()
    
    steps = [
        {"name": "NIfTI Input", "icon": "📥", "x": 0.1},
        {"name": "Marching Cubes", "icon": "🔲", "x": 0.28},
        {"name": "Novel Smoothing", "icon": "✨", "x": 0.46},
        {"name": "Quality Metrics", "icon": "📊", "x": 0.64},
        {"name": "3D Visualization", "icon": "🖥️", "x": 0.82}
    ]
    
    # Draw boxes and arrows
    for i, step in enumerate(steps):
        # Box
        fig.add_shape(
            type="rect",
            x0=step["x"]-0.07, y0=0.35, x1=step["x"]+0.07, y1=0.65,
            fillcolor="rgba(102,126,234,0.2)",
            line=dict(color="#667eea", width=2)
        )
        # Icon
        fig.add_annotation(
            x=step["x"], y=0.55,
            text=step["icon"],
            showarrow=False,
            font=dict(size=24)
        )
        # Name
        fig.add_annotation(
            x=step["x"], y=0.42,
            text=step["name"],
            showarrow=False,
            font=dict(size=11, color="#333")
        )
        
        # Arrow to next
        if i < len(steps) - 1:
            fig.add_annotation(
                x=step["x"]+0.09, y=0.5,
                ax=step["x"]+0.05, ay=0.5,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowcolor="#38ef7d"
            )
    
    # Title
    fig.add_annotation(
        x=0.5, y=0.85,
        text="<b>Mesh Improvement Pipeline Architecture</b>",
        showarrow=False,
        font=dict(size=18, color="#333")
    )
    
    fig.update_layout(
        height=350,
        width=1000,
        paper_bgcolor='white',
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    fig.write_html(f"{OUTPUT_DIR}/fig6_pipeline.html")
    fig.write_image(f"{OUTPUT_DIR}/fig6_pipeline.png", scale=2)
    print("✅ Created Figure 6: Pipeline Diagram")
    return fig


# ============================================================================
# FIGURE 7: RESULTS SUMMARY TABLE
# ============================================================================

def create_summary_table():
    """Create visual summary table."""
    
    algorithms = list(RESULTS.keys())
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Algorithm</b>', '<b>Volume Change</b>', '<b>Smoothness</b>', 
                   '<b>AR Improve</b>', '<b>Time</b>', '<b>Type</b>'],
            fill_color='#667eea',
            font=dict(color='white', size=12),
            align='center',
            height=35
        ),
        cells=dict(
            values=[
                algorithms,
                [f"{RESULTS[a]['vol_mean']:+.2f}%" for a in algorithms],
                [f"{RESULTS[a]['smooth_mean']:.1f}%" for a in algorithms],
                [f"{RESULTS[a]['ar_mean']:.1f}%" for a in algorithms],
                [f"{RESULTS[a]['time_mean']}ms" for a in algorithms],
                ['Baseline', 'Baseline', 'Novel ✨', 'Novel ✨', 'Novel ✨']
            ],
            fill_color=[
                ['white']*5,
                [('#ffe6e6' if RESULTS[a]['vol_mean'] < -0.5 else '#e6ffe6') for a in algorithms],
                ['white']*5,
                ['white']*5,
                ['white']*5,
                [('#e6ffe6' if 'Novel' in ('Novel' if i >= 2 else 'Baseline') else 'white') for i in range(5)]
            ],
            font=dict(size=11),
            align='center',
            height=30
        )
    )])
    
    fig.update_layout(
        title=dict(text="Algorithm Performance Summary", font=dict(size=16)),
        height=300,
        width=900,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    fig.write_html(f"{OUTPUT_DIR}/fig7_summary_table.html")
    fig.write_image(f"{OUTPUT_DIR}/fig7_summary_table.png", scale=2)
    print("✅ Created Figure 7: Summary Table")
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("  GENERATING FINAL REPORT FIGURES")
    print("="*60 + "\n")
    
    create_algorithm_comparison()
    create_tradeoff_scatter()
    create_radar_chart()
    create_time_comparison()
    create_novelty_diagram()
    create_pipeline_diagram()
    create_summary_table()
    
    print("\n" + "="*60)
    print(f"  ✅ All figures saved to: {OUTPUT_DIR}")
    print("="*60 + "\n")
    
    # List generated files
    files = os.listdir(OUTPUT_DIR)
    print("Generated files:")
    for f in sorted(files):
        print(f"  📄 {f}")


if __name__ == "__main__":
    main()
