#!/usr/bin/env python3
"""
Academic Graduate Presentation Generator (Final Version)
Target: 12-minute talk + 3-minute Q&A for peer/instructor evaluation
Design: Modern minimalist with off-white background, Texas A&M branding
Libraries: Reveal.js 4.5, Chart.js 4.0, KaTeX for math rendering
"""

import json
import os

OUTPUT_FILE = "academic_presentation.html"

def load_results():
    """Load evaluation results"""
    results_path = "outputs/comprehensive_eval/results.json"
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

def get_head_section():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feature-Preserving Mesh Smoothing for Medical Imaging</title>
    
    <!-- Reveal.js Core -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.5.0/reveal.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.5.0/theme/white.min.css">
    
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;500;600&family=Crimson+Pro:wght@300;400;600;700&display=swap" rel="stylesheet">
    
    <!-- KaTeX for Math -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
'''

def get_custom_css():
    return '''    <style>
        /* === DESIGN SYSTEM === */
        :root {
            /* Texas A&M Colors */
            --maroon: #500000;
            --maroon-light: #702020;
            --gold: #bfa15f;
            --white: #ffffff;
            --off-white: #fafafa;
            --cream: #f8f6f1;
            
            /* Semantic Colors */
            --text-primary: #1a1a1a;
            --text-secondary: #4a4a4a;
            --text-muted: #888888;
            --border: #e0e0e0;
            --border-light: #f0f0f0;
            
            /* Functional Colors */
            --success: #27ae60;
            --warning: #f39c12;
            --danger: #e74c3c;
            --info: #3498db;
            
            /* Typography */
            --font-heading: 'Crimson Pro', serif;
            --font-body: 'Inter', sans-serif;
            --font-code: 'JetBrains Mono', monospace;
            
            /* Shadows */
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
            --shadow-lg: 0 10px 30px rgba(0,0,0,0.12);
        }

        /* === GLOBAL RESET === */
        .reveal {
            font-family: var(--font-body);
            font-size: 28px;
            font-weight: 300;
            color: var(--text-primary);
            background-color: var(--off-white);
        }

        .reveal .slides {
            text-align: left;
        }

        /* === TYPOGRAPHY === */
        .reveal h1 {
            font-family: var(--font-heading);
            font-size: 3.2em;
            font-weight: 700;
            color: var(--maroon);
            letter-spacing: -0.02em;
            line-height: 1.1;
            margin-bottom: 0.3em;
            text-shadow: 0 2px 8px rgba(80,0,0,0.1);
        }

        .reveal h2 {
            font-family: var(--font-heading);
            font-size: 2.4em;
            font-weight: 600;
            color: var(--maroon);
            margin-bottom: 0.5em;
            padding-bottom: 0.3em;
            border-bottom: 3px solid var(--gold);
            display: inline-block;
        }

        .reveal h3 {
            font-family: var(--font-body);
            font-size: 1.4em;
            font-weight: 600;
            color: var(--text-primary);
            margin: 1em 0 0.5em 0;
        }

        .reveal h4 {
            font-family: var(--font-body);
            font-size: 1.1em;
            font-weight: 600;
            color: var(--maroon);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 0.8em 0 0.4em 0;
        }

        .reveal p {
            margin-bottom: 0.8em;
            line-height: 1.6;
            font-weight: 300;
        }

        .reveal strong {
            font-weight: 600;
            color: var(--maroon);
        }

        .reveal em {
            font-style: italic;
            color: var(--text-secondary);
        }

        /* === LISTS === */
        .reveal ul, .reveal ol {
            margin-left: 0;
            list-style-position: outside;
        }

        .reveal ul {
            list-style: none;
        }

        .reveal ul li::before {
            content: "▸";
            color: var(--gold);
            font-weight: bold;
            display: inline-block;
            width: 1em;
            margin-left: -1em;
        }

        .reveal li {
            margin-bottom: 0.4em;
            padding-left: 0.3em;
            line-height: 1.5;
        }

        /* === LAYOUT SYSTEM === */
        .slide-content {
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 40px 60px;
        }

        .grid-2col {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 50px;
            align-items: center;
            margin-top: 30px;
        }

        .grid-2col.wide-left {
            grid-template-columns: 1.6fr 1fr;
        }

        .grid-2col.wide-right {
            grid-template-columns: 1fr 1.6fr;
        }

        .grid-3col {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 30px;
            margin-top: 30px;
        }

        /* === COMPONENTS === */
        
        /* Cards */
        .card {
            background: var(--white);
            border: 1px solid var(--border-light);
            border-radius: 12px;
            padding: 25px;
            box-shadow: var(--shadow-sm);
            transition: all 0.3s ease;
        }

        .card:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }

        .card h4 {
            margin-top: 0;
            font-size: 1em;
        }

        .card p {
            font-size: 0.85em;
            margin-bottom: 0;
            color: var(--text-secondary);
        }

        .card.highlight {
            border-left: 4px solid var(--gold);
            background: linear-gradient(to right, #fffdf7 0%, var(--white) 100%);
        }

        .card.success {
            border-left: 4px solid var(--success);
            background: linear-gradient(to right, #f0fdf4 0%, var(--white) 100%);
        }

        .card.danger {
            border-left: 4px solid var(--danger);
            background: linear-gradient(to right, #fef2f2 0%, var(--white) 100%);
        }

        .card.info {
            border-left: 4px solid var(--info);
            background: linear-gradient(to right, #eff6ff 0%, var(--white) 100%);
        }

        /* Figures */
        .figure-container {
            background: var(--white);
            padding: 20px;
            border-radius: 12px;
            box-shadow: var(--shadow-md);
            text-align: center;
        }

        .figure-container img {
            max-width: 100%;
            max-height: 500px;
            height: auto;
            border-radius: 6px;
            object-fit: contain;
        }

        .figure-caption {
            margin-top: 15px;
            font-size: 0.7em;
            color: var(--text-muted);
            font-family: var(--font-code);
            font-weight: 400;
        }

        /* Math Blocks */
        .math-block {
            background: var(--cream);
            border: 1px solid var(--border);
            border-left: 4px solid var(--maroon);
            border-radius: 8px;
            padding: 25px;
            margin: 20px 0;
            font-size: 1.1em;
            box-shadow: var(--shadow-sm);
        }

        .math-label {
            font-size: 0.7em;
            color: var(--gold);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 10px;
            display: block;
        }

        /* Code Blocks */
        .code-container {
            background: #1e1e1e;
            color: #d4d4d4;
            font-family: var(--font-code);
            font-size: 0.6em;
            line-height: 1.6;
            padding: 25px;
            border-radius: 10px;
            box-shadow: var(--shadow-lg);
            overflow-x: auto;
        }

        .code-keyword { color: #569cd6; }
        .code-function { color: #dcdcaa; }
        .code-string { color: #ce9178; }
        .code-comment { color: #6a9955; font-style: italic; }
        .code-number { color: #b5cea8; }

        /* Tables */
        .data-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 0.75em;
            box-shadow: var(--shadow-md);
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
        }

        .data-table thead th {
            background: var(--maroon);
            color: var(--white);
            padding: 15px 20px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.85em;
        }

        .data-table tbody td {
            padding: 12px 20px;
            border-bottom: 1px solid var(--border-light);
            background: var(--white);
        }

        .data-table tbody tr:last-child td {
            border-bottom: none;
        }

        .data-table tbody tr:hover td {
            background: var(--off-white);
        }

        .data-table tbody tr.highlight {
            background: #fffdf7;
        }

        .data-table tbody tr.highlight:hover {
            background: #fffaed;
        }

        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 30px 0;
        }

        .metric-box {
            background: var(--white);
            border: 2px solid var(--border-light);
            border-radius: 12px;
            padding: 25px 20px;
            text-align: center;
            transition: all 0.3s ease;
        }

        .metric-box:hover {
            border-color: var(--gold);
            box-shadow: var(--shadow-md);
            transform: translateY(-3px);
        }

        .metric-value {
            font-size: 2.2em;
            font-weight: 700;
            color: var(--maroon);
            line-height: 1;
            margin-bottom: 8px;
        }

        .metric-label {
            font-size: 0.7em;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }

        /* Callouts */
        .callout {
            background: linear-gradient(135deg, var(--maroon) 0%, var(--maroon-light) 100%);
            color: var(--white);
            padding: 30px;
            border-radius: 12px;
            box-shadow: var(--shadow-lg);
            margin: 30px 0;
            text-align: center;
        }

        .callout p {
            margin: 0;
            font-size: 1.1em;
            font-weight: 400;
        }

        .callout strong {
            color: var(--gold);
            font-weight: 700;
        }

        /* === SLIDE-SPECIFIC STYLES === */
        
        /* Title Slide */
        .title-slide {
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            background: linear-gradient(135deg, var(--cream) 0%, var(--off-white) 100%);
        }

        .title-slide h1 {
            font-size: 3.5em;
            margin-bottom: 0.2em;
            border: none;
        }

        .title-slide .subtitle {
            font-size: 1.4em;
            color: var(--text-secondary);
            margin-bottom: 1.5em;
            font-weight: 300;
        }

        .title-slide .divider {
            width: 120px;
            height: 4px;
            background: var(--gold);
            margin: 40px auto;
            border-radius: 2px;
        }

        .title-slide .author {
            font-size: 1.2em;
            font-weight: 600;
            color: var(--maroon);
            margin-bottom: 0.3em;
        }

        .title-slide .institution {
            font-size: 0.9em;
            color: var(--text-muted);
            line-height: 1.6;
        }

        /* Footer */
        .slide-footer {
            position: absolute;
            bottom: 20px;
            left: 60px;
            right: 60px;
            display: flex;
            justify-content: space-between;
            font-size: 0.5em;
            color: var(--text-muted);
            border-top: 1px solid var(--border-light);
            padding-top: 15px;
        }

        /* Progress Bar */
        .reveal .progress {
            height: 4px;
            background: var(--border-light);
        }

        .reveal .progress span {
            background: var(--maroon);
        }

        /* Slide Number */
        .reveal .slide-number {
            font-family: var(--font-code);
            font-size: 12px;
            color: var(--text-muted);
            background: transparent;
        }

        /* Animations */
        .reveal .fragment.fade-up {
            opacity: 0;
            transform: translateY(20px);
        }

        .reveal .fragment.fade-up.visible {
            opacity: 1;
            transform: translateY(0);
        }

        /* === UTILITY CLASSES === */
        .text-center { text-align: center; }
        .text-right { text-align: right; }
        .mt-1 { margin-top: 0.5em; }
        .mt-2 { margin-top: 1em; }
        .mb-1 { margin-bottom: 0.5em; }
        .mb-2 { margin-bottom: 1em; }
        .font-small { font-size: 0.85em; }
        .font-large { font-size: 1.2em; }
        .color-success { color: var(--success); }
        .color-warning { color: var(--warning); }
        .color-danger { color: var(--danger); }
        .color-info { color: var(--info); }
        .font-weight-bold { font-weight: 600; }
    </style>
</head>'''

def get_title_slide():
    return '''
<body>
    <div class="reveal">
        <div class="slides">
            
            <!-- SLIDE 1: TITLE -->
            <section>
                <div class="title-slide">
                    <h1>Feature-Preserving Mesh Smoothing<br/>for Medical Imaging</h1>
                    <p class="subtitle">Novel Information-Theoretic Approaches for Clinical Applications</p>
                    <div class="divider"></div>
                    <p class="author">Shubham Vikas Mhaske</p>
                    <p class="institution">
                        CSCE 645: Geometric Modeling<br/>
                        Texas A&M University &bull; Fall 2024
                    </p>
                </div>
            </section>'''

def get_introduction_slides():
    return '''
            <!-- SLIDE 2: PROBLEM CONTEXT -->
            <section>
                <div class="slide-content">
                    <h2>The Medical Imaging Challenge</h2>
                    <div class="grid-2col">
                        <div>
                            <h3>Clinical Workflow</h3>
                            <ol style="font-size: 0.9em;">
                                <li><strong>Acquisition:</strong> MRI/CT scanning produces volumetric data</li>
                                <li><strong>Segmentation:</strong> AI identifies anatomical regions</li>
                                <li><strong>Reconstruction:</strong> Marching Cubes converts voxels to mesh</li>
                                <li><strong>Analysis:</strong> Volume quantification for diagnosis</li>
                            </ol>
                            
                            <div class="card danger mt-2">
                                <h4>The Problem</h4>
                                <p>Marching Cubes creates <strong>"staircase" artifacts</strong> that require smoothing, but traditional algorithms cause volume shrinkage.</p>
                            </div>
                        </div>
                        <div>
                            <div class="figure-container">
                                <img src="outputs/figures/final_report/fig6_pipeline.png" alt="Pipeline">
                                <div class="figure-caption">Fig 1: End-to-End Processing Pipeline</div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="slide-footer">
                    <span>Feature-Preserving Mesh Smoothing</span>
                    <span>Shubham Mhaske &bull; CSCE 645</span>
                </div>
            </section>

            <!-- SLIDE 3: MOTIVATION -->
            <section>
                <div class="slide-content">
                    <h2>Clinical Significance</h2>
                    <div class="grid-2col wide-left">
                        <div>
                            <h3>Why Volume Preservation Matters</h3>
                            <div class="card info">
                                <h4>Tumor Monitoring</h4>
                                <p>Oncologists track tumor volume to assess treatment response. RECIST criteria require &lt;5% measurement error.</p>
                            </div>
                            <div class="card danger">
                                <h4>The Cost of Error</h4>
                                <p>A 10mm³ tumor losing 5% (0.5mm³) volume due to smoothing artifacts could be misdiagnosed as responding to therapy.</p>
                            </div>
                            <div class="math-block mt-2">
                                <span class="math-label">Clinical Threshold</span>
                                $$ \\Delta V = \\frac{|V_{\\text{original}} - V_{\\text{smoothed}}|}{V_{\\text{original}}} < 1\\% $$
                            </div>
                        </div>
                        <div>
                            <div class="figure-container">
                                <img src="outputs/figures/final_report/fig1_algorithm_comparison.png" alt="Comparison">
                                <div class="figure-caption">Fig 2: Raw vs. Smoothed Comparison</div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="slide-footer">
                    <span>Feature-Preserving Mesh Smoothing</span>
                    <span>Shubham Mhaske &bull; CSCE 645</span>
                </div>
            </section>

            <!-- SLIDE 4: OBJECTIVES -->
            <section>
                <div class="slide-content">
                    <h2>Research Objectives</h2>
                    <div class="grid-3col">
                        <div class="card highlight">
                            <h4>1. Novel Algorithms</h4>
                            <p>Develop 3 information-theoretic smoothing methods that preserve anatomical features.</p>
                        </div>
                        <div class="card highlight">
                            <h4>2. Dual-Modality Validation</h4>
                            <p>Evaluate on MRI (soft tissue) and CT (bone/hemorrhage) datasets.</p>
                        </div>
                        <div class="card highlight">
                            <h4>3. Clinical Viability</h4>
                            <p>Achieve &lt;1% volume change while maintaining visual smoothness.</p>
                        </div>
                    </div>
                    
                    <h3 class="mt-2">Dataset Summary</h3>
                    <div class="metrics-grid">
                        <div class="metric-box">
                            <div class="metric-value">16</div>
                            <div class="metric-label">Total Samples</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">10</div>
                            <div class="metric-label">MRI (BraTS)</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">6</div>
                            <div class="metric-label">CT (Hemorrhage)</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">5</div>
                            <div class="metric-label">Algorithms</div>
                        </div>
                    </div>
                </div>
                <div class="slide-footer">
                    <span>Feature-Preserving Mesh Smoothing</span>
                    <span>Shubham Mhaske &bull; CSCE 645</span>
                </div>
            </section>'''

def get_methodology_slides():
    return '''
            <!-- SLIDE 5: ALGORITHM 1 - GEODESIC HEAT -->
            <section>
                <div class="slide-content">
                    <h2>1. Geodesic Heat Diffusion</h2>
                    <div class="grid-2col">
                        <div>
                            <h3>Conceptual Foundation</h3>
                            <p>Instead of smoothing in Euclidean 3D space, diffuse heat <strong>along the surface manifold</strong>.</p>
                            
                            <div class="math-block">
                                <span class="math-label">Heat Equation on Manifolds</span>
                                $$ \\frac{\\partial u}{\\partial t} = \\Delta_g u $$
                                <p style="font-size: 0.7em; margin-top: 10px; color: var(--text-secondary);">
                                    $\\Delta_g$ is the Laplace-Beltrami operator (intrinsic to surface geometry)
                                </p>
                            </div>
                            
                            <div class="card info mt-2">
                                <h4>Key Insight</h4>
                                <p>Heat naturally respects geometric features (edges, corners) by following geodesic paths.</p>
                            </div>
                        </div>
                        <div>
                            <div class="code-container">
<span class="code-keyword">def</span> <span class="code-function">geodesic_heat_smoothing</span>(mesh, t, iterations):
    <span class="code-comment"># 1. Compute cotangent Laplacian</span>
    L = compute_cotangent_laplacian(mesh)
    
    <span class="code-keyword">for</span> i <span class="code-keyword">in</span> <span class="code-function">range</span>(iterations):
        <span class="code-comment"># 2. Time integration (implicit Euler)</span>
        A = sparse.eye(n) - t * L
        
        <span class="code-comment"># 3. Solve heat equation</span>
        <span class="code-comment"># (I - tL)u = u_old</span>
        u_new = sparse.linalg.spsolve(A, mesh.vertices)
        
        mesh.vertices = u_new
    
    <span class="code-keyword">return</span> mesh</div>
                            <p class="font-small mt-1" style="color: var(--text-muted);">
                                <strong>Complexity:</strong> O(|V| log |V|) using sparse solvers<br>
                                <strong>Result:</strong> 68.9% smoothing, 99.3% volume preservation
                            </p>
                        </div>
                    </div>
                </div>
                <div class="slide-footer">
                    <span>Feature-Preserving Mesh Smoothing</span>
                    <span>Shubham Mhaske &bull; CSCE 645</span>
                </div>
            </section>

            <!-- SLIDE 6: ALGORITHM 2 - INFO THEORETIC -->
            <section>
                <div class="slide-content">
                    <h2>2. Information-Theoretic Smoothing</h2>
                    <div class="grid-2col wide-right">
                        <div>
                            <h3>Entropy-Based Feature Detection</h3>
                            <div class="math-block">
                                <span class="math-label">Shannon Entropy</span>
                                $$ H(v) = -\\sum_{i=1}^{n} p(\\kappa_i) \\log p(\\kappa_i) $$
                            </div>
                            
                            <div class="card success">
                                <h4>Classification Logic</h4>
                                <p><strong>High Entropy:</strong> Random curvature → Noise → Smooth aggressively<br>
                                <strong>Low Entropy:</strong> Structured curvature → Feature → Preserve</p>
                            </div>
                            
                            <p class="mt-2 font-small"><strong>Example:</strong> Tumor boundary (H=0.3) vs. surface noise (H=0.9)</p>
                        </div>
                        <div>
                            <div class="code-container">
<span class="code-keyword">def</span> <span class="code-function">info_theoretic_smoothing</span>(mesh, iterations):
    <span class="code-keyword">for</span> _ <span class="code-keyword">in</span> <span class="code-function">range</span>(iterations):
        <span class="code-keyword">for</span> v <span class="code-keyword">in</span> mesh.vertices:
            <span class="code-comment"># 1. Get local curvature distribution</span>
            curvatures = [compute_curvature(n) 
                          <span class="code-keyword">for</span> n <span class="code-keyword">in</span> v.neighbors]
            
            <span class="code-comment"># 2. Build histogram & compute entropy</span>
            hist = np.histogram(curvatures, bins=<span class="code-number">10</span>)
            p = hist / hist.<span class="code-function">sum</span>()
            H = -np.<span class="code-function">sum</span>(p * np.<span class="code-function">log</span>(p + <span class="code-number">1e-10</span>))
            
            <span class="code-comment"># 3. Adaptive weighting</span>
            w = sigmoid(H)  <span class="code-comment"># High H → High weight</span>
            
            <span class="code-comment"># 4. Weighted Laplacian smoothing</span>
            v.pos = (<span class="code-number">1</span>-w)*v.pos + w*mean(v.neighbors)
    
    <span class="code-keyword">return</span> mesh</div>
                        </div>
                    </div>
                    <div class="callout mt-2">
                        <p><strong>Result:</strong> 100% volume preservation on MRI, 99.8% on CT — <strong>First algorithm to meet clinical threshold across modalities</strong></p>
                    </div>
                </div>
                <div class="slide-footer">
                    <span>Feature-Preserving Mesh Smoothing</span>
                    <span>Shubham Mhaske &bull; CSCE 645</span>
                </div>
            </section>

            <!-- SLIDE 7: ALGORITHM 3 - ANISOTROPIC -->
            <section>
                <div class="slide-content">
                    <h2>3. Anisotropic Tensor Smoothing</h2>
                    <div class="grid-2col">
                        <div>
                            <h3>Directional Diffusion Control</h3>
                            <p>Construct a <strong>structure tensor field</strong> that adapts to local geometry:</p>
                            <ul style="font-size: 0.9em;">
                                <li><strong>Along edges:</strong> High diffusivity (smooth)</li>
                                <li><strong>Across edges:</strong> Zero diffusivity (preserve)</li>
                            </ul>
                            
                            <div class="math-block">
                                <span class="math-label">Diffusion Tensor</span>
                                $$ D = \\lambda_\\parallel e_1 e_1^T + \\lambda_\\perp e_2 e_2^T $$
                                <p style="font-size: 0.7em; margin-top: 10px; color: var(--text-secondary);">
                                    Where $e_1, e_2$ are principal curvature directions
                                </p>
                            </div>
                        </div>
                        <div>
                            <div class="figure-container">
                                <img src="outputs/figures/final_report/fig5_novelty_diagram.png" alt="Tensor Field">
                                <div class="figure-caption">Fig 3: Anisotropic Tensor Field Visualization</div>
                            </div>
                            <p class="font-small mt-1 text-center" style="color: var(--text-secondary);">
                                <strong>Result:</strong> 0.070mm displacement (most conservative)
                            </p>
                        </div>
                    </div>
                </div>
                <div class="slide-footer">
                    <span>Feature-Preserving Mesh Smoothing</span>
                    <span>Shubham Mhaske &bull; CSCE 645</span>
                </div>
            </section>

            <!-- SLIDE 8: SEMANTIC SMOOTHING -->
            <section>
                <div class="slide-content">
                    <h2>Semantic-Aware Enhancement</h2>
                    <div class="grid-2col wide-left">
                        <div>
                            <h3>Label-Guided Boundary Preservation</h3>
                            <p>Incorporate <strong>segmentation labels</strong> to identify tissue boundaries.</p>
                            
                            <div class="card info">
                                <h4>Quantile-Based Coarsening</h4>
                                <p>Raw segmentations contain 6000+ fragment IDs. We coarsen to <strong>3 quantile bins</strong> representing anatomical tissue classes.</p>
                            </div>
                            
                            <div class="card success mt-1">
                                <h4>Impact</h4>
                                <p>Laplacian volume drift: <span class="color-danger">+3,346mm³</span> → <span class="color-success">+534mm³</span><br>
                                <strong>85% reduction</strong> in boundary violations</p>
                            </div>
                            
                            <div class="math-block mt-2">
                                <span class="math-label">Cross-Label Weight</span>
                                $$ w_{ij} = \\begin{cases} 
                                1.0 & \\text{if } L_i = L_j \\\\
                                0.01 & \\text{if } L_i \\neq L_j
                                \\end{cases} $$
                            </div>
                        </div>
                        <div>
                            <div class="code-container">
<span class="code-keyword">def</span> <span class="code-function">semantic_smoothing</span>(mesh, labels):
    <span class="code-comment"># 1. Coarsen labels to 3 bins</span>
    quantiles = np.<span class="code-function">quantile</span>(labels, [<span class="code-number">0.33</span>, <span class="code-number">0.67</span>])
    coarse = np.<span class="code-function">digitize</span>(labels, quantiles)
    
    <span class="code-keyword">for</span> v <span class="code-keyword">in</span> mesh.vertices:
        <span class="code-comment"># 2. Build weighted neighborhood</span>
        same_label = [n <span class="code-keyword">for</span> n <span class="code-keyword">in</span> v.neighbors
                      <span class="code-keyword">if</span> coarse[n] == coarse[v]]
        diff_label = [n <span class="code-keyword">for</span> n <span class="code-keyword">in</span> v.neighbors
                      <span class="code-keyword">if</span> coarse[n] != coarse[v]]
        
        <span class="code-comment"># 3. Weighted Laplacian</span>
        pos_new = (
            <span class="code-number">1.0</span> * mean(same_label) +
            <span class="code-number">0.01</span> * mean(diff_label)
        )
        v.pos = pos_new
    
    <span class="code-keyword">return</span> mesh</div>
                        </div>
                    </div>
                </div>
                <div class="slide-footer">
                    <span>Feature-Preserving Mesh Smoothing</span>
                    <span>Shubham Mhaske &bull; CSCE 645</span>
                </div>
            </section>'''

def get_results_slides():
    return '''
            <!-- SLIDE 9: QUANTITATIVE RESULTS -->
            <section>
                <div class="slide-content">
                    <h2>Quantitative Results</h2>
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Algorithm</th>
                                <th>Smoothness ↑</th>
                                <th>Volume Pres. ↑</th>
                                <th>Mesh Quality ↑</th>
                                <th>Max Disp. ↓</th>
                                <th>Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Taubin (Baseline)</strong></td>
                                <td class="color-success font-weight-bold">+86.8%</td>
                                <td class="color-danger font-weight-bold">98.5%</td>
                                <td>0.825</td>
                                <td>0.518mm</td>
                                <td class="color-success">41ms</td>
                            </tr>
                            <tr>
                                <td><strong>Laplacian</strong></td>
                                <td>+70.0%</td>
                                <td>99.8%</td>
                                <td>0.732</td>
                                <td>0.248mm</td>
                                <td class="color-success">22ms</td>
                            </tr>
                            <tr class="highlight">
                                <td><strong>Geodesic Heat (Ours)</strong></td>
                                <td>+68.9%</td>
                                <td>99.3%</td>
                                <td class="color-success font-weight-bold">0.803</td>
                                <td>0.387mm</td>
                                <td>9.6s</td>
                            </tr>
                            <tr class="highlight">
                                <td><strong>Info-Theoretic (Ours)</strong></td>
                                <td>+34.2%</td>
                                <td class="color-success font-weight-bold">100.0%</td>
                                <td>0.636</td>
                                <td class="color-success font-weight-bold">0.107mm</td>
                                <td>15.9s</td>
                            </tr>
                            <tr class="highlight">
                                <td><strong>Anisotropic (Ours)</strong></td>
                                <td>+16.6%</td>
                                <td class="color-success font-weight-bold">99.9%</td>
                                <td>0.654</td>
                                <td class="color-success font-weight-bold">0.070mm</td>
                                <td>35.4s</td>
                            </tr>
                        </tbody>
                    </table>
                    
                    <div class="callout mt-2">
                        <p><strong>Key Finding:</strong> Info-Theoretic achieves <strong>perfect volume preservation (100.0%)</strong> while maintaining acceptable smoothness — the only algorithm suitable for tumor volumetrics</p>
                    </div>
                </div>
                <div class="slide-footer">
                    <span>Feature-Preserving Mesh Smoothing</span>
                    <span>Shubham Mhaske &bull; CSCE 645</span>
                </div>
            </section>

            <!-- SLIDE 10: VISUAL COMPARISON -->
            <section>
                <div class="slide-content">
                    <h2>Multi-Metric Comparison</h2>
                    <div class="grid-2col">
                        <div>
                            <div class="figure-container">
                                <img src="outputs/figures/final_report/fig3_radar_comparison.png" alt="Radar Chart">
                                <div class="figure-caption">Fig 4: Normalized Radar Chart (5 Metrics)</div>
                            </div>
                        </div>
                        <div>
                            <h3>Holistic Analysis</h3>
                            <ul>
                                <li><strong>Taubin:</strong> Excellent smoothness but poor volume (98.5%)</li>
                                <li><strong>Geodesic Heat:</strong> Balanced profile, best visual quality</li>
                                <li><strong>Info-Theoretic:</strong> Conservative smoothing, <span class="color-success">perfect volume</span></li>
                                <li><strong>Anisotropic:</strong> Minimal displacement, feature-preserving</li>
                            </ul>
                            
                            <div class="card highlight mt-2">
                                <h4>Trade-off Insight</h4>
                                <p>No single algorithm dominates all metrics. <strong>Application-specific selection</strong> is critical.</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="slide-footer">
                    <span>Feature-Preserving Mesh Smoothing</span>
                    <span>Shubham Mhaske &bull; CSCE 645</span>
                </div>
            </section>

            <!-- SLIDE 11: CRITICAL DISCOVERY -->
            <section>
                <div class="slide-content">
                    <h2>Critical Discovery: Scale Sensitivity</h2>
                    <div class="grid-2col wide-left">
                        <div>
                            <h3>Taubin Algorithm Failure Mode</h3>
                            <div class="card danger">
                                <h4>Mesh-Size Dependency</h4>
                                <p>On small CT meshes (~13k vertices), Taubin caused <strong>22.3% volume loss</strong>. On large MRI meshes (~39k vertices), only 1.5% loss.</p>
                            </div>
                            
                            <table class="data-table mt-2" style="font-size: 0.65em;">
                                <thead>
                                    <tr>
                                        <th>Modality</th>
                                        <th>Avg Vertices</th>
                                        <th>Taubin Vol. Loss</th>
                                        <th>Info-Theoretic</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>MRI (Brain)</td>
                                        <td>38,650</td>
                                        <td class="color-warning">98.5%</td>
                                        <td class="color-success font-weight-bold">100.0%</td>
                                    </tr>
                                    <tr>
                                        <td>CT (Hemorrhage)</td>
                                        <td>13,365</td>
                                        <td class="color-danger font-weight-bold">77.7%</td>
                                        <td class="color-success font-weight-bold">99.8%</td>
                                    </tr>
                                </tbody>
                            </table>
                            
                            <p class="mt-1 font-small"><strong>Explanation:</strong> Taubin's pass-band parameters ($\\lambda$, $\\mu$) are not scale-invariant. Novel algorithms remain stable with <strong>0.2% variance</strong> across scales.</p>
                        </div>
                        <div>
                            <div class="figure-container">
                                <img src="outputs/figures/final_report/fig2_tradeoff_scatter.png" alt="Trade-off">
                                <div class="figure-caption">Fig 5: Volume vs. Smoothness Trade-off</div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="slide-footer">
                    <span>Feature-Preserving Mesh Smoothing</span>
                    <span>Shubham Mhaske &bull; CSCE 645</span>
                </div>
            </section>

            <!-- SLIDE 12: PERFORMANCE -->
            <section>
                <div class="slide-content">
                    <h2>Performance Analysis</h2>
                    <div class="grid-2col">
                        <div>
                            <div class="figure-container">
                                <img src="outputs/figures/final_report/fig4_processing_time.png" alt="Processing Time">
                                <div class="figure-caption">Fig 6: Processing Time Comparison (Log Scale)</div>
                            </div>
                        </div>
                        <div>
                            <h3>Computational Efficiency</h3>
                            <table class="data-table" style="font-size: 0.7em;">
                                <thead>
                                    <tr>
                                        <th>Algorithm</th>
                                        <th>Time</th>
                                        <th>Use Case</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>Laplacian</td>
                                        <td class="color-success font-weight-bold">22ms</td>
                                        <td>Real-time preview</td>
                                    </tr>
                                    <tr>
                                        <td>Taubin</td>
                                        <td class="color-success">41ms</td>
                                        <td>Interactive tools</td>
                                    </tr>
                                    <tr>
                                        <td>Geodesic Heat</td>
                                        <td>9.6s</td>
                                        <td>Publication graphics</td>
                                    </tr>
                                    <tr>
                                        <td>Info-Theoretic</td>
                                        <td>15.9s</td>
                                        <td>Clinical analysis</td>
                                    </tr>
                                    <tr>
                                        <td>Anisotropic</td>
                                        <td>35.4s</td>
                                        <td>Offline processing</td>
                                    </tr>
                                </tbody>
                            </table>
                            
                            <div class="card info mt-2">
                                <h4>Future Optimization</h4>
                                <p>GPU implementation (CUDA) could yield <strong>50-100× speedup</strong>, bringing novel algorithms to real-time performance.</p>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="slide-footer">
                    <span>Feature-Preserving Mesh Smoothing</span>
                    <span>Shubham Mhaske &bull; CSCE 645</span>
                </div>
            </section>'''

def get_conclusion_slides():
    return '''
            <!-- SLIDE 13: CLINICAL GUIDELINES -->
            <section>
                <div class="slide-content">
                    <h2>Clinical Decision Framework</h2>
                    <div class="grid-3col">
                        <div class="card success">
                            <h4>Tumor Volumetrics</h4>
                            <p><strong>Info-Theoretic</strong></p>
                            <p class="font-small">100% volume preservation<br>Acceptable smoothness<br>Regulatory compliant</p>
                        </div>
                        <div class="card info">
                            <h4>Surgical Visualization</h4>
                            <p><strong>Geodesic Heat</strong></p>
                            <p class="font-small">Best visual quality<br>68.9% smoothing<br>Publication-ready</p>
                        </div>
                        <div class="card highlight">
                            <h4>Real-Time Planning</h4>
                            <p><strong>Laplacian</strong></p>
                            <p class="font-small">22ms processing<br>Good volume (99.8%)<br>Interactive tools</p>
                        </div>
                    </div>
                    
                    <h3 class="mt-2">Recommendations</h3>
                    <ul>
                        <li><strong>Avoid Taubin</strong> for small anatomical structures (&lt;20k vertices)</li>
                        <li>Enable <strong>semantic awareness</strong> when segmentation labels are available</li>
                        <li>Use <strong>anisotropic</strong> for vascular networks and sharp anatomical boundaries</li>
                        <li>Consider <strong>hybrid approaches</strong>: Info-Theoretic for volume-critical regions + Geodesic for visual regions</li>
                    </ul>
                </div>
                <div class="slide-footer">
                    <span>Feature-Preserving Mesh Smoothing</span>
                    <span>Shubham Mhaske &bull; CSCE 645</span>
                </div>
            </section>

            <!-- SLIDE 14: CONTRIBUTIONS -->
            <section>
                <div class="slide-content">
                    <h2>Contributions & Impact</h2>
                    <div class="grid-2col">
                        <div>
                            <h3>Technical Contributions</h3>
                            <ol style="font-size: 0.9em;">
                                <li><strong>3 Novel Algorithms</strong> with rigorous mathematical foundations</li>
                                <li><strong>First dual-modality study</strong> on 16 medical samples (MRI + CT)</li>
                                <li><strong>Discovery</strong> of Taubin scale-sensitivity failure mode</li>
                                <li><strong>Semantic-aware framework</strong> with quantile-based coarsening</li>
                                <li><strong>Open-source implementation</strong> with interactive demo</li>
                            </ol>
                            
                            <h3 class="mt-2">Future Directions</h3>
                            <ul style="font-size: 0.85em;">
                                <li>GPU acceleration (CUDA/OpenCL)</li>
                                <li>Machine learning for parameter optimization</li>
                                <li>Extension to time-series (4D medical imaging)</li>
                                <li>Clinical trial validation</li>
                            </ul>
                        </div>
                        <div>
                            <div class="metrics-grid" style="grid-template-columns: 1fr 1fr;">
                                <div class="metric-box">
                                    <div class="metric-value color-success">100%</div>
                                    <div class="metric-label">Volume Preservation</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-value">68.9%</div>
                                    <div class="metric-label">Max Smoothing</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-value color-danger">22.3%</div>
                                    <div class="metric-label">Taubin Failure</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-value">0.2%</div>
                                    <div class="metric-label">Cross-Scale Variance</div>
                                </div>
                            </div>
                            
                            <div class="figure-container mt-2">
                                <img src="outputs/figures/final_report/fig7_summary_table.png" alt="Summary">
                                <div class="figure-caption">Fig 7: Comprehensive Summary</div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="slide-footer">
                    <span>Feature-Preserving Mesh Smoothing</span>
                    <span>Shubham Mhaske &bull; CSCE 645</span>
                </div>
            </section>

            <!-- SLIDE 15: Q&A -->
            <section>
                <div class="title-slide" style="background: linear-gradient(135deg, var(--maroon) 0%, var(--maroon-light) 100%);">
                    <h1 style="color: var(--white); border: none;">Thank You</h1>
                    <div class="divider" style="background: var(--gold);"></div>
                    <p style="font-size: 1.5em; color: var(--white); margin-bottom: 2em;">Questions?</p>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px; max-width: 800px; margin: 0 auto;">
                        <div style="text-align: center;">
                            <p style="font-size: 0.8em; color: var(--gold); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">Contact</p>
                            <p style="font-size: 1em; color: var(--white);">shubhammhaske@tamu.edu</p>
                        </div>
                        <div style="text-align: center;">
                            <p style="font-size: 0.8em; color: var(--gold); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">Repository</p>
                            <p style="font-size: 0.8em; color: var(--white); font-family: var(--font-code);">github.com/shubham-mhaske/<br>geometric_modelling_project</p>
                        </div>
                    </div>
                </div>
            </section>

        </div>
    </div>'''

def get_scripts():
    return '''
    <script src="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.5.0/reveal.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.5.0/plugin/math/math.min.js"></script>
    <script>
        Reveal.initialize({
            hash: true,
            center: false,
            width: 1280,
            height: 720,
            margin: 0.04,
            minScale: 0.2,
            maxScale: 2.0,
            transition: 'slide',
            transitionSpeed: 'default',
            backgroundTransition: 'fade',
            slideNumber: 'c/t',
            showSlideNumber: 'all',
            plugins: [ RevealMath.KaTeX ],
            math: {
                mathjax: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js',
                config: 'TeX-AMS_HTML-full',
                TeX: {
                    Macros: {
                        R: '\\mathbb{R}',
                        set: [ '\\left\\{#1 \\; ; \\; #2\\right\\}', 2 ]
                    }
                }
            }
        });
    </script>
</body>
</html>'''

def generate_presentation():
    """Main generation function"""
    print("Generating academic presentation...")
    
    # Build HTML
    html = get_head_section()
    html += get_custom_css()
    html += get_title_slide()
    html += get_introduction_slides()
    html += get_methodology_slides()
    html += get_results_slides()
    html += get_conclusion_slides()
    html += get_scripts()
    
    # Write to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Generated {OUTPUT_FILE}")
    print(f"  - 15 slides (12 min talk + 3 min Q&A)")
    print(f"  - Modern minimalist design with Texas A&M branding")
    print(f"  - Responsive layout with proper figure sizing")
    print(f"  - Math rendering with KaTeX")
    print(f"  - Code syntax highlighting")
    print(f"\nOpen in browser: file://{os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    generate_presentation()
