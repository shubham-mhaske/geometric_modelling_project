# Speaker Script: High-Fidelity Mesh Smoothing for Medical Brain MRI Data

## Presentation Duration: 12 minutes + 3 minutes Q&A

---

## SLIDE 1: Title Slide (30 seconds)

**Script:**

"Good morning/afternoon. My name is Shubham Vikas Mhaske, and today I'll be presenting my final project for CSCE 645: Geometric Modeling.

My project focuses on High-Fidelity Mesh Smoothing for Medical Brain MRI Data, specifically addressing the challenge of semantic-aware surface reconstruction with volume preservation.

This work addresses a critical gap in medical imaging workflows where traditional mesh processing techniques fail to meet clinical accuracy requirements."

---

## SLIDE 2: Problem Statement (2 minutes)

**Script:**

"Let me start by explaining the clinical workflow, and—importantly—what our software actually does to the data.

In modern medical imaging, we follow a five-step pipeline. First, an MRI scanner captures 3D volumetric data at approximately 1 cubic millimeter resolution. Then, AI-based segmentation identifies tumor regions using standardized labeling from the BraTS challenge.

**In our project, the input we directly operate on is the 3D segmentation mask** (a voxel grid where each voxel has a label like necrotic core, edema, or enhancing tumor). From that mask:
1) We run **Marching Cubes** on the labeled region to extract a **surface mesh** (vertices + triangular faces).
2) We build the mesh neighborhood structure (adjacency, and for some methods cotangent weights).
3) We apply a smoothing algorithm for $k$ iterations, which updates the vertex positions.
4) We compute metrics **before vs after** (volume change, smoothness, triangle quality, and runtime).
5) We visualize the result in the Streamlit app and optionally export meshes/figures.

However, this is where the problem arises. Marching Cubes produces meshes with significant artifacts:
- Jagged staircase edges at voxel boundaries
- Poor-quality triangles that are elongated or degenerate
- Artificial high-curvature regions at voxel corners
- Visual noise that obscures fine anatomical details

The natural solution would be to smooth these meshes, but here's the critical issue: traditional Laplacian smoothing causes 1 to 4 percent volume shrinkage. This is completely unacceptable in clinical settings.

The FDA and RECIST 1.1 guidelines require volume measurement accuracy of less than 1 percent for tumor monitoring. When oncologists track tumor response to treatment, even small volume errors can lead to incorrect treatment decisions.

Our project goal is ambitious: achieve less than 0.1 percent volume change—ten times better than FDA requirements—while maintaining high visual quality."

**Transition:** "Let me now describe the dataset we used to validate our approach."

---

## SLIDE 3: Dataset - BraTS 2023 (1.5 minutes)

**Script:**

"Very briefly on data, because I want to spend time on results and the demo.

All headline numbers in this talk are on **BraTS tumor segmentations**, using **n=20 cases**.

BraTS provides voxel-wise labels for tumor sub-regions (core, edema, enhancing). Our pipeline operates directly on that **segmentation mask**, converts it to a mesh with Marching Cubes, then smooths and evaluates.

Across these 20 cases, mesh sizes range from roughly **6K to 119K vertices**, so we also test runtime scaling.

That’s enough dataset context—next I’ll describe the objectives and metrics." 

**Transition:** "With this dataset established, let me explain our research objectives and evaluation framework."

---

## SLIDE 4: Research Objectives & Metrics (1.5 minutes)

**Script:**

"Our project had four main objectives:

First, Algorithm Development: We implemented five distinct smoothing algorithms—Laplacian, Taubin lambda-mu, Geodesic Heat, Anisotropic Tensor, and Information-Theoretic smoothing.

Second, Comprehensive Evaluation: We tested these algorithms on 20 BraTS meshes ranging from 5,990 to 118,970 vertices, measuring 4 quality metrics for each algorithm-sample combination.

Third, Quantitative Analysis: We measured volume preservation, smoothing quality, aspect ratio improvement, and processing time across all algorithms.

Fourth, Application Guidelines: We created a decision framework to help practitioners select the right algorithm for their specific use case.

For evaluation, we focused on four core metrics (the same ones shown on the slides):
- Volume change (targeting very small drift, ideally < 0.1%)
- Smoothness (curvature-variance reduction)
- Aspect ratio improvement (triangle quality)
- Processing time (latency)

In total, this gave us 400 individual measurements: 20 meshes × 5 algorithms × 4 metrics.

In the codebase, I also compute optional diagnostics like curvature correlation and Hausdorff distance, but they are not part of the headline n=20 summary in this talk."

**Transition:** "Let me now walk through each algorithm, starting with our baseline."

---

## SLIDE 5: Laplacian Smoothing Baseline (2 minutes)

**Script:**

"Laplacian smoothing is the most widely used mesh smoothing technique, so it serves as our baseline.

The mathematical principle is straightforward. The Laplacian operator computes the difference between a vertex and the centroid of its neighbors. The update rule then moves each vertex toward that centroid by a factor lambda, typically 0.5.

In our evaluation pipeline, this is applied after Marching Cubes extraction. For each BraTS mesh, we keep the triangle connectivity (faces) fixed, and only update vertex positions. Concretely, we build the neighbor structure once (adjacency), then run the same iteration budget $k$ per mesh, and compute our before/after metrics: volume change, smoothness change (curvature-variance reduction), triangle aspect ratio improvement, and runtime.

Looking at our vectorized implementation: we build a row-normalized adjacency matrix W, construct the smoothing operator K as a weighted combination of identity and adjacency, then iteratively apply K to the vertex positions.

Now let's look at the experimental results from our 20 BraTS samples.

For volume change, we measured a mean of negative 0.92 percent plus or minus 0.79 percent across all samples. Note that volume loss varies significantly with mesh size — larger meshes show less percentage loss.

For smoothness, we achieved 97.4 percent plus or minus 0.8 percent curvature reduction—the best among all algorithms.

While the smoothing quality is excellent, the volume loss—though varying by mesh—compounds with more iterations. For volume-sensitive applications, we need better alternatives."

**Transition:** "To address this shrinkage, Gabriel Taubin proposed a signal processing approach."

---

## SLIDE 6: Taubin Lambda-Mu Smoothing (2 minutes)

**Script:**

"Taubin's key insight was to view mesh smoothing through the lens of signal processing. He recognized that Laplacian smoothing is essentially a low-pass filter that attenuates all frequencies, including low frequencies that represent the overall shape.

His solution is elegant: alternate between shrinking and expanding steps. The first step applies positive lambda, causing shrinkage. The second step applies negative mu, causing expansion. When the parameters satisfy the constraint 0 less than lambda less than negative mu, these effects largely cancel out.

We use the standard parameters: lambda equals 0.5, mu equals negative 0.53.

In our evaluation loop, we apply Taubin to the exact same input meshes as Laplacian (same Marching Cubes surfaces, faces fixed). The only difference is that each iteration is a two-step update: one pass with $\lambda$ (shrink), followed immediately by a pass with $\mu$ (expand). After the final iteration, we compute the same before/after metrics and record runtime.

The results are dramatically different. Our mean volume change is now positive 0.056 percent plus or minus 0.047 percent—16 times better than Laplacian.

Looking at individual samples, all 20 BraTS samples showed consistent volume preservation, with the variation coming from mesh size differences.

The trade-off is smoothing quality. We achieve 89.0 percent plus or minus 1.9 percent smoothness compared to Laplacian's 97.4 percent—about 8 percent less aggressive smoothing.

But for volume-sensitive applications, this is the right trade-off. We preserve volume while still achieving significant noise reduction. This is our recommended algorithm for most applications requiring accurate measurements."

**Transition:** "We also developed three novel algorithms. Let me describe our Geodesic Heat approach."

---

## SLIDE 7: Geodesic Heat Diffusion (1.5 minutes)

**Script:**

"Our third algorithm is Geodesic Heat Diffusion smoothing.

The key concept is using geodesic distances—distances measured along the surface—instead of Euclidean distances for neighbor weighting.

In practice for our pipeline, this means we still start with the same Marching Cubes surface mesh, but we replace uniform neighbor averaging with geometry-aware weights. We compute an intrinsic operator (cotangent Laplacian) from the mesh angles, use it to approximate geodesic-aware influence, and then run $k$ smoothing updates on vertex positions (faces fixed), followed by the same metric evaluation.

We implement this using the heat kernel. The weight between two vertices decays exponentially with the square of their geodesic distance, scaled by a time parameter t.

We approximate geodesic distances using the cotangent Laplacian, which is more geometrically accurate than the uniform Laplacian. The formula shown here computes mean curvature using the cotangent of angles opposite each edge.

Our method characteristics include:
- Using the heat equation to approximate geodesic distances efficiently
- Feature-aware stopping based on curvature gradients
- Respecting the intrinsic geometry of the surface manifold

Results show a mean volume change of negative 0.82 percent plus or minus 0.71 percent—comparable to Laplacian. However, we achieve 97.0 percent plus or minus 0.9 percent smoothness, which is near-best performance.

The algorithm achieves good aspect ratio improvement at approximately positive 14 percent.

This makes Geodesic Heat ideal for visualization and 3D printing where visual quality is the priority and small volume changes are acceptable."

**Transition:** "Our next two novel algorithms take different approaches."

---

## SLIDE 8: Information-Theoretic Smoothing (1 minute)

**Script:**

"Next is Information-Theoretic smoothing. It uses Shannon entropy to distinguish noise from features. The idea is simple but powerful: random noise has high entropy in the curvature distribution, while coherent features have low entropy.

For each vertex, we compute the entropy of curvature values in its neighborhood. We then use a sigmoid function to convert this entropy to a smoothing weight. High entropy regions get smoothed aggressively; low entropy regions—where features exist—are preserved.

How we apply it during evaluation: for each mesh, we estimate a local curvature signal, compute neighborhood entropy, convert that into per-vertex (or per-edge) weights, and then apply a weighted smoothing update. This makes the algorithm adaptive: two vertices with the same degree can move very differently if one lies in a “noisy” curvature neighborhood and the other lies on a stable feature region. Then we compute the same before/after metrics and runtime.

Results: +0.042% ± 0.035% volume change with consistent behavior across all 20 samples. This makes it a strong option for volume-sensitive work with meaningful smoothing at 84.4% ± 2.2%. Processing time averages 44ms."

**Transition:** "Next is our Anisotropic Tensor method, which targets maximum volume preservation."

---

## SLIDE 9: Anisotropic Tensor Smoothing (1 minute)

**Script:**

"Anisotropic Tensor smoothing takes a geometric approach. I compute principal curvature directions at each vertex, then build a diffusion tensor so smoothing happens more along stable directions and less across sharp features.

In our pipeline context, this means the smoothing step is not isotropic averaging. Instead, we treat smoothing like directional diffusion on the surface: we estimate local directions, construct a tensor that damps motion across boundaries, and then update vertex positions for $k$ steps while keeping faces fixed. We evaluate it with the same metric set, which is why it stands out for near-zero volume change even though the smoothness improvement is intentionally more conservative.

Results: −0.022% ± 0.019% volume change—virtually zero. The trade-off is moderate smoothing at 59.5% ± 1.8%, and average runtime around 126ms.

So, Info-Theoretic is the best balance among our methods, while Anisotropic is the choice when volume preservation dominates."

**Transition:** "Now I’ll summarize all five algorithms side-by-side across the n=20 evaluation."

---

## SLIDE 10: Comprehensive Results (2 minutes)

**Script:**

"This table summarizes all evaluations across our 5 algorithms and 20 BraTS samples ranging from 5K to 119K vertices.

Starting with volume preservation—the most critical metric:
- Laplacian: negative 0.92 percent plus or minus 0.79 percent
- Geodesic Heat: negative 0.82 percent plus or minus 0.71 percent
- Taubin: positive 0.056 percent plus or minus 0.047 percent, excellent
- Anisotropic: negative 0.022 percent plus or minus 0.019 percent, best overall
- Info-Theoretic: positive 0.042 percent plus or minus 0.035 percent, very consistent

For smoothing quality:
- Laplacian leads at 97.4 percent plus or minus 0.8 percent
- Geodesic Heat second at 97.0 percent plus or minus 0.9 percent
- Taubin at 89.0 percent plus or minus 1.9 percent
- Info-Theoretic at 84.4 percent plus or minus 2.2 percent
- Anisotropic last at 59.5 percent plus or minus 1.8 percent

Processing time averages across all mesh sizes:
- Laplacian is fastest at 17ms average
- Taubin at 25ms average
- Geodesic Heat at 27ms average
- Info-Theoretic at 44ms average
- Anisotropic slowest at 126ms average

The 'Suitable' column summarizes our recommendation:
- Laplacian: No, due to volume loss that compounds
- Geodesic Heat: Caution, small but consistent volume loss
- Taubin, Info-Theoretic, Anisotropic: Yes, safe for volume-sensitive applications"

**Transition:** "Let me show you the visual comparisons."

---

## SLIDE 11: Visual Results (1 minute)

**Script:**

"These visual comparisons show what the metrics mean in practice.

On the left, you can see the original mesh overlaid with Taubin smoothing. The surface noise is reduced, but the overall scale remains consistent—this is why Taubin is a safe default for volumetrics.

On the right, the ‘all algorithms’ comparison shows the qualitative trade-offs: Laplacian and Geodesic Heat look very smooth but shrink; Info-Theoretic and Anisotropic preserve structure better; Taubin stays in the middle with strong smoothing and reliable volume behavior.

So visually, the recommendation matches the numbers: Taubin for general clinical use; Info-Theoretic when feature preservation is a priority."

**Transition:** "The scatter plot and radar chart provide deeper analysis."

---

## SLIDE 12: Trade-off Analysis (1 minute)

**Script:**

"The scatter plot on the left shows the fundamental trade-off between volume preservation and smoothing quality.

The ideal operating region is the top-right: high smoothing with low volume change.

Laplacian sits in the bottom-right—high smoothing but unacceptable volume loss. It's dominated by Taubin, which achieves similar smoothing with much better volume.

Anisotropic occupies the top-left extreme—perfect volume but minimal smoothing.

Info-Theoretic lies on the Pareto front—no algorithm dominates it in both metrics simultaneously.

The radar chart shows all metrics at once. Notice how different algorithms have different strengths:
- Laplacian excels in smoothing and speed
- Anisotropic excels in volume preservation
- Info-Theoretic and Taubin provide balance across all dimensions

There is no single best algorithm—the choice depends on clinical priorities."

**Transition:** "Finally, I’ll show how semantic labels help preserve clinically meaningful boundaries during smoothing."

---

## SLIDE 13: Semantic-Aware Smoothing (1 minute)

**Script:**

"The BraTS dataset gives us voxel-wise labels for tumor sub-regions. We can use those labels to reduce smoothing across region boundaries.

Implementation idea: modify adjacency weights. If two connected vertices share the same label, keep the weight. If they differ, multiply by a boundary factor α, and we used α = 0.3.

This preserves transitions like enhancing tumor vs edema while still smoothing inside each region.

In our semantic-aware experiment, boundary preservation improved by 84% and volume drift reduced by 85%, making boundary-sensitive smoothing much more clinically plausible."

**Transition:** "Based on these results, we developed application guidelines."

---

## SLIDE 14: Practical Application Guidelines (1.5 minutes)

**Script:**

"We conclude with practical guidelines for application.

For Tumor Volumetrics—the most common use case—we recommend Taubin lambda-mu smoothing.
- Validated at positive 0.056 percent plus or minus 0.047 percent volume change on n=20 meshes
- Best balance of volume preservation and smoothing
- 25 millisecond average processing
- Use for: treatment response monitoring, longitudinal studies, RECIST measurements

For Surgical Planning where visual quality matters more, we recommend Geodesic Heat.
- 97.0 percent smoothness for best visual quality
- Respects intrinsic surface geometry
- 27 millisecond processing
- Use for: 3D printing, VR visualization, surgical simulation

For Real-Time Preview during parameter exploration, Laplacian is acceptable.
- 97.4 percent smoothness
- Fastest at 17 milliseconds average
- Simple implementation
- Use ONLY for preview—never for final measurements

Critical warnings:
- Never use Laplacian for volume measurement—up to 3.64 percent loss on smaller meshes
- Geodesic Heat has variable volume change up to 3.25 percent—not suitable for longitudinal studies

Default recommendation: Taubin with lambda 0.5, mu negative 0.53, 10 iterations."

**Transition:** "Let me summarize my contributions and the key takeaways."

---

## SLIDE 15: Contributions & Conclusions (1.5 minutes)

**Script:**

"To summarize my contributions in this project:

First, I implemented a suite of 5 smoothing algorithms with efficient vectorized NumPy/SciPy.

Second, I implemented three novel methods—Geodesic Heat, Information-Theoretic, and Anisotropic—grounded in differential geometry and information theory.

Third, I implemented a semantic-aware boundary-preserving extension using BraTS labels.

Fourth, I ran the comprehensive evaluation: 20 BraTS meshes × 5 algorithms × 4 metrics = 400 measurements, and distilled the results into practical guidelines.

Fifth, I built an interactive Streamlit application for real-time 3D visualization and export, and generated the figure set used in the report and slides.

Key findings from n=20 samples:
- Taubin achieves excellent balance at positive 0.056 percent plus or minus 0.047 percent volume with 89.0 percent smoothing
- Laplacian provides highest smoothing at 97.4 percent but poor volume preservation at negative 0.92 percent
- Geodesic Heat offers excellent visual quality at 97.0 percent smoothness
- Information-Theoretic provides good balance with +0.042 percent volume and 84.4 percent smoothing
- Anisotropic achieves best volume preservation at negative 0.022 percent plus or minus 0.019 percent
- All algorithms perform well across mesh sizes from 5K to 119K vertices

Future work includes GPU acceleration for 10 to 50 times speedup, deep learning-based adaptive smoothing, and extension to CT and PET imaging.

The key takeaway: For any application requiring volume accuracy, use Taubin lambda-mu smoothing with the parameters we've validated on 20 BraTS brain tumor meshes."

**Transition:** "Now I’ll use the last couple of minutes to show the Streamlit demo live—what to click, what to look for, and how it ties directly to the metrics."

---

## LIVE DEMO (End of presentation, ~2.5–3 minutes)

**Purpose (say this once):**
"This demo is not just eye-candy—I'm going to show (1) the visual improvement, and (2) the quantitative evidence modes that explain *where* and *how much* the surface changed."

### Pre-demo setup (do silently before the talk starts)
- Open the app in the browser and load any patient once (to warm caches).
- Keep browser zoom at 100%.

### Demo runbook (what to click + what to say)

**1) Start in the clean geometry view (15–20s)**
- **Click:** Sidebar → *Visualization* = **Geometry (single color)**
- **Click:** *Geometry style* = **Cool** (use Neutral if the room/projector is dim)
- **Say:** "Left is the original Marching Cubes mesh; right is the smoothed mesh. Faces are identical—only vertex positions move."

**2) Show the recommended 'safe default' (30–40s)**
- **Click:** *Smoothing* = **Taubin**
- **Click:** *Iterations* = **10**
- **Say:** "Taubin removes the staircase artifacts but keeps overall size very stable, which is why it’s the default recommendation for volumetrics."

**3) Contrast with Laplacian (30–40s)**
- **Click:** *Smoothing* = **Laplacian** (same iterations)
- **Say:** "Laplacian often looks very smooth, but it tends to shrink volume—so it’s great for fast preview, not final measurement."

**4) Evidence mode: displacement magnitude (35–45s)**
- **Click:** *Visualization* = **Displacement (magnitude)**
- **Say:** "The original plot is a baseline. The smoothed plot shows displacement in millimeters—so we can quantify how much the surface moved and where."

**5) Evidence mode: curvature change |Δ| (35–45s)**
- **Click:** *Visualization* = **Curvature change (|Δ|)**
- **Say:** "This highlights where local surface roughness changed the most. It explains *where* smoothing occurred rather than just 'it looks smoother'."

**Optional (only if time): tumor labels (15–20s)**
- **Click:** *Visualization* = **Tumor labels**
- **Say:** "Because the input is a labeled mask, we can also visualize sub-regions on the mesh—useful for boundary-aware smoothing extensions."

**Demo close (10s)**
"So the live demo matches the table: Taubin is the balanced default for safe volumetrics; Laplacian is preview-only; and the evidence views make the differences measurable."

---

## SLIDE 16: Questions (open-ended)

**Script:**

"Thank you for your attention. I'm happy to answer any questions about the algorithms, implementation details, or clinical applications.

You can reach me at shubhammhaske@tamu.edu, and the code is available on GitHub at the repository shown.

Are there any questions?"

---

## Potential Q&A Topics

### Q: Why not use GPU acceleration?

**A:** "Our CPU implementation is already fast for interactive use in most cases—Laplacian, Taubin, Geodesic Heat, and Info-Theoretic are typically under 100ms on our tested meshes. The anisotropic method is the slowest (about 126ms on average, up to ~329ms on the largest mesh), so GPU acceleration is planned to make even the most expensive method comfortably real-time and to support large-scale batch processing."

### Q: How does semantic smoothing work with overlapping labels?

**A:** "In BraTS, labels are mutually exclusive per voxel. When mapping to vertices, we use nearest-neighbor interpolation. If a vertex falls between regions, the cross-label weight applies, providing a smooth transition rather than hard boundary."

### Q: Can this work with other imaging modalities?

**A:** "Yes, the algorithms are modality-agnostic. Any segmentation mask that can be converted to a surface mesh via Marching Cubes can use these techniques. CT for bone or organ segmentation and PET for metabolic regions are natural extensions."

### Q: What parameters did you tune for each algorithm?

**A:** "Taubin uses standard parameters: lambda=0.5, mu=-0.53, 10 iterations. Geodesic Heat uses time scale proportional to mean edge length squared. Info-Theoretic threshold was set to median entropy. Anisotropic diffusion uses curvature-adaptive coefficients. All were validated across our n=20 set."

### Q: Why is Anisotropic smoothing so weak?

**A:** "Anisotropic smoothing is designed to preserve features by diffusing only along principal directions. Medical meshes from Marching Cubes have noise in all directions, so anisotropic diffusion is inherently limited. It excels when preserving specific features is more important than overall smoothing."

---

## Timing Summary

| Slide | Topic | Duration |
|-------|-------|----------|
| 1 | Title | 0:20 |
| 2 | Problem Statement | 1:20 |
| 3 | Dataset (BraTS n=20) | 0:25 |
| 4 | Objectives & Metrics | 0:55 |
| 5 | Laplacian Baseline | 0:55 |
| 6 | Taubin | 1:00 |
| 7 | Geodesic Heat | 0:50 |
| 8 | Info-Theoretic | 0:55 |
| 9 | Anisotropic | 0:45 |
| 10 | Comprehensive Results | 1:10 |
| 11 | Visual Results | 0:45 |
| 12 | Trade-off Analysis | 0:45 |
| 13 | Semantic-Aware Smoothing | 0:45 |
| 14 | Application Guidelines | 1:10 |
| 15 | Contributions & Conclusions | 0:55 |
| Demo | Live demo (Streamlit) | 2:40 |
| 16 | Q&A | 3:00 |

**Total talk: ~12 minutes + ~3 minutes Q&A**

If time is tight, compress Slides 11–13 into quick one-sentence takeaways and keep Slides 6, 10, 14, and 15 as the “must-hit” core.
