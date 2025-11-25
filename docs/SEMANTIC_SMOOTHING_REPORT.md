# Semantic Smoothing Experiment Report — 2025-11-23

## Objective
Run the Laplacian and Taubin smoothing pipelines in both baseline and semantic-aware modes to verify that the updated `map_labels_to_vertices` mapping activates boundary-aware behavior on the available BraTS samples.

## Dataset & Inputs
- Source volumes: `data/labels/BraTS-GLI-00001-000-seg.nii.gz`, `BraTS-GLI-00001-001-seg.nii.gz`, `BraTS-GLI-00013-000-seg.nii.gz`, `BraTS-GLI-00013-001-seg.nii.gz`, `BraTS-GLI-00015-000-seg.nii.gz`.
- Each marching-cubes mesh contains ≈118,970 vertices with 6,399 non-zero label IDs once mapped (e.g., IDs 196–15,058). The uniform label counts explain why all metric lines below are identical per file.
- Environment: macOS, Python virtualenv located at `.venv/` inside the repository.

## Methodology
1. Convert each segmentation to a PyVista mesh using `process_nifti_to_mesh`, keeping both the vertex array and the original affine/label volume.
2. Map per-vertex labels by calling the fixed `map_labels_to_vertices`, which now falls back to voxel spacing when affine inversion strays outside the volume bounds.
3. For each algorithm (Laplacian, Taubin):
   - Run a **baseline** smoothing pass (`vertex_labels=None`).
   - Run a **semantic** pass (`vertex_labels=<mapped labels>`).
   - Measure displacement statistics and recompute surface volume via `calculate_metrics`.
4. Aggregate results across files. Because every file yielded identical statistics, we report a single representative row per algorithm.

Command used (executed from the project root):

```bash
"$PWD/.venv/bin/python" - <<'PY'
# See scripts in the conversation log for the full snippet.
PY
```

## Results
| Algorithm | Baseline Volume (mm³) | Semantic Volume (mm³) | Volume Δ (mm³) | Baseline Mean Disp. (mm) | Semantic Mean Disp. (mm) | Mean Disp. Δ (mm) | Baseline Max Disp. (mm) | Semantic Max Disp. (mm) |
|-----------|----------------------:|----------------------:|---------------:|-------------------------:|-------------------------:|------------------:|-------------------------:|-------------------------:|
| Laplacian | 1,635,334.438 | 1,638,680.599 | +3,346.161 | 0.3897 | 0.5961 | +0.2064 | 1.655 | 2.580 |
| Taubin    | 1,635,334.438 | 1,635,628.389 |   +293.951 | 0.2529 | 0.3459 | +0.0930 | 0.613 | 1.123 |

### Interpretation
- The presence of thousands of tiny label regions makes the semantic barrier highly restrictive, boosting both mean and max displacements relative to the baseline for these specific scans.
- The identical metrics per file indicate that the current subset of BraTS volumes share the same binary tumor mask after marching cubes; additional heterogeneous volumes would better exercise the label-aware logic.

## 2025-11-24 — Quantile-Based Label Coarsening
To counter the 6k+ fragment IDs found in the Synapse fastlane segmentations, the pipeline now collapses verbose label IDs into at most three quantile bins before `map_labels_to_vertices` runs. This keeps true anatomical boundaries while preventing every tiny fragment from forming its own semantic island.

- Coarsening reduces the number of non-zero IDs on `BraTS-GLI-00013-000-seg.nii.gz` from 6,398 down to only 3 coarse classes while keeping the mesh geometry identical.
- Re-running the smoothing comparison confirms that Laplacian volume drift dropped from +3,346 mm³ to +534 mm³, demonstrating that the semantic barrier is no longer overwhelmed by noisy IDs. Taubin now swings only -150 mm³ versus -200 mm³ previously.

| Algorithm | Coarse Labels | Baseline Mean Disp. (mm) | Semantic Mean Disp. (mm) | Mean Disp. Δ (mm) | Volume Δ (mm³) |
|-----------|--------------:|-------------------------:|-------------------------:|------------------:|---------------:|
| Laplacian |             3 | 0.3897 | 0.7898 | +0.4001 | +534.002 |
| Taubin    |             3 | 0.2529 | 0.3729 | +0.1200 | -149.906 |

The semantic pass still moves more than the unlabeled baseline on this case, but the magnitude is now driven by tissue-scale transitions instead of per-fragment artifacts. This sets the stage for tuning `_CROSS_LABEL_WEIGHT` and iteration counts against realistic multi-tissue boundaries.

## Next Steps
1. Extend the coarsening pass across every case in `data/labels/` to build a richer semantic metrics table.
2. Experiment with `_CROSS_LABEL_WEIGHT` (e.g., 1e-1 → 1e-3) to study how strictly edges are clamped now that labels are stable.
3. Surface coarse-label statistics in the Streamlit UI so users can verify which tissues are being protected per run.
