# Weekly Summary — dec_2025 / week_4

## Weekly Summary: Tuning Curve Stability Analysis (Dec 23-28, 2023)

This week focused on investigating the stability of speed tuning curves in neuronal data from the Wilbur dataset (20210512). The overarching goal is to understand if these tuning curves remain consistent within and across recording epochs and to identify potential confounding factors.

**Accomplishments:**

*   **Data Acquisition & Initial Exploration:**  Initial examination of data from Wilbur (20210512), containing approximately 250 cells (200 from left mPFC, 50 from right), revealed a mix of 'bell-shaped', 'increasing' and 'decreasing' tuning curve types.
*   **Position Estimation Refinement:**  A critical issue arose with position estimation due to excessive NaN values. This was resolved by increasing the `max_gap` parameter in the `SmoothInterp` population from 5cm to 15cm and re-running the position estimation cycle.
*   **Within-Epoch Tuning Curve Stability Analysis:** A framework was developed to assess tuning curve stability within a single epoch by dividing epochs into first and second halves.  The following metrics were computed and used in statistical comparisons:
    *   Curve Correlation
    *   Modulation Depth (normalizes RMSE)
    *   Normalized RMSE
    *   Shift in Peak Bin
    *   Mean Firing Rate
    *   Mean Firing Rate Difference
*   **Initial Results on Within-Epoch Stability:**  Preliminary analysis utilizing random epoch splits (n=50) indicated a "stationary" label for 201 out of 250 cells, with 58 cells labeled as "drifting" and 4 with "insufficient" data.  Shape drift was observed in 10 cells, and rate drift in 53. Null baseline metrics (median across units) were recorded as: corr = 0.4933, nrmse = 0.3206.
* **Progress/Speed Relationship:** Initial exploration showed a correlation between speed and progress, prompting the need for further investigation to rule out progress as a confound.

**Key Findings/Results:**

*   A diverse range of tuning curve shapes were observed in the Wilbur dataset.
*   An initial assessment of within-epoch tuning curve stability revealed a significant proportion of cells (58/250) displaying drift.
*   Correlation and peak shift were used as shape metrics, while mean rate difference and nRMSE were used as rate metrics.
*   The correlation (corr) and normalized RMSE (nrmse) were recorded as 0.4933 and 0.3206, respectively, for the null baseline.



**Problems Encountered & Solutions:**

*   **NaN Values in Position Estimation:**  The initial `max_gap` setting of 5cm in `SmoothInterp` resulted in numerous NaN values.  This was remedied by increasing the parameter to 15cm.
*   **Statistical Analysis Design:** A framework for assessing stability was developed including defining tuning bins, calculating metrics, and establishing a p-value threshold (alpha=0.05) for classifying units.

**Outstanding To-Dos:**

*   Complete testing of tuning curve stability across epochs (first vs. last).
*   Investigate the relationship between speed and progress to determine if progress is confounding tuning curve measurements. Implement stratification and progress weight equalization to assess confounding.
*   Generate figures for a poster presentation including:
    *   Spikes vs. position (rest vs. run)
    *   Visualization of tuning curve types.
    *   Illustrations of metric calculation methods.
    *   Fraction of cells exhibiting changes in tuning curves.
    *   Example tuning curves illustrating changes over time.
    *   Plots of speed vs. progress with correlation coefficient.
*   Finish initial poster introduction.
*   Plan a strategy for combining single-unit analysis results across animals.
*   Get speed, location and progress tuning results from Wilbur 20210512.