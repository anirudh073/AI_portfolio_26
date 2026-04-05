# Weekly Summary — dec_2025 / week_1

## Weekly Summary: J16 Speed Tuning Analysis (Dec 23-27, 2025)

This week focused on analyzing speed tuning properties of single neurons in animal J16, utilizing data from two recording epochs.

**Accomplishments:**

*   **Speed Tuning Curves Generated:** Speed tuning curves were generated for 33 mPFC neurons and 133 OFC neurons.
*   **Firing Property Retrieval Functionality:** A function was implemented in `spikes_analysis.py` to retrieve firing properties for neurons being examined for speed tuning (demonstrated and documented in `spikes_analysis.ipynb`).

**Key Findings/Results:**

*   Neurons exhibited a range of responses to speed changes, including unresponsiveness, increases, decreases, and bell-shaped firing rate responses.

**Problems Encountered and Solutions:**

*   **Noisy Linearization Data:** Initial linearization of dataset J1620210722 produced noisy data when `use_hmm = False`. This was addressed by re-running the linearization with `use_hmm = True`.
*   **Confidence Interval Calculation:**  The method for calculating confidence intervals for firing rates, currently assuming a Poisson distribution, is being re-evaluated and bootstrapping will be considered as an alternative.

**Outstanding To-Dos:**

*   **Speed Tuning Code Completion:** Finalize the speed tuning code located in `spike_analysis.py`.
*   **Further Investigation of Observed Phenomena:**
    *   Investigate the relationship between neuronal firing and animal location and "journey" progress.
    *   Compute cell centroids using limbs.
    *   Plot the journey speed distribution.
    *   Order cells by speed tuning.
    *   Investigate inflection points observed in some cell's tuning curves.
    *   Examine movements at the preferred speed of bell-curve cells.
    *   Examine if speed modulates autocorrelograms.
    *   Retrieve waveforms, spike rasters, and cell type information (interneuron/pyramidal cell) for each cell.