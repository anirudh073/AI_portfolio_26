# Weekly Summary — dec_2025 / week_2

## Weekly Lab Summary: December 18-24, 2023

**What was accomplished:**

*   Preliminary analysis of mPFC cell firing rates (N=33) from J1620210722 epoch 2 (wtrack) was conducted, comparing outbound vs. inbound directions.
*   Initial YAML file creation for animal Wilbur.
*   Work began on PFC speed tuning analysis.

**Key findings/results:**

*   Fifteen significant mPFC units were identified with differential firing rates on the central arm during outbound trials.
*   One significant unit was identified on all arms during outbound trials.
*   Three significant units were identified on the central arm during inbound trials, and seven units showed significance on all arms.
*   Tuning curves on the central arm exhibited differences between inbound and outbound directions, with some curves not being maintained.

**Problems encountered and how they were addressed:**

*   The PFC speed tuning analysis is currently hindered by a large number of variables and a need for improvement in the position tuning analysis. The current method uses 40 cm linear position bins which may be too large to detect tuning.
*   Further investigation is needed to analyze firing rates of cells *not* located on the central arm.

**Outstanding to-dos:**

*   Add data from animals Wilbur, Peanut, and Senor (in order of data availability) to the PFC speed tuning analysis. Both Senor and Peanut have obstacle days.
*   Continue the PFC speed tuning analysis on the existing (1 animal, 1 epoch) data.
*   Refine the position tuning analysis method used in the PFC speed tuning analysis, likely by reducing the bin size.