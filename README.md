# Anomaly Detection in Internet Speed Data of Chicago

## Overview
This project was undertaken as part of my work with the Internet Equity Initiative at the Data Science Institute, University of Chicago. More details about the initiative are here http://internetequity.uchicago.edu/about/the-initiative/. 

1. The primary goal of the project was to develop a Python API that automates ingestion of monthly internet speed data for each internet router device registered in the research study, and generates a report of speed tier change-points (or anomalies) in the download and upload speeds obtained from NDT7 and OOKLA speed test data, where the internet speed tier of the device changes by a percentage greater than a custom defined threshold.

2. Another goal of this API is to reduce the number of false positives identified in the anomaly detection report i.e. reduce the number of changepoints that are falsely flagged as changepoints.

## Outcome
The API is scalable and so far has been used on 104 devices (3,71,125 observations) across 5 months of speed testing data. 

## Methodology

We use 2 heuristics to reduce the number of false positives (i.e. indexes that are flagged as change points but are not changepoints actually) in identifying change points:

1. **Minimum Segment Size**: We set a minimum segment size for the signal on which a change point is calculated. This is because the smaller a sub-signal is, the more will be the number of sub-signals, hence more the number of change points overall. Hence, we set a minimum segment size for each device so that a given sub-signal on which change point is being computed has atleast a minimum size. Minimum segment size is calculated as the number of tests run on a device over 7 days. So this is computed by multiplying 7 days with the minimum number of tests run on a device on a given day (for a given protocol and direction).

2. **Thresholding on difference in means of sub-signals**: After identifying change points, we find a difference in the mean of the sub-signal before changepoint and mean of the sub-signal after the changepoint. The sub-signals are created by using all observations before a changepoint index and all observations after the changepoint index upto the next changepoint index. Only if the difference in means exceeds a custom defined threshold, then the changepoint is considered valid in identifying a speed tier change.


## Output
PDF reports are generated at the level of each device. Y-axis indicates Download or Upload Speed. X-axis shows the index number of the test run on that device over May 2022 to September 2022 time period.

There are 3 combinations of search algorithm and cost function pairs, for which reports are generated:
1. PELT search and rbf cost
2. PELT search and rank cost
3. Window search and rbf cost

For each of the above combinations, there are 5 reports generated using the following combinations of heursitics to detect the anomalies in speed tier changes:
1. No heuristics implemented (baseline = True)
2. Heuristic 1: Only minimum segment size (baseline = False)
3. Heuristic 2: Only mean thresholding (baseline = False)
4. Both heuristics together - with 5% mean thresholding (baseline = False)
5. Both heuristics together - with 10% mean thresholding (baseline = False)

Inside each of the above reports, a given device id will have 4 visualization plots for anomaly detection:
1. 2 graphs for anomaly detection in Upload Speed - one for each speed test: ndt7, ookla
2. 2 graphs for anomaly detection in Download Speed - one for each speed test: ndt7, ookla

## Scripts

The script speedtest_anomaly_detection_v5.ipynb was initially coded in Jupyter notebook for rapid testing. Since this file is too large, it can be made available on request. This was converted to a Python script so that it can be imported as an API for further use by the Internet Equity Initiative's research team anomaly_detection_v1.py inside the code directory.
