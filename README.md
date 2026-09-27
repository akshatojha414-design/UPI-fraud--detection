\# UPI Fraud \& Anomaly Detection System



A multi-layer fraud detection pipeline combining statistical, unsupervised ML, 

and time-series anomaly detection, served via a Flask API.



\## Problem

Confirmed fraud labels are rare in real-world payment data, so this project 

builds an \*\*unsupervised, multi-signal\*\* detection system that doesn't depend 

on pre-labeled fraud examples — closer to how real fraud systems must operate.



\## Approach

1\. \*\*Synthetic data generation\*\* — 5000+ UPI transactions (Faker), \~3% injected 

&#x20;  fraud patterns (abnormal amounts, odd hours, rapid-fire bursts)

2\. \*\*Feature engineering\*\* — sender transaction history, amount deviation from 

&#x20;  personal average, hour/day/night flags

3\. \*\*Three independent detectors:\*\*

&#x20;  - IQR-based outlier detection on amount (caught \~40% of fraud alone)

&#x20;  - Isolation Forest (unsupervised ML across all features)

&#x20;  - Rolling z-score on log-transformed transaction gaps (catches rapid-fire 

&#x20;    fraud amount-based methods miss)

4\. \*\*Combined risk scoring\*\* — flagged if any detector fires; 0-3 score shows 

&#x20;  how many agree

5\. \*\*Deployment\*\* — Flask REST API (`/predict`) for real-time verdicts



\## Results

\- Combined system: \*\*72.6% recall, 22.5% precision\*\*

\- Tradeoff is intentional — in fraud detection, missing real fraud is costlier 

&#x20; than a false alarm needing manual review



\## Tech stack

Python · Pandas · NumPy · Faker · Scikit-learn (Isolation Forest) · Flask · Power BI

















