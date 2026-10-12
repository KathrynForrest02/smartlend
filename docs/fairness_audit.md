# SmartLend Fairness Audit

This document records the fairness assessment of the baseline SmartLend credit risk model developed as part of the COMP10086 Artificial Intelligence Applications project.

## Model Under Review

### Model Type

Logistic Regression

### Features Used

- RevolvingUtilizationOfUnsecuredLines
- age
- NumberOfTime30-59DaysPastDueNotWorse
- DebtRatio
- MonthlyIncome
- NumberOfOpenCreditLinesAndLoans
- NumberOfTimes90DaysLate
- NumberRealEstateLoansOrLines
- NumberOfTime60-89DaysPastDueNotWorse
- NumberOfDependents

### Training Configuration

- StandardScaler applied before training
- LogisticRegression(max_iter=1000, random_state=42)
- No class weighting
- 80/20 train-test split
- Stratified sampling used to preserve class balance

## Dataset

The model was evaluated using the **Give Me Some Credit** dataset.

Preprocessed dataset:

```text
data/processed/cs-processed.csv
```

Dataset size:

```text
146,678 rows
```

Test set size:

```text
29,336 rows
```

## Groups Examined

The audit examined model performance across three age groups.

| Age Group | Records |
|------------|---------:|
| Under 35 | 3,662 |
| 35-60 | 16,635 |
| Over 60 | 9,039 |

## Metrics Computed

| Group | Accuracy | Predicted Default Rate | False Positive Rate | False Negative Rate |
|------------|---------:|---------:|---------:|---------:|
| Under 35 | 0.897 | 0.016 | 0.009 | 0.928 |
| 35-60 | 0.931 | 0.005 | 0.002 | 0.967 |
| Over 60 | 0.973 | 0.000 | 0.000 | 0.996 |

## Fairness Metric Selected

### False Positive Rate Gap

The False Positive Rate (FPR) Gap was selected as the primary fairness measure.

This metric measures how often applicants who would actually repay their loans are incorrectly classified as default risks. In the credit domain this is important because a false positive can result in a creditworthy applicant being denied access to credit.

## Gap Value and Interpretation

### False Positive Rate Gap

```text
0.009
```

The highest false positive rate was observed for the **Under 35** age group (0.009), while the lowest false positive rate was observed for the **Over 60** age group (0.000).

This suggests that younger applicants are more likely to be incorrectly flagged as default risks. As a result, the Under 35 group appears to be the most disadvantaged according to the selected fairness criterion because it is more frequently over-flagged as a high-risk borrower.

## Likely Causes

Several factors may contribute to the observed disparity:

- The age groups exhibit different default rates within the dataset.
- Younger applicants may have shorter credit histories and different borrowing patterns.
- Some model features may indirectly correlate with age.
- The dataset is highly imbalanced, containing approximately 94% non-default cases.
- The baseline logistic regression model is relatively simple and may not capture more complex relationships in the data.

The observed disparity does not necessarily indicate discrimination. Differences in underlying feature distributions and default behaviour may explain part of the gap.

## Recommended Action

### Deploy as-is with documented monitoring

The observed False Positive Rate gap is relatively small (0.009), suggesting that the disparity between age groups is limited according to the selected metric.

However, the model demonstrates extremely high false negative rates across all groups and identifies relatively few actual default cases. Future development should focus on improving predictive performance while continuing to monitor fairness metrics during model refinement and evaluation.

## Known Limitations of this Audit

This audit did not examine:

- Income-based subgroups
- Geographic differences
- Number of dependants as a subgroup
- Intersectional groups (for example, older low-income applicants)
- Alternative classification thresholds
- Long-term deployment effects
- Feedback loops caused by model decisions
- Additional fairness criteria beyond those calculated in this audit

Future audits should investigate additional subgroup definitions and compare results across multiple fairness metrics.