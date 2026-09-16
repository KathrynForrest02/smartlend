# SmartLend Business Brief

SmartLend is a fictional UK fintech company that offers personal loans through a mostly automated application process. The company wants to build a loan default prediction service to support lending decisions.

## Business Problem

SmartLend wants to estimate whether an applicant is likely to become seriously delinquent within the next two years. The prediction should support risk review, pricing, and responsible lending decisions.

## Dataset

The teaching project uses the Give Me Some Credit dataset. The target variable is:

- `SeriousDlqin2yrs`: `1` if the borrower experienced 90+ days past due delinquency within two years, otherwise `0`.

The dataset contains numeric financial and demographic features including credit utilisation, age, debt ratio, monthly income, number of open credit lines, late payment counts, real estate loans, and dependants.

## Important Constraints

- The dataset is highly imbalanced: most applicants do not default.
- Missing values occur in `MonthlyIncome` and `NumberOfDependents`.
- Credit decisions are high-stakes and can materially affect applicants.
- Fairness, explainability, auditability, and reproducibility matter alongside predictive performance.

## Stakeholders

- Applicants seeking fair access to credit
- SmartLend risk and underwriting teams
- Compliance and governance staff
- Regulators and external auditors
- Engineering teams responsible for maintaining the system

## Initial Success Criteria

A useful system should produce reproducible predictions, be evaluated using metrics suitable for imbalanced classification, avoid obvious data leakage, and include documentation of known risks and limitations.

