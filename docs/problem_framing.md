# SmartLend Problem Framing

## Prediction Task

The model is being asked to predict whether a loan applicant will experience serious delinquency within the next two years. This is a binary classification problem where the target variable is `SeriousDlqin2yrs` (1 = serious delinquency, 0 = no serious delinquency).

## Likely Users of the Prediction

Direct users include SmartLend's automated loan approval system and loan officers reviewing applications. Indirect stakeholders include loan applicants, SmartLend management, regulators, and compliance teams who are affected by the fairness and accuracy of lending decisions.

## Potential Failure Modes

A false negative occurs when the model predicts that an applicant is low risk but they later default on their loan. This could lead to financial losses for SmartLend.

A false positive occurs when the model predicts that an applicant will default but they would have repaid the loan successfully. This could result in a creditworthy applicant being denied access to credit and may raise fairness concerns if certain groups are affected disproportionately.

## Definition of Success

A successful model should identify high-risk applicants while minimising incorrect rejections of creditworthy applicants. Because the dataset is highly imbalanced (approximately 93% non-default and 7% default), metrics such as recall, precision and F1-score are more useful than accuracy alone. The model should be reliable, fair, explainable, and compliant with regulatory requirements before deployment.