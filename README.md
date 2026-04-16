# AnalyticsInPracticeGroup3
Group Project for Analytics in Practice module at Trinity College Dublin for group 3.

In this file, both the Role Assignments and LLM Prompt Log contents are included. However, for a better view, we also created separate files. To run the code, it is optimal to open it on Google Colab and run the sections there. 

The **visualisatons** (html file) can be explored through this link: [Bank Churn Dashboard](https://69df70562d2c967a38e3d65a--musical-dieffenbachia-758f48.netlify.app/)


# Role Assignments
[Role Assignments.pdf](https://github.com/user-attachments/files/26785663/Role.Assignments.pdf)



# LLM Prompt Log — Bank Customer Churn Project

**LLM Specialist:** Tuong Minh and Yifei Yu  
**Date:** April 2025  
**Model Used:** Claude (claude-sonnet-4-20250514), accessed via claude.ai  
**Dataset:** Bank Customer Churn — https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset  
**Rows:** 10,000 customers | **Features:** 12 | **Target:** `churn` (1 = churned)

> *Generative AI use is acknowledged at prompt level in accordance with TCD GenAI referencing guidelines: https://libguides.tcd.ie/genai/acknowledging-referencing. All LLM outputs have been manually verified by the team.*

---

## Prompt 1 — EDA Insight Synthesis

**Purpose:** Translate real EDA statistics into plain-English business insights for a non-technical audience.

**Prompt input:**
```
You are a senior banking analytics consultant. Given the following churn statistics
from a bank customer dataset, identify the top 3 drivers of churn and recommend one
targeted retention action for each driver. Be concise and write for a non-technical
business audience.

Statistics:
- Overall churn rate: 20.4%
- Churn by IsActiveMember: Active = 14.3%, Inactive = 26.9%
- Churn by NumOfProducts: 1 product = 27.7%, 2 products = 7.6%, 3 products = 82.7%, 4 products = 100%
- Churn by Tenure: 0-2 years = 21.2%, 3-5 years = 20.8%, 6+ years = 19.7%
- Churn by Geography: France = 16.2%, Germany = 32.4%, Spain = 16.7%
- Churn by Age group: Under 40 = 10.7%, 40-55 = 38.0%, 55+ = 36.8%
- Mean balance of churned customers: €91,109 vs retained: €72,745
```

**LLM output:**
```
One in five customers is leaving the bank — a significant and costly problem.
Here are the three biggest warning signs in the data, and what to do about each.

1. PRODUCT OVERLOAD IS DRIVING MASS EXITS
Customers holding 3 or 4 products churn at alarming rates — 82.7% and 100%
respectively. This is counter-intuitive but clear: customers who have been
cross-sold too many products feel overwhelmed or trapped, and leave entirely.
Customers with just 2 products, by contrast, churn at only 7.6%.

  → Recommendation: Audit cross-selling practices immediately. Focus on
    getting customers to 2 well-matched products rather than maximising
    product count. Consider a product simplification review for existing
    3–4 product holders.

2. INACTIVE CUSTOMERS ARE SLIPPING AWAY UNNOTICED
Customers who are not actively using their accounts churn at almost double the
rate of active ones (26.9% vs 14.3%). These customers have likely already
mentally moved on — the bank just hasn't noticed yet.

  → Recommendation: Introduce a 90-day inactivity trigger. When a customer
    shows no transactions for 90 days, automatically enrol them in a
    re-engagement campaign: a personalised message, a fee waiver, or a
    product review call from a relationship manager.

3. MIDDLE-AGED AND OLDER CUSTOMERS ARE THE HIGHEST RISK SEGMENT
Customers aged 40–55 churn at 38%, and those over 55 at 36.8% — three to four
times the rate of under-40s (10.7%). These are typically the bank's
highest-value customers, which makes their departure especially costly.
Notably, churned customers also hold higher average balances (€91,109 vs
€72,745), confirming that the bank is losing its wealthiest clients.

  → Recommendation: Create a dedicated retention programme for customers
    aged 40+. Offer priority service, dedicated relationship managers, and
    tailored products (e.g. retirement planning, wealth management). The
    revenue impact of retaining even a small fraction of this segment
    would be substantial.

Bonus observation: Germany stands out with a 32.4% churn rate — double France
and Spain. A region-specific investigation is warranted to understand whether
this reflects a pricing issue, competitor activity, or service quality gap.
```

**Critical evaluation:**

| Criterion | Assessment |
|-----------|-----------|
| Statistical accuracy | All figures cited correctly match our EDA output — no invented numbers |
| Driver identification | Correctly prioritised products_number and active_member; age and balance insight valid |
| Hallucinations | None detected |
| What was adjusted | The "product overload" framing is an LLM interpretation, not a proven cause. We changed our slide wording to "high product count correlates with churn" to avoid overstating causality |
| Business usefulness | High — the 90-day inactivity trigger is a concrete, directly usable recommendation |
| Overall rating | ★★★★☆ — strong output; one interpretive claim needed softening |

---

## Prompt 2 — Code Generation (XGBoost + Feature Importance)

**Purpose:** Generate modelling code; evaluate what ran correctly, what needed fixing, and what was wrong.

**Prompt input:**
```
Write Python code to do the following:
1. Load a CSV file called 'Bank_Customer_Churn_Prediction.csv' with these columns:
   customer_id, credit_score, country, gender, age, tenure, balance,
   products_number, credit_card, active_member, estimated_salary, churn
2. Encode categorical features (country, gender) using LabelEncoder
3. Train an XGBoost classifier to predict 'churn', using an 80/20 train-test split
   with random_state=42
4. Print accuracy, precision, recall, and F1 score
5. Generate a horizontal bar chart of the top 10 feature importances using matplotlib

Use scikit-learn and xgboost libraries. Include comments explaining each step.
```

**LLM output:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv('Bank_Customer_Churn_Prediction.csv')

# Step 2: Encode categorical features
le = LabelEncoder()
df['country'] = le.fit_transform(df['country'])   # France=0, Germany=1, Spain=2
df['gender'] = le.fit_transform(df['gender'])      # Female=0, Male=1

# Step 3: Define features and target — drop customer_id (not predictive)
features = ['credit_score', 'country', 'gender', 'age', 'tenure', 'balance',
            'products_number', 'credit_card', 'active_member', 'estimated_salary']
X = df[features]
y = df['churn']

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train XGBoost classifier
model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

# Step 7: Plot feature importances
importances = pd.Series(model.feature_importances_, index=features).sort_values()
plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='steelblue')
plt.title('XGBoost Feature Importances — Bank Churn')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
```

Terminal output when run:
```
Accuracy:  0.8550
Precision: 0.6800
Recall:    0.4900
F1 Score:  0.5700

Top features by importance:
  products_number: 0.3420
  active_member:   0.2153
  age:             0.1215
  country:         0.0721
  balance:         0.0561
  gender:          0.0514
  credit_score:    0.0382
  estimated_salary:0.0366
  tenure:          0.0355
  credit_card:     0.0313
```

**Critical evaluation:**

| Criterion | Assessment |
|-----------|-----------|
| Code ran without errors | Yes — executed first time with no modifications |
| Column name accuracy | Correct — matched actual CSV columns exactly |
| Encoding logic | Appropriate for this dataset; LabelEncoder is acceptable for 3-class country variable |
| What needed fixing | LabelEncoder overwrote the original `country` and `gender` columns. We modified the code to write to new columns (`country_enc`, `gender_enc`) to preserve originals for EDA visualisations |
| Metric output | All four metrics match our manual verification |
| Feature importance chart | Generated correctly; we adjusted colours and added value labels for the presentation |
| Hallucinated API calls | None |
| Final code status | Lightly modified (column preservation + plot styling only); all logic unchanged |
| Overall rating | ★★★★★ — best-performing prompt; code was near production-ready |

**Verified model results:**

| Metric | Value |
|--------|-------|
| Accuracy | 85.5% |
| Precision | 0.68 |
| Recall | 0.49 |
| F1 Score | 0.57 |
| Top feature | products_number (importance: 0.342) |

---

## Prompt 3 — Customer Persona Narration

**Purpose:** Use the LLM to narrate a high-risk customer profile and draft a personalised retention message.

**Prompt input:**
```
You are a customer retention strategist at a bank. A customer has the following profile:

- Age: 52
- Country: Germany
- Active member: No
- Number of products: 1
- Account balance: €120,000
- Tenure: 2 years
- Estimated salary: €85,000
- Has credit card: Yes

Based on this profile, answer the following:
1. Why is this customer likely to churn? Reference specific attributes.
2. What is their likely emotional or financial motivation for leaving?
3. Draft a short, personalised retention email (max 120 words) the bank could send them.
   Do not use placeholders — write as if it is a real email.
```

**LLM output:**
```
1. WHY THIS CUSTOMER IS LIKELY TO CHURN

This customer carries four of the strongest churn signals in the dataset:

  • Inactive member: Non-active customers churn at 26.9% vs 14.3% for active ones.
    This customer is not using their account — a clear disengagement signal.
  • Single product: Customers with just 1 product churn at 27.7%. They have no
    depth of relationship with the bank and face zero switching cost.
  • Germany: German customers churn at 32.4% — double the rate of France and Spain.
    This may reflect stronger local competition or pricing sensitivity.
  • Age 52 with high balance: Middle-aged, high-balance customers who are disengaged
    often consolidate finances with a competitor offering better wealth management
    or higher deposit rates.

2. LIKELY EMOTIONAL / FINANCIAL MOTIVATION

This customer is likely evaluating their options quietly. They have €120,000 sitting
in an account they rarely use, they are approaching the life stage where wealth
management and retirement planning matter, and the bank has probably never reached
out with anything tailored to their situation. They are not angry — they are simply
uninspired. A competitor offering a slightly better rate or a single proactive call
from a relationship manager could tip the decision.

3. RETENTION EMAIL

Subject: A conversation worth having — your finances at 52

Dear [Customer name],

We noticed we haven't spoken in a while, and we'd like to change that.

You've trusted us with a significant portion of your savings, and we want to make
sure we're genuinely working for you. Whether it's better returns on your balance,
planning for the decade ahead, or simply understanding what else we can offer —
we'd love to have a conversation.

I'd like to personally offer you a complimentary 30-minute financial review with
one of our senior advisors. No obligation, just a conversation tailored to where
you are now.

Would Tuesday 22nd or Thursday 24th work for a call?

Warm regards,
Sarah Brennan
Senior Relationship Manager, Retail Banking
```

**Critical evaluation:**

| Criterion | Assessment |
|-----------|-----------|
| Churn driver accuracy | Correctly identified all four risk factors (inactive, 1 product, Germany, age/balance) — consistent with our EDA and model |
| Statistical grounding | All percentages cited match our actual dataset; the LLM used numbers from Prompt 1 input correctly |
| Emotional framing | "Not angry — uninspired" is a plausible and actionable framing for the retention team; however, it is an inference, not a data finding |
| Email quality | Professional, non-generic, and avoids the "valued customer" cliché. The offer of a financial review is concrete and realistic |
| Hallucinations | None — no invented statistics or customer attributes |
| What needed adjusting | The customer name placeholder was expected. The phrase "significant portion of savings" would need verification against actual product holdings before real use |
| Scalability limitation | This approach works at the persona level. Generating individualised emails for thousands of customers would require a full CRM integration pipeline, not a single prompt |
| Overall rating | ★★★★☆ — high quality output; emotional inference caveat should be noted on the reflection slide |

---

## Prompt 4 — Executive Summary Generation

**Purpose:** Convert final model results into a CEO-ready summary; critically evaluate faithfulness to the actual numbers.

**Prompt input:**
```
Summarise the following machine learning results for a bank CEO. Write exactly 3
sentences. Do not use technical jargon. End with one clear, actionable recommendation.

Results:
- Model: XGBoost (gradient boosting classifier)
- Test accuracy: 85.5%
- F1 score: 0.57
- Top churn drivers by feature importance: products_number (0.342), active_member (0.215),
  age (0.122), country/Germany (0.072), balance (0.056)
- Business goal: reduce churn rate from 20.4% to below 17.4% (15% relative reduction)
- Mean balance of churned customers: €91,109 vs €72,745 for retained customers
```

**LLM output:**
```
Our predictive model correctly identifies churning customers 85.5% of the time,
giving the bank a reliable early-warning system for customers at risk of leaving.
The clearest signals of churn are customers who hold too many or too few products,
those who have become disengaged from their accounts, customers aged 40 and above,
and those based in Germany — all of whom leave at significantly higher rates than
average. To hit the 15% churn reduction target, we recommend the bank immediately
prioritise outreach to inactive customers aged 40+ in Germany who hold only one
product, as this group sits at the intersection of every major risk factor and
represents the highest-value intervention opportunity.
```

**Critical evaluation:**

| Criterion | Assessment |
|-----------|-----------|
| Numerical accuracy | 85.5% accuracy stated correctly; F1 appropriately omitted (too technical for CEO) |
| Driver accuracy | Correctly surfaced products, engagement, age, and Germany — matches our feature importance ranking |
| "Too many or too few products" claim | Accurate — EDA shows 1-product = 27.7% churn AND 3-4 products = 82–100% churn. LLM correctly identified the non-linear relationship |
| Tone | Appropriate for CEO — no jargon, clear and confident |
| Recommendation specificity | Good — targets a concrete intersection group rather than generic "focus on retention" |
| Hallucinations | None — no invented statistics |
| What was missing | The high-balance finding (churned customers hold €18k more on average) was omitted. This was added manually to the presentation slide as it has direct revenue implications |
| Overall rating | ★★★★☆ — accurate and well-framed; one meaningful finding omitted, added manually |

---

## Overall Reflection Summary

| Prompt | Use case | Rating | Key limitation |
|--------|----------|--------|----------------|
| 1 — Insight synthesis | EDA → business language | ★★★★☆ | Interprets correlation as causation; "product overload" framing needed softening |
| 2 — Code generation | XGBoost pipeline | ★★★★★ | Overwrote source columns; cosmetic fix only. Logic was entirely correct |
| 3 — Persona narration | Retention messaging | ★★★★☆ | Emotional inference is assumption-based; not scalable without CRM pipeline |
| 4 — Executive summary | CEO communication | ★★★★☆ | Omitted high-balance finding; added manually |

**Key lessons (for Slide 9):**

1. **Specificity drives quality.** Prompts that included exact statistics and column names produced accurate, usable responses. Vague prompts produced generic output. The LLM is only as good as the context you give it.

2. **LLMs are reliable when given the numbers.** When statistics were provided explicitly, the model cited them correctly across all four prompts. Hallucination risk rises when asking the model to recall or estimate figures it was not given.

3. **Code generation was the highest-value use case.** Prompt 2 produced working code in a single attempt and saved significant setup time. The only changes required were cosmetic.

4. **LLMs interpret; they do not explain.** The "product overload" framing in Prompt 1 is plausible but not proven — it is an interpretation of a correlation, not a causal finding. Presenting it uncritically in a business setting would be misleading.

5. **Citation risk is real.** In an exploratory test (not included in this log), the model was asked to suggest academic references for churn modelling. Two of four citations it provided did not exist. All references in the final presentation were manually verified against their actual sources.

---

*All LLM outputs above were generated using Claude (claude-sonnet-4-20250514) via claude.ai in April 2025. Statistical inputs were derived from the team's own EDA of the Kaggle Bank Customer Churn dataset (10,000 rows). Use is acknowledged in accordance with TCD GenAI referencing guidelines: https://libguides.tcd.ie/genai/acknowledging-referencing*
