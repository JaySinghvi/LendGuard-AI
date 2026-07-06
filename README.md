# Analyzing Credit Risk

A two-part project on predicting loan default risk from applicant financial data:

1. **Statistical analysis** (`Final_Project.Rmd` / `.pdf`) — logistic regression and random forest models identifying the strongest predictors of loan default.
2. **LendGuard** (`App/`) — a Shiny web app that scores a loan applicant with a trained XGBoost model and uses Claude to draft a plain-English, ECOA/Regulation B-style credit decision notice.

Live app: https://jsinghvi.shinyapps.io/lendguard/

## Dataset

[`credit_risk_dataset.csv`](credit_risk_dataset.csv) — the Kaggle [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset), a simulated credit bureau dataset with 32,581 rows and 12 variables, including:

| Column | Description |
|---|---|
| `person_age`, `person_income`, `person_emp_length` | Borrower age, income, employment length |
| `person_home_ownership` | Home ownership status |
| `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate` | Loan purpose, grade, amount, interest rate |
| `loan_percent_income` | Loan amount as a percentage of income |
| `cb_person_default_on_file`, `cb_person_cred_hist_length` | Prior default flag, credit history length |
| `loan_status` | Target: 1 = defaulted, 0 = did not default |

## 1. Statistical Analysis (`Final_Project.Rmd`)

Cleans the dataset (removes missing values and unrealistic outliers in age, employment length, income, and loan amount, leaving 25,701 observations), then fits and compares three logistic regression models plus a random forest:

- **LR1** — full model (all predictors except `loan_grade`)
- **LR2** — LR1 with statistically insignificant predictors removed
- **LR3** — stepwise-selected model using `person_home_ownership`, `loan_intent`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, and `cb_person_cred_hist_length`
- **Random forest** on LR3's predictors, to rank variable importance

**Conclusion:** home ownership, loan intent, loan amount, interest rate, loan-to-income ratio, and credit history length are the most significant predictors of default. Prior default on file was a surprisingly weak predictor.

Render with:
```r
rmarkdown::render("Final_Project.Rmd")
```

## 2. LendGuard App (`App/`)

A Shiny app that:
1. Takes applicant inputs (home ownership, loan purpose, amount, interest rate, income, credit history length).
2. Scores default probability with a pretrained XGBoost model (`lendguard_xgb.model`, `lendguard_meta.rds`) against a decision threshold.
3. Surfaces the top contributing risk factors via SHAP-style prediction contributions.
4. Calls Claude (via the `ellmer` package) to generate a compliant, human-readable decision notice citing the applicant's actual figures.

### Running locally

```r
install.packages(c("shiny", "xgboost", "caret", "ellmer"))
```

Set an `ANTHROPIC_API_KEY` (e.g. in `App/.Renviron`) to enable AI-generated decision notices; without it, the app falls back to a plain summary of the score and top risk factors.

```r
shiny::runApp("App")
```

**Note:** `App/.Renviron` is a local secrets file and should never be committed to version control.

## Repository Structure

```
credit_risk_dataset.csv       Source dataset
Final_Project.Rmd / .pdf      Statistical analysis and writeup
App/
  app.R                       Shiny app source
  lendguard_xgb.model         Trained XGBoost booster
  lendguard_meta.rds          Feature metadata, factor levels, decision threshold
  .Renviron                   Local environment variables (not for version control)
```
