# Data dictionary

`grocery_data.csv` contains 2,000 **synthetic** household-month records generated with NumPy seed 42. It is not a real survey and should not be interpreted as evidence about Indian households.

| Column | Unit / range | Meaning |
|---|---|---|
| household_size | people, 1–8 generated | Number of people in household |
| monthly_household_income_inr | INR/month | Household income, right-skewed synthetic distribution |
| shopping_trips_per_month | trips/month, 1–30 | Number of grocery shopping trips |
| avg_items_per_trip | items/trip, 3–70 | Average grocery items in each trip |
| branded_spending_pct | percent, 5–85 | Share allocated to branded products |
| fresh_produce_spending_pct | percent, 12–90 | Share allocated to fresh produce |
| online_shopping_pct | percent, 0–85 | Share of grocery shopping done online |
| previous_month_expenditure_inr | INR/month | Historical spend observed before the prediction month |
| monthly_grocery_expenditure_inr | INR/month | Target: current month's grocery expenditure |

Generation uses household-size-dependent shopping habits, bounded income and percentage distributions, a latent demand baseline, and independent random noise. Current spend blends that baseline with the prior month's observed spend plus additional noise. The historical value is included because it is available at the start of the current month; no current-month target information is used to form it. This constructed relationship is a teaching device and does not represent measured behavior.
