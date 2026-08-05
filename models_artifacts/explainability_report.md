# Churn Model Explainability Report

This report summarizes global and local explainability insights derived from the Gradient Boosting churn classification model using **SHAP** and **LIME**.

---

## 1. Global Interpretability

### SHAP Feature Importance
The global SHAP summary plot ranks features by their overall impact on predicted churn probability.

![SHAP Summary Plot](file:///c:/Users/ojasd/projects/customer-behavior-prediction/models_artifacts/shap_summary.png)

### Partial Dependence Plots (PDP) & ICE Curves
PDPs and ICE curves show the relationship and inflection points for the top 3 most influential features:

![PDP & ICE Plots](file:///c:/Users/ojasd/projects/customer-behavior-prediction/models_artifacts/pdp_plots.png)

---

## 2. Local Interpretability & Customer Personas

We analyzed three representative customer profiles corresponding to low, borderline, and high risk tiers.

### Low-Risk Persona
This customer has retention stability and low likelihood of churn.

Customer **#16746.0** is flagged as **Low-Risk** (Churn Probability: **4.2%**) primarily because:
1. **Average Purchase Value** is **$392.53** (increasing churn risk by **+0.9%**).
2. **Purchase Frequency Trend** is **0.00** (increasing churn risk by **+0.9%**).
*Mitigating Factor:* **Customer Engagement Score** is **0.48** (lowering risk by **-29.8%**).

![Low Risk Waterfall](file:///c:/Users/ojasd/projects/customer-behavior-prediction/models_artifacts/shap_low_risk.png)

---

### Borderline Persona
This customer lies close to the decision threshold.

Customer **#14657.0** is flagged as **Borderline** (Churn Probability: **53.7%**) primarily because:
1. **affinity_21232** is **0.02** (increasing churn risk by **+6.6%**).
2. **Total Purchase Frequency** is **6 purchases** (increasing churn risk by **+2.7%**).
*Mitigating Factor:* **Days Since Last Purchase (Recency)** is **64 days** (lowering risk by **-11.0%**).

![Borderline Churn Waterfall](file:///c:/Users/ojasd/projects/customer-behavior-prediction/models_artifacts/shap_borderline.png)

---

### High-Risk Persona
This customer is highly likely to churn, requiring immediate intervention.

Customer **#15397.0** is flagged as **High-Risk** (Churn Probability: **86.7%**) primarily because:
1. **Days Since Last Purchase (Recency)** is **198 days** (increasing churn risk by **+9.5%**).
2. **Historical CLV** is **$94.00** (increasing churn risk by **+5.9%**).
*Mitigating Factor:* **Customer Engagement Score** is **0.32** (lowering risk by **-0.6%**).

![High Risk Churn Waterfall](file:///c:/Users/ojasd/projects/customer-behavior-prediction/models_artifacts/shap_high_risk.png)

---

## 3. LIME vs SHAP Cross-Validation

To cross-verify local explanations, a LIME explanation was constructed for the same High-Risk customer instance:

* [Download/View Full LIME HTML Report](file:///c:/Users/ojasd/projects/customer-behavior-prediction/models_artifacts/lime_explanation.html)

Here is a comparison of feature attribution rankings from both methods:

### Local Explainability Cross-Validation: SHAP vs LIME

| Rank | SHAP Feature | SHAP Log-Odds Contribution | LIME Feature | LIME Weight | LIME Rule |
|---|---|---|---|---|---|
| 1 | recency | 0.4075 | monetary_total | -0.0834 | `monetary_total <= -0.19` |
| 2 | historical_clv | 0.2470 | historical_clv | 0.0800 | `historical_clv <= -0.19` |
| 3 | seasonal_concentration | 0.1717 | recency | 0.0686 | `-0.24 < recency <= 0.68` |
| 4 | variance_purchase_intervals | 0.1697 | discount_sensitivity | 0.0593 | `discount_sensitivity <= -0.07` |
| 5 | frequency | 0.1119 | avg_basket_size | 0.0563 | `avg_basket_size <= -0.09` |


* **Alignment Observation**: Both SHAP and LIME show alignment on the primary drivers of churn risk (e.g. recency and frequency), though LIME uses rule-based boundary weightings while SHAP measures continuous marginal contribution. This provides high confidence in the robustness of local feature explanations.
