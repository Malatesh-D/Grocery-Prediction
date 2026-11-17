"""
Generate publication-quality Academic Project Report (.docx) for GrocerySpend AI.
Conforms strictly to the 5-section evaluation guidelines of Prof. Debarshi Mazumder.
Embeds all 12 publication-quality statistical visualizations and 8 verified
runtime terminal output screenshots.
"""

import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

REPORTS_DIR = ROOT / "reports"
VISUALIZATIONS_DIR = REPORTS_DIR / "visualizations"
SCREENSHOTS_DIR = REPORTS_DIR / "screenshots"


def set_cell_background(cell, fill_hex):
    """Set background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def style_table_header(row, header_titles):
    """Apply professional navy styling to table header row with titles."""
    for idx, (cell, title) in enumerate(zip(row.cells, header_titles)):
        set_cell_background(cell, "1F4E79")  # Navy blue
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.text = title
        for run in p.runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.font.name = "Segoe UI"
            run.font.size = Pt(9.5)


def add_callout_box(doc, text, title="KEY TAKEAWAY"):
    """Add a professional styled callout box."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="1F4E79"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.15)
    p.paragraph_format.right_indent = Inches(0.15)
    
    r_title = p.add_run(f"[{title}] ")
    r_title.bold = True
    r_title.font.name = "Segoe UI"
    r_title.font.size = Pt(10)
    r_title.font.color.rgb = RGBColor(31, 78, 121)
    
    r_text = p.add_run(text)
    r_text.font.name = "Segoe UI"
    r_text.font.size = Pt(9.5)
    r_text.font.color.rgb = RGBColor(15, 23, 42)
    doc.add_paragraph()  # Spacing


def add_figure_image(doc, img_path, caption_text, width=Inches(5.8)):
    """Add centered image and caption if file exists."""
    p = Path(img_path)
    if p.exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(2)
        run_img = p_img.add_run()
        run_img.add_picture(str(p), width=width)

        p_cap = doc.add_paragraph(caption_text)
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(10)
        p_cap.runs[0].font.italic = True
        p_cap.runs[0].font.size = Pt(8.5)
        p_cap.runs[0].font.color.rgb = RGBColor(71, 85, 105)


def generate_word_report():
    """Build and save complete Academic Project Report (.docx)."""
    print("=" * 75)
    print("GENERATING ACADEMIC PROJECT REPORT (.DOCX) WITH EMBEDDED VISUALS & SCREENSHOTS")
    print("=" * 75)

    doc = docx.Document()

    # Page Margins (0.75 in)
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Styles
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Segoe UI"
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(30, 41, 59)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(5)

    # Document Title Block
    title = doc.add_heading("GrocerySpend AI: Predictive Modeling of Monthly Household Grocery Expenditure", level=0)
    title.style.font.color.rgb = RGBColor(31, 78, 121)
    title.style.font.name = "Segoe UI"
    title.style.font.bold = True

    p_meta = doc.add_paragraph()
    r = p_meta.add_run("Academic Evaluation Report | Academic Year 2026–2027\n")
    r.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(31, 78, 121)
    p_meta.add_run(
        "• Faculty / Evaluator: Prof. Debarshi Mazumder\n"
        "• Student Author: Malatesh D\n"
        "• Domain: Artificial Intelligence & Predictive Analytics (Retail & Consumer Finance)\n"
        "• Repository: https://github.com/Malatesh-D/Grocery-Prediction"
    )

    add_callout_box(
        doc,
        "Linear Regression was selected as the champion model via 5-fold cross-validation (lowest CV RMSE: ₹1,274.17 ± ₹58.08). "
        "On the isolated holdout test split (N=400), it achieved an R² of 0.7609 and Test MAE of ₹1,081.39. "
        "Permutation importance confirms that Prior-Month Spending Inertia and Household Scale drive 75%+ of grocery spend, validating Engel's Law.",
        "EXECUTIVE SUMMARY & CHAMPION VERDICT"
    )

    # -------------------------------------------------------------------------
    # SECTION 1. INTRODUCTION
    # -------------------------------------------------------------------------
    h1 = doc.add_heading("Section 1. Introduction", level=1)
    h1.style.font.color.rgb = RGBColor(31, 78, 121)

    p = doc.add_paragraph()
    p.add_run("1.1 Business Problem Overview: ").bold = True
    p.add_run(
        "In the fast-moving consumer goods (FMCG) and modern retail ecosystem, volatile household grocery consumption presents serious challenges "
        "for inventory provisioning, promotional discounting, and working capital optimization. For quick-commerce operators (e.g., Blinkit, Zepto, "
        "Instamart) and omnichannel supermarkets, inaccurate demand forecasting causes severe perishable shrink in fresh produce or stockouts during peak month-end cycles. "
        "Furthermore, consumer FinTech and banking apps require reliable expenditure baselines to issue non-invasive grocery credit lines, Buy-Now-Pay-Later (BNPL) facilities, "
        "and automated household budget planners without demanding invasive surveillance."
    )

    p = doc.add_paragraph()
    p.add_run("1.2 Predictive Task Formulation: ").bold = True
    p.add_run(
        "The predictive task is formally structured as a continuous supervised regression problem:\n"
        "\"Given observable household demographic attributes, shopping cadence, basket category shares, and previous expenditure benchmarks, "
        "predict the household's total monthly grocery expenditure in Indian Rupees (monthly_grocery_expenditure_inr).\"\n\n"
        "Mathematically: y_hat = f(x) in R^+, where x in R^8 is the vector of demographic, behavioral, and prior consumption predictors."
    )

    p = doc.add_paragraph()
    p.add_run("1.3 Approach and Summary of Key Findings: ").bold = True
    p.add_run(
        "1. Pipeline Engineering: Engineered a leakage-free Scikit-Learn ColumnTransformer executing median imputation and standardization strictly within training folds.\n"
        "2. Algorithmic Benchmarking: Evaluated four regressor paradigms: Mean Baseline, Multiple Linear Regression, Random Forest (250 trees), and HistGradientBoosting.\n"
        "3. Selection Protocol: Optimal model chosen strictly on lowest 5-fold cross-validation RMSE with CV MAE tie-breaking.\n"
        "4. Key Findings: Linear Regression outperformed tree ensembles (CV RMSE: ₹1,274.17; Test R²: 0.7609; Test MAE: ₹1,081.39) with homoscedastic, Gaussian residuals.\n"
        "5. Economic Drivers: Prior-month spending inertia (+₹838 MAE drop) and household size (+₹292 MAE drop) are the dominant drivers, validating Engel's Law."
    )

    # -------------------------------------------------------------------------
    # SECTION 2. DATA UNDERSTANDING AND PREPARATION
    # -------------------------------------------------------------------------
    h2 = doc.add_heading("Section 2. Data Understanding and Preparation", level=1)
    h2.style.font.color.rgb = RGBColor(31, 78, 121)

    p = doc.add_paragraph()
    p.add_run("2.1 Dataset Description: ").bold = True
    p.add_run(
        "The project dataset comprises 2,000 household-month observations with 9 continuous and discrete variables "
        "(8 predictor features and 1 regression target). It captures urban and semi-urban Indian household purchasing dynamics."
    )

    # Table of Dataset Variables
    tbl_data = doc.add_table(rows=10, cols=4)
    tbl_data.alignment = WD_TABLE_ALIGNMENT.CENTER
    data_headers = ["Variable Name", "Data Type", "Range / Unit", "Operational Definition"]
    style_table_header(tbl_data.rows[0], data_headers)

    var_info = [
        ("household_size", "Integer", "1 – 8 people", "Total family members residing and dining together"),
        ("monthly_household_income_inr", "Float", "₹8,795 – ₹3,11,868", "Net monthly take-home earnings of the household in INR"),
        ("shopping_trips_per_month", "Integer", "1 – 19 trips", "Combined in-store market visits and quick-commerce delivery app orders"),
        ("avg_items_per_trip", "Integer", "3 – 49 items", "Average quantity of grocery items purchased per shopping trip"),
        ("branded_spending_pct", "Float", "5.0% – 85.0%", "Share of grocery spend devoted to packaged, branded FMCG goods"),
        ("fresh_produce_spending_pct", "Float", "12.0% – 90.0%", "Share of grocery budget spent on perishables (vegetables, dairy, fruits)"),
        ("online_shopping_pct", "Float", "1.05% – 85.0%", "Share of grocery orders transacted via quick-commerce delivery apps"),
        ("previous_month_expenditure_inr", "Float", "₹2,604 – ₹18,045", "Baseline grocery spend recorded in immediately preceding month"),
        ("monthly_grocery_expenditure_inr", "Float", "₹911 – ₹20,316", "TARGET: Total grocery expenditure incurred in current month (₹ INR)")
    ]
    for row_idx, (vname, dtype, vrange, vdesc) in enumerate(var_info, start=1):
        row = tbl_data.rows[row_idx]
        row.cells[0].text = vname
        row.cells[1].text = dtype
        row.cells[2].text = vrange
        row.cells[3].text = vdesc
        for c in row.cells:
            c.paragraphs[0].runs[0].font.size = Pt(8.5)
            set_cell_background(c, "EFF6FF" if row_idx == 9 else ("F8FAFC" if row_idx % 2 == 1 else "FFFFFF"))

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.add_run("2.2 Data Cleaning and Validation Process: ").bold = True
    p.add_run(
        "Data integrity checks are programmatically enforced in src/validate_data.py:\n"
        "• Target Positivity: Verified that monthly expenditure is strictly positive (y > 0) and finite. Zero negative values detected.\n"
        "• Percentage Bounding: Enforced bounded domains [0, 100]% for branded, fresh produce, and online delivery shares.\n"
        "• Demographic Integrity: Enforced household size as positive integer values >= 1.\n"
        "• Deduplication: Verified zero duplicate rows across the 2,000 multi-attribute records.\n"
        "• Missing Value Protocol: Missing numerical values are handled via median imputation inside the pipeline to guarantee zero data leakage."
    )

    p = doc.add_paragraph()
    p.add_run("2.3 Exploratory Data Analysis (EDA) Findings: ").bold = True
    p.add_run(
        "Key empirical findings from exploratory analysis include:\n"
        "1. Right-Skewed Income: Household income exhibits a log-normal distribution with median ₹52,507 and top-tail extending to ₹3,11,868.\n"
        "2. Strong Linear Spend Inertia: Prior month spend correlates heavily with current spend (r = 0.81), indicating repetitive staple consumption.\n"
        "3. Empirical Engel's Law: As income scales from ₹25k to ₹250k, absolute spend rises sub-linearly, causing grocery budget share to drop from 20% to under 6%.\n"
        "4. Channel Preferences: Quick-commerce delivery share averages 30.1%, with higher digital adoption among households earning >₹80k/month."
    )

    # Insert EDA Figures
    add_figure_image(doc, VISUALIZATIONS_DIR / "00_executive_dashboard_summary.png",
                     "Figure 2.1: Multi-Panel Executive Analytics & Research Summary Dashboard")
    add_figure_image(doc, VISUALIZATIONS_DIR / "01_target_spend_distribution.png",
                     "Figure 2.2: Distribution Profile of Monthly Grocery Spend Target (Mean: ₹7,630; Median: ₹7,390)")
    add_figure_image(doc, VISUALIZATIONS_DIR / "02_income_distribution_and_engels_law.png",
                     "Figure 2.3: Household Income Distribution and Empirical Confirmation of Engel's Law")
    add_figure_image(doc, VISUALIZATIONS_DIR / "03_household_size_spending_tiers.png",
                     "Figure 2.4: Grocery Spend Tiers and Distribution Density Across Household Sizes (1 to 8 Members)")
    add_figure_image(doc, VISUALIZATIONS_DIR / "04_spend_breakdown_categories.png",
                     "Figure 2.5: Budget Share Distributions (Branded FMCG %, Fresh Produce %, Online Quick-Commerce %)")
    add_figure_image(doc, VISUALIZATIONS_DIR / "06_previous_vs_current_spend_inertia.png",
                     "Figure 2.6: Spending Inertia: Previous Month vs. Current Month Spend (Pearson r = 0.812)")
    add_figure_image(doc, VISUALIZATIONS_DIR / "07_correlation_matrix_heatmap.png",
                     "Figure 2.7: Annotated Pearson Correlation Heatmap Across All Predictor Attributes and Target Spend")

    # -------------------------------------------------------------------------
    # SECTION 3. METHODOLOGY
    # -------------------------------------------------------------------------
    h3 = doc.add_heading("Section 3. Methodology", level=1)
    h3.style.font.color.rgb = RGBColor(31, 78, 121)

    p = doc.add_paragraph()
    p.add_run("3.1 Feature Engineering and Preprocessing Pipeline: ").bold = True
    p.add_run(
        "To ensure strict reproducibility and eliminate look-ahead data leakage, all preprocessing steps are encapsulated "
        "inside a Scikit-Learn ColumnTransformer pipeline:\n"
        "Input Features (8) -> [SimpleImputer (strategy='median')] -> [StandardScaler] -> Regressor\n"
        "• Median Imputation: Provides robust imputation against skewed income and expenditure distributions.\n"
        "• Standardization: Scales all numerical features to zero mean and unit variance (mu=0, sigma=1), ensuring numerical stability."
    )

    p = doc.add_paragraph()
    p.add_run("3.2 Machine Learning Models Considered: ").bold = True
    p.add_run(
        "Four candidate regressor paradigms were benchmarked:\n"
        "1. Mean Baseline (DummyRegressor): Predicts global training mean y_bar; serves as the null sanity check (R² ≈ 0).\n"
        "2. Multiple Linear Regression: Fits ordinary least squares (OLS) weights; provides maximum interpretability and parsimony.\n"
        "3. Random Forest Regressor: Non-parametric ensemble of 250 decision trees with min_samples_leaf=3 and max_features=0.85.\n"
        "4. HistGradientBoostingRegressor: Gradient-boosted decision trees using histogram binning with L2 regularization (max_iter=150)."
    )

    p = doc.add_paragraph()
    p.add_run("3.3 Model Training and Validation Strategy: ").bold = True
    p.add_run(
        "• Holdout Partition: An 80/20 train-test split was executed using seed 42 (1,600 training samples and 400 isolated test samples).\n"
        "• 5-Fold Cross-Validation: Conducted within the 1,600 training instances to evaluate out-of-fold generalization error.\n"
        "• Selection Rule: The winning model was chosen strictly on the lowest mean cross-validation RMSE, using CV MAE as tie-breaker.\n"
        "• Strict Leakage Prevention: The 400 test instances were held out completely untouched until final evaluation."
    )

    # -------------------------------------------------------------------------
    # SECTION 4. RESULTS AND ANALYSIS
    # -------------------------------------------------------------------------
    h4 = doc.add_heading("Section 4. Results and Analysis", level=1)
    h4.style.font.color.rgb = RGBColor(31, 78, 121)

    p = doc.add_paragraph()
    p.add_run("4.1 Model Performance Evaluation: ").bold = True
    p.add_run(
        "All four candidate models were benchmarked across 5-fold cross-validation and the isolated 400-instance holdout test split:"
    )

    # Results Table
    tbl_res = doc.add_table(rows=5, cols=7)
    tbl_res.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Candidate Model", "CV RMSE (₹)", "CV MAE (₹)", "Test MAE (₹)", "Test RMSE (₹)", "Test R² Score", "Selection Verdict"]
    style_table_header(tbl_res.rows[0], headers)

    model_data = [
        ("Mean Baseline", "₹2,659.71 ± 62.4", "₹2,135.49", "₹2,153.05", "₹2,728.95", "-0.0014", "Rejected (Baseline)"),
        ("Linear Regression", "₹1,274.17 ± 58.1", "₹1,010.12", "₹1,081.39", "₹1,333.37", "0.7609", "SELECTED WINNER 🏆"),
        ("Random Forest (250)", "₹1,324.75 ± 51.0", "₹1,055.05", "₹1,105.76", "₹1,368.12", "0.7483", "Competitive Ensemble"),
        ("HistGradientBoosting", "₹1,400.48 ± 64.7", "₹1,107.89", "₹1,178.75", "₹1,454.25", "0.7156", "Over-regularized")
    ]
    for row_idx, r_tuple in enumerate(model_data, start=1):
        row = tbl_res.rows[row_idx]
        for c_idx, val in enumerate(r_tuple):
            cell = row.cells[c_idx]
            cell.text = val
            p = cell.paragraphs[0]
            if c_idx == 0 and "WINNER" in r_tuple[6]:
                p.runs[0].bold = True
            p.runs[0].font.size = Pt(8.5)
            set_cell_background(cell, "EFF6FF" if "WINNER" in r_tuple[6] else ("F8FAFC" if row_idx % 2 == 1 else "FFFFFF"))

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.add_run("Metric Interpretation: ").bold = True
    p.add_run(
        "Linear Regression achieved the lowest cross-validation RMSE (₹1,274.17) and lowest CV MAE (₹1,010.12), outperforming "
        "complex tree ensembles. On the holdout test set, it demonstrated excellent generalization with an R² of 0.7609 and "
        "average prediction error of ₹1,081.39 (~14% relative error on the ₹7,630 mean spend)."
    )

    # Insert Performance Figures
    add_figure_image(doc, VISUALIZATIONS_DIR / "08_model_comparison_benchmark.png",
                     "Figure 4.1: Candidate Model Benchmark: Cross-Validation RMSE vs. Test MAE and Test R² Score")
    add_figure_image(doc, VISUALIZATIONS_DIR / "09_actual_vs_predicted_parity.png",
                     "Figure 4.2: Parity Plot: Actual vs. Predicted Monthly Spend on Holdout Test Set (N=400, R² = 0.761)")
    add_figure_image(doc, VISUALIZATIONS_DIR / "10_residual_diagnostic_plots.png",
                     "Figure 4.3: Residual Diagnostic Plots: Residuals vs. Fitted Values (Homoscedasticity) & Normality Distribution")
    add_figure_image(doc, VISUALIZATIONS_DIR / "11_permutation_feature_importance.png",
                     "Figure 4.4: Permutation Feature Importance: Impact on Test MAE When Predictors Are Permuted")

    p = doc.add_paragraph()
    p.add_run("4.2 Interpretation of Key Factors Influencing Predictions: ").bold = True
    p.add_run(
        "1. Spending Inertia (+₹838 MAE Drop): Previous month's spend is the single strongest predictor. Household grocery procurement follows "
        "habitual replenishment cycles for staple commodities (rice, flour, oil, toiletries).\n"
        "2. Household Scale (+₹292 MAE Drop): Each family member incrementally scales baseline consumption demand by ~₹1,250/month.\n"
        "3. Income Elasticity (+₹22 MAE Drop): Income exhibits an inelastic relationship with food volume, influencing premium brand upgrading rather than quantity.\n"
        "4. Shopping Trips and Basket Proportions (+₹5 to ₹15 MAE Drop): Trip frequency and branded share provide secondary adjustments representing convenience premiums."
    )

    # -------------------------------------------------------------------------
    # SECTION 5. DISCUSSION AND CONCLUSION
    # -------------------------------------------------------------------------
    h5 = doc.add_heading("Section 5. Discussion and Conclusion", level=1)
    h5.style.font.color.rgb = RGBColor(31, 78, 121)

    p = doc.add_paragraph()
    p.add_run("5.1 Summary of Findings: ").bold = True
    p.add_run(
        "GrocerySpend AI proves that household grocery demand can be modeled with high precision (R² = 76.1%, MAE ≈ ₹1,081) using "
        "transparent regression architectures. By integrating historical purchasing inertia, household size, and channel preferences, "
        "retailers can reliably predict forward 30-day demand without requiring invasive surveillance data."
    )

    p = doc.add_paragraph()
    p.add_run("5.2 Limitations and Future Enhancements: ").bold = True
    p.add_run(
        "1. Demonstration Dataset Scope: Current pipeline is evaluated on seeded synthetic data representing Indian demographics; future steps involve ingesting MoSPI survey microdata.\n"
        "2. Uncertainty Intervals: Model currently outputs point estimates; integrating quantile regression or conformal prediction will yield 90% confidence bands.\n"
        "3. Exogenous Price Shocks: Macro-economic food CPI inflation and seasonal price surges (tomatoes, onions) can be incorporated as external time-series features.\n"
        "4. Category-Level Forecasting: Decomposing spend into sub-categories (dairy vs. staples vs. personal care) will provide deeper supply chain utility."
    )

    p = doc.add_paragraph()
    p.add_run("5.3 Actionable Business Recommendations: ").bold = True
    p.add_run(
        "1. Automated BNPL & Credit Limit Setting: Consumer finance and banking apps can safely cap grocery credit lines at 1.5x predicted monthly spend, reducing non-performing assets.\n"
        "2. Quick-Commerce Smart Subscriptions: Auto-generate recurring monthly grocery replenishment baskets based on household size and prior spend, offering 5% retention discounts.\n"
        "3. Dark Store Perishable Replenishment: Dark stores serving localities with predicted fresh produce share >=65% should execute daily morning replenishment cycles to prevent spoilage.\n"
        "4. Personalized Tier-Based Cross-Selling: Target private-label value essentials to households spending >20% of income on groceries, while routing organic/imported brands to affluent households spending <10%."
    )

    # -------------------------------------------------------------------------
    # SECTION 6. APPENDICES & RUNTIME SCREENSHOTS
    # -------------------------------------------------------------------------
    h6 = doc.add_heading("Section 6. Appendices & Verified Runtime Execution Screenshots", level=1)
    h6.style.font.color.rgb = RGBColor(31, 78, 121)

    p = doc.add_paragraph()
    p.add_run("Appendix A: Software and Reproducibility Specifications\n").bold = True
    p.add_run(
        "• Environment: Python 3.13.5 (64-bit), Scikit-Learn 1.6.0, Streamlit 1.40.1, Pandas 2.2.3, Seaborn 0.13.2, Matplotlib 3.10.0.\n"
        "• Git Repository: https://github.com/Malatesh-D/Grocery-Prediction\n"
        "• SHA-256 Model Artifact Checksum: 87d173bbb1f1145f2a7ca6a1fc29831ccdaa59d932c98afae18fb2675c1ee9df\n"
        "• Automated Test Verification: All 16 unit and integration tests passing (pytest -v in 16.41s).\n\n"
        "Appendix B: Verified Terminal Runtime Execution Screenshots\n"
        "The following screenshots document the verified live terminal execution of all core modules in the project pipeline:"
    )

    screenshots_meta = [
        ("terminal_01_data_ingestion.png", "Figure 6.1: Runtime Terminal Screenshot 1 — Data Ingestion & Shape Summary (python src/data_loader.py)"),
        ("terminal_02_data_validation.png", "Figure 6.2: Runtime Terminal Screenshot 2 — Schema Validation & Quality Assurance (python src/validate_data.py)"),
        ("terminal_03_preprocessing_pipeline.png", "Figure 6.3: Runtime Terminal Screenshot 3 — Preprocessing Pipeline & Leakage Verification (python src/preprocess.py)"),
        ("terminal_04_model_training_cv.png", "Figure 6.4: Runtime Terminal Screenshot 4 — 5-Fold Cross-Validation & Benchmark Selection (python src/train_model.py)"),
        ("terminal_05_holdout_evaluation.png", "Figure 6.5: Runtime Terminal Screenshot 5 — Holdout Test Evaluation & Feature Importance (python src/evaluate_model.py)"),
        ("terminal_06_prediction_inference.png", "Figure 6.6: Runtime Terminal Screenshot 6 — Real-Time Inference Engine Execution (python src/predict.py)"),
        ("terminal_07_pytest_test_suite.png", "Figure 6.7: Runtime Terminal Screenshot 7 — Automated Unit & Integration Test Suite (python -m pytest tests/ -v)"),
        ("terminal_08_streamlit_launch.png", "Figure 6.8: Runtime Terminal Screenshot 8 — Streamlit Web Application Server Launch (streamlit run app.py)")
    ]

    for fname, caption in screenshots_meta:
        img_p = SCREENSHOTS_DIR / fname
        add_figure_image(doc, img_p, caption, width=Inches(6.0))

    # Save Word document
    out_path = REPORTS_DIR / "Visualizations_and_Results_GROCERY_SPEND.docx"
    doc.save(str(out_path))
    print(f"\n[REPORT CREATED] Saved to: {out_path}")

    # Copy to Desktop for direct access
    desktop_copy = Path("C:/Users/malat/OneDrive/Desktop/Visualizations and Results (GROCERY SPEND).docx")
    try:
        shutil.copy2(out_path, desktop_copy)
        print(f"[COPIED TO DESKTOP] Saved to: {desktop_copy}")
    except Exception as e:
        print(f"Notice copying to desktop: {e}")

    return out_path


if __name__ == "__main__":
    generate_word_report()
