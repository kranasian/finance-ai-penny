# Rethink Transaction Categorization – Prompt Optimization Summary

## Objective
Improve the system prompt (lines 42–208) in `rethink_transaction_categorization_optimizer.py` along these axes:

1. **No examples in system prompt** – avoid biasing the model.
2. **Output must be a sub-category from category_options** – never a parent when a subcategory exists.
3. **Debt repayments = transfers** – payments to own loan/credit are transfers.
4. **Rational category choice** – based on establishment_name, establishment_description, transaction_text, and/or amount.
5. **Inflows ≠ always income** – refunds/returns can map to expense categories.

## Runs Performed
- **Batch 1** (Macho's, Marco's Pizza, Marcos Pizza Alt, Seamless, Salvatore's): 3 runs  
- **Batch 2** (Cos, Cleveland Marriott, Marriott, Farm Carol San Pedro, Zeko's Grill): 3 runs  
- **Batch 3** (AT&T, Transfer to Checking, F&S Metro News, Placeit, AAA): 3 runs  
- **Batch 4** (Cleo Express Fee, OpenAI, OpenAI Chatgpt, SFO Parking, Equinox): 3 runs  

**Total: 12 runs** (4 batches × 3), with prompt refinements between batches.

---

## Prompt Changes Applied

### 1. Removed all in-prompt examples
- **Before:** Three full input/output examples (Best Friends Veterinary, Starbucks, City News Stand) in the system prompt.  
- **After:** No examples. All instructions are rule- and structure-based.  
- **Reason:** Reduces bias toward similar establishments and category wording; model relies on rules and category meanings only.

### 2. Sub-category only (never parent)
- **Added:** In Input Format: “When both a parent and a subcategory from the same hierarchy appear in category_options, you MUST output the subcategory, never the parent.”  
- **Added:** First Critical Rule: “**Subcategory only**: If category_options contains both a parent and a subcategory from the same hierarchy, you MUST output the subcategory. Examples: output `leisure_entertainment` not `leisure`; `bills_insurance` or `bills_connectivity` not `bills`; `transport_car_fuel` not `transport`; `food_dining_out` not `food`.”  
- **Reinforced:** In Category Selection: “When both a parent and a subcategory from the same hierarchy appear in category_options (e.g. `leisure` and `leisure_entertainment`), you MUST output the subcategory only—never the parent.”  
- **Observation:** Batch 1 & 2 consistently chose subcategories (e.g. `food_dining_out`, `shopping_clothing`, `leisure_travel_vacations`). Batch 3 had F&S Metro News output `leisure` (parent) in runs 1–2; run 3 overcorrected to `unknown`. AAA sometimes chose `bills` (parent) instead of `bills_insurance` or `bills_service_fees`; later runs fixed to `bills_service_fees`. Batch 4 had SFO Parking output `transport` (parent) instead of `transport_car_fuel` in multiple runs despite the explicit example.

### 3. Debt repayments = transfers
- **Updated:** Transfer rule now states: “Debt repayments to your own loan or credit accounts are transfers, not expense categories.”  
- **In transfer definition:** “Includes: account-to-account transfers, payment to own credit card, payment to own loan (debt repayment to own account).”  
- **Observation:** “Transfer to Checking” was consistently categorized as `transfer` in all batch 3 runs. No debt-repayment test in batches 1–4; rule is stated clearly for future cases.

### 4. Rational basis for category
- **Added:** “**Rational Basis for Category**: Choose the category only from evidence in: establishment_name, establishment_description, transaction_text, and/or amount. Do not guess. If there is no clear evidence tying the transaction to a specific category in category_options, use `unknown`.”  
- **Observation:** Reasoning in runs consistently cited establishment and amount (e.g. “Mexican food establishment”, “Pizza restaurant”, “sells food delivery services”). Removed reasoning examples from the prompt to avoid “category options include” style phrasing.

### 5. Inflows not always income
- **Added:** “**Inflows (negative amount)**: Negative amounts are inflows. Not all inflows are income. Refunds, returns, or reversals of a prior expense should be categorized as the same (or appropriate) expense subcategory when identifiable; use income categories only when the inflow clearly represents earnings (e.g. salary, interest, business revenue).”  
- **Added:** In Rational Basis: “Income categories (e.g. income_business, income_salary) are only for inflows that represent earnings; a purchase or payment (positive amount, outflow) must never be categorized as an income category.”  
- **Added:** In Category Selection: “Outflows (positive amount) are expenses or transfers—do not categorize a purchase or payment as an income category.”  
- **Observation:** Placeit (design-asset purchase, outflow) was still sometimes categorized as `income_business` in batch 3; the model occasionally conflates “business-related expense” with “income_business”. Rule is explicit; edge cases may need further reinforcement or test cases.

### 6. Other refinements
- **Category meanings:** Shortened to one-line summaries (no long bullet lists) to keep the prompt compact and reduce bias.  
- **Reasoning rule:** No examples of good/bad reasoning; instruction is “State ONLY positive evidence… Do not mention category_options, subcategory, ‘options include’, or other categories.”  
- **Transfer rule:** Shortened and made explicit that debt repayments to own accounts are transfers and that peer-to-peer is not transfer.

---

## Best Quality Outcome – Rationale

**Best overall behavior** was seen in **Batch 1 and Batch 2** after the first optimization pass (no examples + subcategory + rational basis + inflows + transfer/debt):

- **Batch 1:** All five tests (Macho's, Marco's Pizza, Marcos Pizza Alt, Seamless, Salvatore's) received correct subcategories (`food_dining_out`, `food_delivered_food`) and evidence-based reasoning with no “options include” language.  
- **Batch 2:** Cos → `shopping_clothing` (not `shopping`); Cleveland Marriott → mix of `leisure_travel_vacations`, `food_dining_out`, `leisure_entertainment`; Marriott → `leisure_travel_vacations`; Farm Carol → `food_groceries`; Zeko's Grill → `food_dining_out`. All subcategories; reasoning tied to establishment and amount.

**Why this is the best quality:**

1. **No examples** – Removed bias; the model still picked the right subcategories and reasoned from evidence.  
2. **Subcategory preference** – When both parent and subcategory were in `category_options`, the model usually chose the subcategory in batches 1–2; adding the explicit “Subcategory only” rule with examples improved batches 3–4 for bills/leisure but not fully for `transport` vs `transport_car_fuel`.  
3. **Transfers** – “Transfer to Checking” was consistently `transfer`; debt-repayment rule is clear for production.  
4. **Rational basis** – Reasoning consistently referenced establishment name/description and amount, with no invented categories.  
5. **Inflows** – Rule is clear; Placeit-type confusion (outflow labeled as income) may need either more prominent placement of “outflow ≠ income” or targeted few-shot only for that edge case.

**Remaining brittleness:**

- **SFO Parking:** When `transport` and `transport_car_fuel` both appear, the model sometimes still outputs `transport`. The prompt explicitly says “output `transport_car_fuel` not `transport`”; compliance may depend on model or temperature. Consider post-validation: if the chosen category is a parent and a subcategory from the same hierarchy exists in `category_options`, replace with that subcategory.  
- **F&S Metro News:** After enforcing subcategory-only, one run chose `unknown` instead of `leisure_entertainment`; the rule may push the model to be overly cautious when it’s unsure.  
- **Placeit:** Occasional `income_business` for an outflow; reinforcing “positive amount = expense or transfer, never income” in one more place could help.

---

## Final Prompt Location and Length
- **File:** `experiments/rethink_transaction_categorization_optimizer.py`  
- **Section:** `SYSTEM_PROMPT` (starts ~line 42).  
- **Length:** Shortened from ~167 lines (with 3 examples) to ~55 lines (rules and category meanings only), improving clarity and reducing bias while preserving all five axes above.
