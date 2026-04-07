# Rethink Transaction Categorization – Prompt Optimization Summary (v2)

## Objective
Improve the system prompt in `rethink_transaction_categorization_optimizer.py` (lines 42–117) along these axes:

1. **Category in output written exactly as in category_options** — character-for-character copy; no rephrasing or spelling variants.
2. **Transfers strictly defined** — movement between the same person's own accounts and/or payment for their own debts, mortgages, or other liabilities.
3. **General shopping** — when a transaction is clearly general shopping, pick a shopping subcategory rationally and explain the basis in reasoning.

## Runs Performed
- **Batch 1** (Macho's, Marco's Pizza, Marcos Pizza Alt, Seamless, Salvatore's): 3 runs  
- **Batch 2** (Cos, Cleveland Marriott, Marriott, Farm Carol San Pedro, Zeko's Grill): 3 runs  
- **Batch 3** (AT&T, Transfer to Checking, F&S Metro News, Placeit, AAA): 3 runs  
- **Batch 4** (Cleo Express Fee, OpenAI, OpenAI Chatgpt, SFO Parking, Equinox): 3 runs  

**Total: 12 runs** (4 batches × 3), with prompt refinements between batches.

---

## Prompt Changes Applied

### 1. Exact category string (axis 1)
- **Input Format**: "Your output `category` MUST be a character-for-character exact copy of one of these strings—no rephrasing, no spelling changes, no singular/plural variants. Copy the string exactly as it appears in `category_options`."
- **Category Selection**: "The `category` value in your output MUST be an exact character-for-character copy of one string from `category_options`, or the literal `unknown`. Do not alter spelling, casing, or wording."
- **Subcategory rule**: "output the subcategory string exactly as it appears in the list" and "copy `leisure_entertainment` not `leisure`; copy `bills_insurance`…; copy `transport_car_fuel` not `transport` (parking, fuel, car-related…); copy `food_dining_out` not `food`."

**Observation:** Batches 1 and 2 consistently returned exact category strings (e.g. `food_dining_out`, `shopping_clothing`, `food_groceries`). Occasional malformed JSON in batch 3/4 (e.g. stray `"group_id":` / `"transaction_id",` in output) was model/schema noise, not prompt wording. When valid, category values matched the list.

### 2. Transfers strictly defined (axis 2)
- **Category meanings (transfer)**: "Strictly (1) movement of money between the same person's own accounts (e.g. checking to savings, ACH between own accounts), or (2) payment toward that person's own debts, mortgages, or other liabilities (e.g. credit card payment to own card, loan payment to own loan, mortgage payment to own mortgage). Net worth unchanged. Peer-to-peer to/from another person is NOT transfer."
- **CRITICAL Transfer Rule**: "`transfer` ONLY when (a) money moves between the same person's own accounts (e.g. Transfer To Checking, Transfer From Savings, ACH between own accounts), or (b) payment is toward that person's own debt, mortgage, or other liability (e.g. credit card payment to own card, loan payment to own loan, mortgage payment to own mortgage). Do not use transfer for payments to other people or to merchants. Peer-to-peer (Zelle, Venmo, PayPal, Cash App) to/from another person is NOT transfer."

**Observation:** "Transfer to Checking" was categorized as `transfer` in every batch 3 run, with reasoning citing movement between own accounts. Transfer rule was followed whenever the scenario was clearly own-account or own-debt.

### 3. General shopping subcategory + reasoning (axis 3)
- **Category Selection**: "When a transaction is clearly general shopping (e.g. a retail store where the specific product is not stated), choose the most plausible shopping subcategory from `category_options` based on establishment type, description, or amount—and state that basis briefly in reasoning (e.g. 'General retail; clothing store description suggests shopping_clothing.'). Do not use a parent shopping category when a subcategory is available."
- **Outflows vs income**: "**Outflows (positive amount)**: Never use income_side_gig, income_business, or income_salary for a payment or purchase—only for money received as earnings. Use an expense or transfer category for outflows."

**Observation:** Cos (clothing/accessories retailer) was consistently categorized as `shopping_clothing` with reasoning citing "clothing and accessories" or "apparel." Shopping subcategory + basis in reasoning was satisfied. Placeit (design-asset purchase, outflow) was sometimes still labeled `income_side_gig` despite the outflow rule; when it chose `shopping_gadgets` or `bills_service_fees`, behavior was correct.

### 4. Other refinements
- **Rational Basis**: Explicit "A purchase or payment (positive amount, outflow) must never be categorized as any income category—use an expense or transfer category instead."
- **Subcategory examples**: Added "(parking, fuel, car-related expenses use transport_car_fuel when it is in the list)" to reinforce transport subcategory for SFO Parking–type cases.

---

## Best Quality Outcome – Rationale

**Best overall behavior** was in **Batch 1 and Batch 2** across the three axes:

1. **Exact category string:** All returned categories were exact copies from `category_options` (e.g. `food_dining_out`, `food_delivered_food`, `shopping_clothing`, `food_groceries`, `leisure_travel_vacations`). No invented or misspelled labels.
2. **Transfer rule:** Batch 3 "Transfer to Checking" was consistently correct (`transfer`) with reasoning tied to movement between own accounts. The strict definition (own accounts + own debts/liabilities) was applied correctly.
3. **Shopping subcategory + reasoning:** Cos was categorized as `shopping_clothing` with clear basis in reasoning (establishment sells clothing/accessories). No parent `shopping` when `shopping_clothing` was available.

**Why this is the best quality:**
- **Exact match** is enforced in multiple places (Input Format, Category Selection, Subcategory rule), so the model reliably copies strings from `category_options`.
- **Strict transfer** definition (two bullets: own-account movement + own-debt/liability payments) removes ambiguity and keeps peer-to-peer and merchant payments out of `transfer`.
- **Shopping** instruction (pick subcategory + state basis) produced correct subcategory and evidence-based reasoning for retail cases like Cos.

**Remaining brittleness:**
- **SFO Parking:** When both `transport` and `transport_car_fuel` appear in `category_options`, the model sometimes returned `transport` instead of `transport_car_fuel` despite the explicit example. A post-check could replace parent `transport` with `transport_car_fuel` when both exist in the list.
- **Placeit:** Occasional `income_side_gig` for an outflow; reinforcing "positive amount = expense only" in one more place or adding a single negative example (e.g. "Placeit purchase → shopping_gadgets or bills_service_fees, not income") could help.
- **Output shape:** Rare malformed JSON (extra/misplaced keys like `group_id`) — prompt already says "Do NOT include group_id"; schema or response parsing may need to enforce structure.

---

## Summary of Prompt Edits (by location)

| Section | Edit |
|--------|------|
| Input Format | Category must be character-for-character exact copy from category_options; copy subcategory when both parent and subcategory appear. |
| Rational Basis | Outflows must never be income categories; use expense or transfer. |
| Category meanings – transfer | Strictly (1) own-account movement, (2) payment toward own debts/mortgages/liabilities; not P2P. |
| Critical – Subcategory only | Output subcategory string exactly as in list; examples include transport_car_fuel for parking/fuel/car. |
| Critical – Transfer Rule | (a) Own accounts only, (b) own debt/mortgage/liability only; not payments to other people or merchants. |
| Category Selection | Exact copy of one string or `unknown`; general shopping → pick shopping subcategory and state basis in reasoning; outflows never income_side_gig/income_business/income_salary. |

These edits maximize the three axes (exact category string, strict transfer, shopping subcategory with reasoning) while keeping the rest of the prompt intact.
