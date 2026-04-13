# Mask Optimization Summary - Strategic Mask Inclusion

## Objective
Optimize the prompt to ensure masked account numbers are ONLY included when the preliminary name is NOT unique (appears 2+ times). In mixed scenarios, some accounts should have masks (duplicates) while others should not (unique names).

## Key Optimization Axis
**Strategic Mask Inclusion**: 
- Masked account numbers should ONLY be included if the account's preliminary name is NOT unique (appears 2+ times in the complete list)
- If a preliminary name is unique (appears only once), the mask is completely unnecessary and MUST be omitted
- Mixed scenarios are common: some accounts have masks (duplicates), some don't (unique)

## Optimization Journey (15 Runs: 5 Batches × 3 Iterations)

### Iteration 1 (Baseline)
**Initial Prompt**: Step 2 had basic duplicate detection and mask addition logic.

**Key Observations**:
- Batch 4 (duplicate_detection_and_masking): Correctly handled mixed scenario
  - "Chase Total Checking **4227" and "Chase Total Checking **8493" - masks added (duplicates) ✓
  - "Alliant Cashback" - no mask (unique) ✓
  - "Wells Fargo Checking" - no mask (unique) ✓
  - "Wells Fargo Savings **2460" and "Wells Fargo Savings **8123" - masks added (duplicates) ✓
- Behavior was correct but prompt could be more explicit about the logic

### Iteration 2 (Enhanced Clarity)
**Changes Made**:
1. Made duplicate detection more explicit with frequency counting
2. Added explicit examples showing unique vs duplicate scenarios
3. Emphasized that masks are ONLY for non-unique names
4. Added CRITICAL note about mixed scenarios

**Results**:
- ✅ All batches continued to show correct behavior
- ✅ Mixed scenarios handled correctly
- Prompt more explicit but could be even clearer with step-by-step process

### Iteration 3 (Final Refinement)
**Changes Made**:
1. Added **CRITICAL RULE** at the top of Step 2 for immediate visibility
2. Restructured into clear sub-steps (9a, 9b, 9c):
   - Step 9a: Check uniqueness for each account
   - Step 9b: Add masks to non-unique names (duplicates)
   - Step 9c: Omit masks for unique names
3. Added concrete mixed scenario example showing 6 accounts with different patterns
4. Made the logic flow explicit: count → check → apply mask or omit

**Results**:
- ✅ Batch 4: Perfect mixed scenario handling
  - Duplicates get masks: "Chase Total Checking **4227", "Wells Fargo Savings **2460"
  - Unique names have no masks: "Alliant Cashback", "Wells Fargo Checking"
- ✅ All other batches: Correctly omit masks when all names are unique
- ✅ Prompt now has crystal-clear logic flow

## Final Optimized Prompt Features

### 1. Critical Rule at Top
- **CRITICAL RULE** stated upfront: Masks ONLY if preliminary name is NOT unique
- If unique → mask is unnecessary and MUST be omitted

### 2. Explicit Frequency Counting
- Step 8: Count occurrences of each preliminary name
- Clear examples: "appears 1 time → UNIQUE → NO MASK"
- Clear examples: "appears 2 times → NOT UNIQUE → ADD MASK"

### 3. Step-by-Step Process
- **Step 9a**: Check uniqueness for each account (count occurrences)
- **Step 9b**: For non-unique names → add mask to ALL instances
- **Step 9c**: For unique names → output without mask

### 4. Mixed Scenario Example
- Concrete example with 6 accounts showing:
  - 2 duplicates with masks
  - 1 unique without mask
  - 1 unique without mask
  - 2 duplicates with masks

## Best Quality Outcome Rationalization

### Why Iteration 3 is Optimal:

1. **Crystal-Clear Logic Flow**
   - Critical rule stated at top for immediate understanding
   - Step-by-step process (9a → 9b → 9c) makes execution unambiguous
   - Each account evaluated individually for uniqueness

2. **Perfect Mixed Scenario Handling**
   - Batch 4 demonstrates perfect behavior:
     * "Chase Total Checking" (2 occurrences) → Both get masks ✓
     * "Alliant Cashback" (1 occurrence) → No mask ✓
     * "Wells Fargo Checking" (1 occurrence) → No mask ✓
     * "Wells Fargo Savings" (2 occurrences) → Both get masks ✓
   - This is exactly the mixed scenario the user requested

3. **Prevents Common Errors**
   - Explicit "SKIP Step 9b" for unique names prevents accidental mask addition
   - Concrete examples prevent confusion about what constitutes a duplicate
   - Frequency counting makes the logic objective and unambiguous

4. **Strategic Information Inclusion**
   - Masks only added when necessary (non-unique names)
   - Masks omitted when unnecessary (unique names)
   - Maximizes conciseness while maintaining distinguishability

### Quantitative Analysis:

**Batch 4 (Mixed Scenario - 6 accounts)**:
- 2 accounts with "Chase Total Checking" (duplicate) → Masks added: 2 masks
- 1 account with "Alliant Cashback" (unique) → No mask: 0 masks
- 1 account with "Wells Fargo Checking" (unique) → No mask: 0 masks
- 2 accounts with "Wells Fargo Savings" (duplicate) → Masks added: 2 masks
- **Result**: 4 masks total (only on duplicates), 2 accounts without masks (unique names)
- **Efficiency**: 33% of accounts don't need masks (unique names)

**Batches 1, 2, 3, 5 (All Unique Names)**:
- All accounts have unique preliminary names
- **Result**: 0 masks added (correct behavior)
- **Efficiency**: 100% of accounts correctly omit unnecessary masks

## Key Improvements Over Baseline

1. **Explicit Uniqueness Check**: Each account's preliminary name is evaluated for frequency
2. **Conditional Mask Application**: Masks only applied in Step 9b (for non-unique), skipped in Step 9c (for unique)
3. **Concrete Examples**: Mixed scenario example shows exactly how to handle 6 accounts with different patterns
4. **Critical Rule Visibility**: Rule stated at top of Step 2 for immediate reference

## Conclusion

The final optimized prompt (Iteration 3) successfully achieves the goal of **strategic mask inclusion**:
- ✅ Masks ONLY added to accounts with non-unique preliminary names (duplicates)
- ✅ Masks completely omitted from accounts with unique preliminary names
- ✅ Mixed scenarios handled perfectly (some with masks, some without)
- ✅ Logic flow is crystal-clear with step-by-step process
- ✅ Prevents common errors through explicit instructions and examples

The prompt now makes intelligent, context-aware decisions about when masks are necessary, resulting in the most concise account names while maintaining full distinguishability.
