#!/usr/bin/env python
"""
Debug script to analyze the UNSPSC selection process from semantic_matching.py
"""

import pandas as pd

import semantic_matching
from azure_search_utils import AzureSearchUtils
from config import Config
from llm import LLM
from utils import clean_description

# Mock YAML config for testing
CONFIG = {
    "SEARCH": {"similarity_threshold": 0.75, "top_results": 5, "max_results": 25},
    "WEIGHTS": {"mid_point": 0.55, "exp_factor": 10},
    "CONFIDENCE": {"single_match_factor": 0.95, "min_confidence_threshold": 50},
    "UNSPSC": {"level_thresholds": [90, 80, 70, 60], "generic_delta_percentage": 10},
}


def debug_unspsc_selection(description: str):
    """
    Debug the UNSPSC selection process step by step.

    Args:
        description: Item description to analyze
    """
    print("\n" + "=" * 80)
    print(f"DEBUGGING UNSPSC SELECTION PROCESS FOR: '{description}'")
    print("=" * 80)

    # Initialize required components
    config = Config()
    llm = LLM(config)
    search_utils = AzureSearchUtils(config)

    print("\n1. GETTING EMBEDDING FOR DESCRIPTION")
    print("-" * 80)
    embedding = semantic_matching._get_embedding(description, llm)
    print(f"✓ Generated embedding vector with {len(embedding)} dimensions")

    print("\n2. FETCHING SEARCH RESULTS")
    print("-" * 80)
    select_fields = ["ItemID", "DescriptionID", "ItemDescription", "UNSPSC", "MfrPartNum", "MfrName"]

    try:
        results = semantic_matching._fetch_distinct_items(
            azure_search_utils=search_utils,
            embedding=embedding,
            top_distinct_results=CONFIG["SEARCH"]["top_results"],
            similarity_threshold=CONFIG["SEARCH"]["similarity_threshold"],
            initial_fetch_size=CONFIG["SEARCH"]["max_results"],
            select_fields=select_fields,
        )

        if results.empty:
            print("No results found above the similarity threshold.")
            return

        print(f"✓ Found {len(results)} distinct results above threshold")

        # Display top 5 results
        print("\nTOP SEARCH RESULTS:")
        print("-" * 80)
        display_cols = ["ItemID", "ItemDescription", "UNSPSC", "MfrName", "@search.score"]
        pd.set_option("display.max_colwidth", 50)
        print(results[display_cols].head().to_string(index=True))

    except Exception as e:
        print(f"Error during search: {e}")
        return

    print("\n3. CALCULATING WEIGHTS FOR RESULTS")
    print("-" * 80)
    results_with_weights = semantic_matching._calculate_weights(
        results, CONFIG["WEIGHTS"]["mid_point"], CONFIG["WEIGHTS"]["exp_factor"]
    )

    # Display weights for the top results
    weight_display = results_with_weights[["ItemID", "@search.score", "weight"]].head()
    print(weight_display.to_string(index=True))
    print(f"\nTotal weight sum: {results_with_weights['weight'].sum():.4f}")

    print("\n4. GENERATING UNSPSC VARIANTS")
    print("-" * 80)
    # Filter out rows with NaN UNSPSC
    valid_results = results_with_weights.dropna(subset=["UNSPSC"])

    if valid_results.empty:
        print("No valid UNSPSC values found in results.")
        return

    unspsc_variants = semantic_matching._generate_unspsc_variants(valid_results)

    print(f"Generated {len(unspsc_variants)} UNSPSC variants:")
    for variant, level in unspsc_variants.items():
        level_name = {3: "Commodity", 2: "Class", 1: "Family"}[level]
        print(f"  {variant} (Level: {level} - {level_name})")

    print("\n5. CALCULATING CONFIDENCE SCORES FOR EACH VARIANT")
    print("-" * 80)
    variant_confidence = {}
    total_weight = results_with_weights["weight"].sum()

    for variant, code_level in unspsc_variants.items():
        matching_rows = semantic_matching._get_matching_rows_for_unspsc_variant(valid_results, variant, code_level)

        if not matching_rows.empty:
            variant_weight = matching_rows["weight"].sum()
            max_search_score = matching_rows["@search.score"].max()
            confidence = max_search_score * (variant_weight / total_weight) * 100
            variant_confidence[(variant, code_level)] = confidence

            # Print details
            level_name = {3: "Commodity", 2: "Class", 1: "Family"}[code_level]
            print(f"UNSPSC {variant} (Level: {level_name}):")
            print(f"  Matching rows: {len(matching_rows)}")
            print(f"  Max search score: {max_search_score:.4f}")
            print(f"  Weight sum: {variant_weight:.4f} ({(variant_weight / total_weight) * 100:.2f}% of total)")
            print(f"  Confidence score: {confidence:.2f}")
            print()

    print("\n6. SELECTING BEST UNSPSC VARIANT")
    print("-" * 80)
    best_unspsc = semantic_matching._select_best_unspsc_variant(
        variant_confidence, CONFIG["UNSPSC"]["level_thresholds"], CONFIG["UNSPSC"]["generic_delta_percentage"]
    )

    if not best_unspsc:
        print("No suitable UNSPSC variant found.")
        return

    unspsc_code, confidence_score = best_unspsc

    # Check if single match adjustment is needed
    is_single_match = len(results) == 1
    if is_single_match:
        adjusted_confidence = confidence_score * CONFIG["CONFIDENCE"]["single_match_factor"]
        print(f"Single match detected. Adjusting confidence: {confidence_score:.2f} → {adjusted_confidence:.2f}")
        confidence_score = adjusted_confidence

    print("\nSELECTED UNSPSC:")
    print("-" * 80)
    print(f"UNSPSC Code: {unspsc_code}")
    print(f"Confidence Score: {round(confidence_score)}")

    # Check against minimum threshold
    min_threshold = CONFIG["CONFIDENCE"]["min_confidence_threshold"]
    if confidence_score > min_threshold:
        print(f"✓ Confidence score exceeds minimum threshold ({min_threshold})")
    else:
        print(f"✗ Confidence score below minimum threshold ({min_threshold}), UNSPSC would be discarded")

    print("\n7. FINAL RESULT")
    print("-" * 80)
    if confidence_score > min_threshold:
        result = {"UNSPSC": unspsc_code, "UNSPSCConfidenceScore": round(confidence_score)}
        print(f"Final result: {result}")
    else:
        print("No UNSPSC would be returned due to low confidence.")


def _debug_variant_selection_logic(variant_confidence, level_thresholds, generic_delta_percentage):
    """Helper function to debug the variant selection logic"""
    print("\nDEBUGGING VARIANT SELECTION LOGIC:")
    print("-" * 80)

    # Sort variants by hierarchy level and confidence
    sorted_variants = sorted(
        variant_confidence.items(), key=lambda x: (-x[0][1], -x[1])  # Sort by -level (descending), then -confidence (descending)
    )

    print("Sorted variants (by level, then confidence):")
    for (variant, level), confidence in sorted_variants:
        level_name = {3: "Commodity", 2: "Class", 1: "Family"}[level]
        print(f"  {variant} (Level: {level_name}): {confidence:.2f}")

    # Group by level
    variants_by_level = {level: [] for level in [1, 2, 3]}
    for (variant, level), confidence in variant_confidence.items():
        variants_by_level[level].append((variant, confidence))

    # Best variants by level
    print("\nBest variant per level:")
    for level, variants in variants_by_level.items():
        if variants:
            level_name = {3: "Commodity", 2: "Class", 1: "Family"}[level]
            best = max(variants, key=lambda x: x[1])
            print(f"  Best {level_name}: {best[0]} ({best[1]:.2f})")

    # Check thresholds
    print("\nChecking against thresholds:")
    thresholds = [
        level_thresholds.get("threshold_90", 90),
        level_thresholds.get("threshold_80", 80),
        level_thresholds.get("threshold_70", 70),
        level_thresholds.get("threshold_60", 60),
    ]

    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")

        # Check Commodity level (level 3)
        commodity_variants = [(v, c) for ((v, l), c) in variant_confidence.items() if l == 3 and c >= threshold]
        if commodity_variants:
            best_commodity = max(commodity_variants, key=lambda x: x[1])
            print(f"  ✓ Found commodity variant above threshold: {best_commodity[0]} ({best_commodity[1]:.2f})")
            print(f"  → This would be selected at threshold {threshold}")
            break
        else:
            print(f"  ✗ No commodity variants above threshold {threshold}")

        # Check Class level (level 2)
        class_variants = [(v, c) for ((v, l), c) in variant_confidence.items() if l == 2 and c >= threshold]
        if class_variants:
            best_class = max(class_variants, key=lambda x: x[1])
            print(f"  ✓ Found class variant above threshold: {best_class[0]} ({best_class[1]:.2f})")

            # Compare with best commodity
            best_commodity = max(
                [(v, c) for ((v, l), c) in variant_confidence.items() if l == 3], key=lambda x: x[1], default=(None, 0)
            )

            if best_commodity[0]:
                delta = best_class[1] - best_commodity[1]
                threshold_delta = best_commodity[1] * generic_delta_percentage / 100
                print(f"  Comparing with best commodity variant: {best_commodity[0]} ({best_commodity[1]:.2f})")
                print(f"  Delta: {delta:.2f}, Required delta: {threshold_delta:.2f}")

                if delta < threshold_delta:
                    print(f"  → Commodity variant preferred: {best_commodity[0]} ({best_commodity[1]:.2f})")
                    print(f"  → This would be selected at threshold {threshold}")
                else:
                    print(f"  → Class variant significantly better, selecting: {best_class[0]} ({best_class[1]:.2f})")
                    print(f"  → This would be selected at threshold {threshold}")
            else:
                print(f"  → Class variant selected: {best_class[0]} ({best_class[1]:.2f})")
                print(f"  → This would be selected at threshold {threshold}")
            break
        else:
            print(f"  ✗ No class variants above threshold {threshold}")

        # Check Family level (level 1)
        family_variants = [(v, c) for ((v, l), c) in variant_confidence.items() if l == 1 and c >= threshold]
        if family_variants:
            best_family = max(family_variants, key=lambda x: x[1])
            print(f"  ✓ Found family variant above threshold: {best_family[0]} ({best_family[1]:.2f})")

            # Compare with best higher level
            best_higher_level = max(
                [(v, c) for ((v, l), c) in variant_confidence.items() if l > 1], key=lambda x: x[1], default=(None, 0)
            )

            if best_higher_level[0]:
                delta = best_family[1] - best_higher_level[1]
                threshold_delta = best_higher_level[1] * generic_delta_percentage / 100
                print(f"  Comparing with best higher level variant: {best_higher_level[0]} ({best_higher_level[1]:.2f})")
                print(f"  Delta: {delta:.2f}, Required delta: {threshold_delta:.2f}")

                if delta < threshold_delta:
                    print(f"  → Higher level variant preferred: {best_higher_level[0]} ({best_higher_level[1]:.2f})")
                    print(f"  → This would be selected at threshold {threshold}")
                else:
                    print(f"  → Family variant significantly better, selecting: {best_family[0]} ({best_family[1]:.2f})")
                    print(f"  → This would be selected at threshold {threshold}")
            else:
                print(f"  → Family variant selected: {best_family[0]} ({best_family[1]:.2f})")
                print(f"  → This would be selected at threshold {threshold}")
            break
        else:
            print(f"  ✗ No family variants above threshold {threshold}")

    # If no variant meets the thresholds
    if all(len([(v, c) for ((v, l), c) in variant_confidence.items() if c >= t]) == 0 for t in thresholds):
        best_variant = sorted_variants[0][0]  # (variant, level)
        best_score = sorted_variants[0][1]  # confidence score
        print("\n✗ No variants meet any threshold. Using variant with highest confidence:")
        print(f"  → {best_variant[0]} (Level: {best_variant[1]}, Score: {best_score:.2f})")


if __name__ == "__main__":
    # Item description to analyze
    item_description = "OBIT - 4AR1G-58 1G 4S ADJ RING 5/8IN-1-1/4IN D"

    # Run the debug analysis
    debug_unspsc_selection(clean_description(item_description))
