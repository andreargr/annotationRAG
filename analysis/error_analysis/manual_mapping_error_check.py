import pandas as pd

def manual_error_classification(df, concept_type, ontology_prefix):
    """
    Allows user to manually classify each row for a given concept type and ontology.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing columns like CLO_C, CLO_M, CL_C, CL_M, etc.
    concept_type : str
        Concept type to filter (e.g., 'CL', 'CT', 'A').
    ontology_prefix : str
        Ontology prefix (e.g., 'CLO', 'CL', 'UBERON', 'BTO').

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with an extra column 'User_Classification'
        ('I', 'A', or 'No Error'), plus summary rows at the end.
    """

    # 🔹 Filter by concept type
    subset = df[df["Type"] == concept_type].copy()

    if subset.empty:
        raise ValueError(f"No rows found for concept type '{concept_type}'.")

    # 🔹 Identify ontology columns
    col_C = f"{ontology_prefix}_C"
    col_M = f"{ontology_prefix}_M"

    if col_C not in subset.columns or col_M not in subset.columns:
        raise ValueError(f"Columns {col_C} and/or {col_M} not found in dataset.")

    classifications = []

    print(f"\n--- Manual classification for concept type '{concept_type}' and ontology '{ontology_prefix}' ---")
    print("Enter 'I' for Interpretation Error, 'A' for Annotation (Mapping) Error, or press Enter to skip.\n")
    print("If both values are '-' or identical, they will be marked automatically as 'No Error'.\n")

    # 🔹 Iterate only over the filtered subset
    for idx, row in subset.iterrows():
        val_C = str(row[col_C]).strip()
        val_M = str(row[col_M]).strip()

        # Normalize missing or dash values
        if val_C in ["-", "", "nan", "None"]:
            val_C = "-"
        if val_M in ["-", "", "nan", "None"]:
            val_M = "-"

        # ✅ Automatic "No Error" cases
        if (val_C == "-" and val_M == "-") or (val_C == val_M):
            classifications.append("No Error")
            print(f"\nSample_ID: {row['Sample_ID']} — Auto 'No Error' ({val_C} vs {val_M})")
            continue

        # 🧠 Manual classification required
        print(f"\nSample_ID: {row['Sample_ID']}")
        print(f"Label: {row['Label']}")
        print(f"{col_C}: {val_C}")
        print(f"{col_M}: {val_M}")

        user_input = input("→ Classification (I/A): ").strip().upper()
        while user_input not in ["I", "A", ""]:
            user_input = input("Please enter 'I' (Interpretation) or 'A' (Annotation): ").strip().upper()

        classifications.append(user_input if user_input != "" else None)

    subset["User_Classification"] = classifications

    # === 📊 Summary block ===
    summary = (
        subset["User_Classification"]
        .value_counts(dropna=False)
        .rename_axis("Error_Type")
        .reset_index(name="Count")
    )

    summary_rows = pd.DataFrame({
        "Sample_ID": [""] * len(summary),
        "Label": [""] * len(summary),
        "Type": [""] * len(summary),
        col_C: [""] * len(summary),
        col_M: [""] * len(summary),
        "User_Classification": summary["Error_Type"] + ": " + summary["Count"].astype(str)
    })

    final_df = pd.concat([subset, summary_rows], ignore_index=True)
    return final_df[["Sample_ID", "Label", "Type", col_C, col_M, "User_Classification"]]


# === Example usage ===
if __name__ == "__main__":
    df = pd.read_csv("sample_92.csv")

    result = manual_error_classification(df, concept_type="A", ontology_prefix="BTO")

    output_file = "A_BTO_manual_classification.csv"
    result.to_csv(output_file, index=False)

    print(f"\n✅ Manual classification completed and saved to '{output_file}'")