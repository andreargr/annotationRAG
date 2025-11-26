import json  # use JSON data
import pandas as pd  # dataframe manipulation

# Dictionary to count valid patterns per column
pattern_counts = {}

def data_process(filename):
    """
    Load data from a JSON file and convert it to a structured DataFrame.

    Parameters:
        filename (str): Path to the JSON file containing class data.

    Returns:
        pd.DataFrame: Processed DataFrame with identifiers and labels.
    """
    with open(filename, 'r') as archive:
        dict_classes = json.load(archive)
    df = pd.DataFrame.from_dict(dict_classes, orient='index')
    new_row = pd.DataFrame([df.columns], columns=df.columns)
    df_process = pd.concat([new_row, df], ignore_index=True)
    df_process.columns = ['CLO_C', 'CLO_M', 'CL_C', 'CL_M', 'UBERON_C', 'UBERON_M', 'BTO_C', 'BTO_M']
    df_process = df_process.drop(0)

    # Add label column
    keys = list(dict_classes.keys())
    df_process['Label'] = keys

    # Replace missing values
    df_process.fillna("", inplace=True)

    return df_process

def search_common_substrings(string1, string2):
    """
    Find the longest common substring between two strings.

    Parameters:
        string1 (str): First string to compare.
        string2 (str): Second string to compare.

    Returns:
        str: Longest common substring found.
    """
    length = min(len(string1), len(string2))
    for i in range(length, 0, -1):
        for j in range(len(string1) - i + 1):
            if string1[j:j+i] in string2:
                return string1[j:j+i]
    return ""

def check_pattern(df, index, column, pattern, string1, string2, row):
    """
    Check if a common pattern between two strings meets criteria and update the DataFrame if valid.

    Also, count valid patterns per column.

    Parameters:
        df (pd.DataFrame): DataFrame to update.
        index (int): Row index.
        column (str): Column name to update.
        pattern (str): Common substring pattern.
        string1 (str): Original string.
        string2 (str): Modified string.
        row (pd.Series): Full row (for display purposes).

    Returns:
        pd.DataFrame: Updated DataFrame.
    """
    global pattern_counts
    if len(pattern) > 4 and pattern not in [' cell', ' of ']:
        print("For ontology:", column, "\n")
        print(row)
        print(index, 'The similarity between', string1, '-', string2, ':', pattern, len(pattern))
        check = input('Is the pattern valid? Y/N \n')
        if check == 'Y':
            df.at[index, column] = string1
            pattern_counts[column] = pattern_counts.get(column, 0) + 1
    return df

def df_to_dicc(df):
    """
    Convert a DataFrame to a dictionary using the 'Label' as keys and identifier columns as values.

    Parameters:
        df (pd.DataFrame): DataFrame to convert.

    Returns:
        dict: Dictionary mapping label to identifier values.
    """
    dicc = {}
    for index, row in df.iterrows():
        label = row['Label']
        identifiers = row[:8].tolist()
        dicc[label] = identifiers
    return dicc

def pattern_process(type):
    """
    Process each pair of columns depending on the type and validate common patterns.

    Parameters:
        type (str): Ontology type ('CL', 'CT', 'A').
    """
    if type == 'CL':
        df = data_process('k_decision/10K/classnames_CL.json')
        column_sets = [('CLO', 'BTO'), ('CL', 'UBERON')]
    elif type == 'CT':
        df = data_process('k_decision/10K/classnames_CT.json')
        column_sets = [('CL', 'BTO'), ('UBERON', None)]
    elif type == 'A':
        df = data_process('k_decision/10K/classnames_A.json')
        column_sets = [('UBERON', 'BTO')]
    else:
        raise ValueError("Unrecognized type")

    pattern_df = df

    # Iterate through column sets and apply pattern matching
    for col_1, col_2 in column_sets:
        control_1 = f'{col_1}_C'
        test_1 = f'{col_1}_M'

        for index, row in pattern_df.iterrows():
            string1 = row[control_1]
            string2 = row[test_1]
            if string1 != string2:
                pattern = search_common_substrings(string1, string2)
                pattern_df = check_pattern(pattern_df, index, test_1, pattern, string1, string2, row)

        if col_2:
            control_2 = f'{col_2}_C'
            test_2 = f'{col_2}_M'
            for index, row in pattern_df.iterrows():
                string1b = row[control_2]
                string2b = row[test_2]
                if string1b != string2b:
                    pattern = search_common_substrings(string1b, string2b)
                    pattern_df = check_pattern(pattern_df, index, test_2, pattern, string1b, string2b, row)

    # Save results to JSON
    dicc = df_to_dicc(pattern_df)
    file_name = f'pattern_file_{type}.json'
    with open(file_name, 'w') as archive_json:
        json.dump(dicc, archive_json, indent=4)

def main():
    #pattern_process('A')
    #pattern_process('CL')
    pattern_process('CT')

    print("\nSummary of valid patterns per column:")
    for col, cnt in pattern_counts.items():
        print(f"{col}: {cnt} valid patterns")

if __name__ == "__main__":
    main()
