import pandas as pd

def drop_rare_labels(df, col, min_count):
    freq = df[col].value_counts()
    frequent_labels = freq[freq >= min_count].index
    return df[df[col].isin(frequent_labels)]

def load_and_filter_data(filename, min_counts):
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)

    for col, min_count in min_counts.items():
        print(f"Filtering {col} for a minimum count of {min_count}...")
        df = drop_rare_labels(df, col, min_count)
        print(f"{col} now has {df[col].nunique()} unique values.")

    return df

def main():
    filename = "processed_dataset.csv"
    min_counts = {
        'job_title': 200,
        'job_location': 200,
        'company': 200
    }
    
    df = load_and_filter_data(filename, min_counts)
    df.to_csv("further50k_processed_dataset.csv", index=False)
    print("Filtered dataset saved to further_processed_dataset.csv")

if __name__ == "__main__":
    main()
