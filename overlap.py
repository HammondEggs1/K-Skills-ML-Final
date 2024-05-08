import pandas as pd
import argparse
import matplotlib.pyplot as plt

def get_csv_dimensions_and_counts(filename):
    # Reads a CSV file and returns frequency analysis based on multiple criteria.
    df = pd.read_csv(filename)
    # Value counts for job_title, job_location, and company
    title_counts = df['job_title'].value_counts()
    location_counts = df['job_location'].value_counts()
    company_counts = df['company'].value_counts()

    # Filter the main DataFrame to entries where all criteria are met for having more than 100 occurrences
    filtered_df = df[
        (df['job_title'].map(title_counts) > 100) & 
        (df['job_location'].map(location_counts) > 100) & 
        (df['company'].map(company_counts) > 100)
    ]

    # Sum occurrences of filtered entries
    filtered_sum = filtered_df.shape[0]
    total_sum = df.shape[0]
    not_filtered_sum = total_sum - filtered_sum

    return df.shape, filtered_sum, not_filtered_sum
    

def plot_filtered_data_pie_chart(filtered_sum, not_filtered_sum):
    # Plots a pie chart of entries meeting vs not meeting the criteria.
    data = {'Meets Criteria (>100 occurrences)': filtered_sum, 'Does Not Meet Criteria': not_filtered_sum}
    counts = pd.Series(data)
    plt.figure(figsize=(8, 8))
    counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'orange'])
    plt.ylabel('')
    plt.title('Pie Chart of Data Entries Meeting Specified Criteria (>100 Occurrences Each) vs Not Meeting')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze CSV data')
    parser.add_argument('filename', type=str, help='Path to the CSV file')
    args = parser.parse_args()

    dimensions, filtered_sum, not_filtered_sum = get_csv_dimensions_and_counts(args.filename)
    print("Dimensions of the CSV file (rows, columns):", dimensions)
    print(f"Number of entries meeting all criteria (>100 occurrences each): {filtered_sum}")
    print(f"Number of entries not meeting all criteria: {not_filtered_sum}")
    plot_filtered_data_pie_chart(filtered_sum, not_filtered_sum)

if __name__ == "__main__":
    main()
