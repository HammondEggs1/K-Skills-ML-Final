import pandas as pd
import argparse
import matplotlib.pyplot as plt

def get_csv_dimensions_and_count_other(filename):
    # Reads a CSV file, prints its dimensions, and returns frequency analysis of job titles.

    df = pd.read_csv(filename)
    title_counts = df['job_title'].value_counts()  
    other_count = title_counts.get('other', 0)  
    # Calculate the sum of samples for titles occurring 100 or fewer times
    low_frequency_sum = title_counts[title_counts <= 100].sum()  
    return df.shape, title_counts, other_count, low_frequency_sum

def plot_title_frequency_pie_chart(title_counts):
    # Plots a pie chart comparing the total occurrences of job titles that happen 100 times or fewer versus more than 100 times.
    less_equal_100 = title_counts[title_counts <= 100].sum()  # Sum of occurrences for titles ≤ 100 times
    greater_100 = title_counts[title_counts > 100].sum()  # Sum of occurrences for titles > 100 times
    
    data = {'≤ 100 Times': less_equal_100, '> 100 Times': greater_100}
    counts = pd.Series(data)
    plt.figure(figsize=(8, 8))
    counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'salmon'])
    plt.ylabel('')
    plt.title('Pie Chart of Job Title Frequencies (Total Occurrences ≤ 100 Times vs > 100 Times)')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Get dimensions, title frequency, and count of titles labeled "other" of a CSV file.')
    parser.add_argument('filename', type=str, help='Path to the CSV file')
    args = parser.parse_args()

    result = get_csv_dimensions_and_count_other(args.filename)

    dimensions, title_frequency, other_count, low_frequency_sum = result
    print("Dimensions of the CSV file (rows, columns):", dimensions)
    print("Frequency of job titles:")
    print(title_frequency)
    print(f'Number of titles labeled "other": {other_count}')
    print(f'Sum of samples for titles occurring 100 or fewer times: {low_frequency_sum}')
    plot_title_frequency_pie_chart(title_frequency)  

if __name__ == "__main__":
    main()
