import pandas as pd
import argparse
import matplotlib.pyplot as plt

def get_csv_dimensions_and_count_other(filename):
    # Reads a CSV file, prints its dimensions, and returns frequency analysis of company occurrences.
    
    df = pd.read_csv(filename)
    company_counts = df['company'].value_counts()  
    other_count = company_counts.get('other', 0)  
    # Calculate the sum of samples for companies occurring 100 or fewer times
    low_frequency_sum = company_counts[company_counts <= 100].sum()
    return df.shape, company_counts, other_count, low_frequency_sum
    

def plot_company_frequency_pie_chart(company_counts):
    # Plots a pie chart comparing the total occurrences of companies that happen 100 times or fewer versus more than 100 times.
    less_equal_100 = company_counts[company_counts <= 100].sum()  # Sum of occurrences for companies ≤ 100 times
    greater_100 = company_counts[company_counts > 100].sum()  # Sum of occurrences for companies > 100 times

    data = {'≤ 100 Times': less_equal_100, '> 100 Times': greater_100}
    counts = pd.Series(data)
    plt.figure(figsize=(8, 8))
    counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'salmon'])
    plt.ylabel('')
    plt.title('Pie Chart of Company Frequencies (Total Occurrences ≤ 100 Times vs > 100 Times)')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Get dimensions, company frequency, and count of companies labeled "other" of a CSV file.')
    parser.add_argument('filename', type=str, help='Path to the CSV file')
    args = parser.parse_args()

    result = get_csv_dimensions_and_count_other(args.filename)  

    dimensions, company_frequency, other_count, low_frequency_sum = result
    print("Dimensions of the CSV file (rows, columns):", dimensions)
    print("Frequency of companies:")
    print(company_frequency)
    print(f'Number of companies labeled "other": {other_count}')
    print(f'Sum of samples for companies occurring 100 or fewer times: {low_frequency_sum}')
    plot_company_frequency_pie_chart(company_frequency)  

if __name__ == "__main__":
    main()
