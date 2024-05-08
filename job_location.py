import pandas as pd
import argparse
import matplotlib.pyplot as plt

def get_csv_dimensions_and_count_other(filename):
    # Reads a CSV file, prints its dimensions, and returns frequency analysis of job locations.

    df = pd.read_csv(filename)
    location_counts = df['job_location'].value_counts()  
    other_count = location_counts.get('other', 0)  
    # Calculate the sum of samples for locations occurring 100 or fewer times
    low_frequency_sum = location_counts[location_counts <= 100].sum()
    return df.shape, location_counts, other_count, low_frequency_sum
 

def plot_location_frequency_pie_chart(location_counts):
    # Plots a pie chart comparing the total occurrences of job locations that happen 100 times or fewer versus more than 100 times.
    less_equal_100 = location_counts[location_counts <= 100].sum()  # Sum of occurrences for locations ≤ 100 times
    greater_100 = location_counts[location_counts > 100].sum()  # Sum of occurrences for locations > 100 times

    data = {'≤ 100 Times': less_equal_100, '> 100 Times': greater_100}
    counts = pd.Series(data)
    plt.figure(figsize=(8, 8))
    counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'salmon'])
    plt.ylabel('')
    plt.title('Pie Chart of Job Location Frequencies (Total Occurrences ≤ 100 Times vs > 100 Times)')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Get dimensions, location frequency, and count of locations labeled "other" of a CSV file.')
    parser.add_argument('filename', type=str, help='Path to the CSV file')
    args = parser.parse_args()

    result = get_csv_dimensions_and_count_other(args.filename)
    
    dimensions, location_frequency, other_count, low_frequency_sum = result
    print("Dimensions of the CSV file (rows, columns):", dimensions)
    print("Frequency of job locations:")
    print(location_frequency)
    print(f'Number of locations labeled "other": {other_count}')
    print(f'Sum of samples for locations occurring 100 or fewer times: {low_frequency_sum}')
    plot_location_frequency_pie_chart(location_frequency)  

if __name__ == "__main__":
    main()
