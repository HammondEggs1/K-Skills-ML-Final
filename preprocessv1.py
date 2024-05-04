import argparse
from collections import Counter
import string
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import process, fuzz
from multiprocessing import Pool, cpu_count

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def drop_col(df, col_list):
    return df.drop(columns=col_list)

def drop_rare_labels(df, col, threshold=10):
    freq = df[col].value_counts()
    rare_labels = freq[freq < threshold].index
    df[col] = df[col].apply(lambda x: 'Other' if x in rare_labels else x)
    return df

def preprocess_text(text):
    if pd.isna(text):
        return ""  
    text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha()]
    return ' '.join(tokens)

def frequency_of_frequencies_with_labels(df, column):
    value_counts = df[column].value_counts()
    freq_of_freqs = value_counts.value_counts().sort_index()
    details = []
    for frequency, count in freq_of_freqs.items():
        labels = value_counts[value_counts == frequency].index.tolist()
        details.append({'Frequency': frequency, 'Count': count, 'Labels': labels})
    detailed_df = pd.DataFrame(details)
    detailed_df = detailed_df[['Frequency', 'Count', 'Labels']]
    return detailed_df

def fuzzy_match(args):
    title, top_titles, threshold = args
    best_match, score = process.extractOne(title, top_titles, scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        return title, best_match
    else:
        return title, title 



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("combined_data", help="filename for input dataset csv")
    args = parser.parse_args()

    col_to_drop = ['job_link', 'last_processed_time', 'first_seen', 'search_city', 'search_country', 'search_position', 'job_level', 'job_type', 'got_summary', 'got_ner', 'is_being_worked']

    print("Loading data...")
    df = pd.read_csv(args.combined_data, engine='python')
    
    print("Dropping NaNs...")
    df = df.dropna().reset_index(drop=True)
    
    print("Sampling data...")
    df_sample_50k = df.sample(n=50000, random_state=42024, replace=False)

    print("Dropping columns...")
    df_sample_50k = drop_col(df_sample_50k, col_to_drop)

    print("Performing fuzzy matching in parallel...")

    # Calculate frequency of each job title and select the top X percentile
    title_frequency = df_sample_50k['job_title'].value_counts()
    percentile = 0.9  # Adjust this value as needed
    cutoff = title_frequency.quantile(percentile)
    top_titles = title_frequency[title_frequency >= cutoff].index.tolist()

    # Perform fuzzy matching in parallel
    threshold = 80  # Set a threshold for minimum match quality

    # Set up multiprocessing pool
    num_processes = cpu_count()  
    with Pool(num_processes) as pool:
    # Map fuzzy_match function to each job title in the sample
        results = list(tqdm(pool.imap(fuzzy_match,
                        [(title, top_titles, threshold) for title in df_sample_50k['job_title']]), 
                        total=len(df_sample_50k), 
                        desc="Matching job titles"))

    # Collect results into dictionary
    matches = {title: match for title, match in results}

    print("Sample fuzzy matching results:", matches)
    
    # Preprocess and analyze columns
    for col in tqdm(['job_title', 'job_location', 'company'], desc="Processing columns"):
        df_sample_50k[col] = df_sample_50k[col].apply(preprocess_text)
        df_sample_50k = drop_rare_labels(df_sample_50k, col, threshold=2)
        analysis = frequency_of_frequencies_with_labels(df_sample_50k, col)
        print(f"Detailed analysis for {col}:\n", analysis, "\n")

        
    # Save the processed sample to CSV
    df_sample_50k.to_csv("50k_sample_processed_t2_p9.csv", index=False)
    
if __name__ == "__main__":
    main()
