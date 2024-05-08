import argparse
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

def drop_rare_labels(df, col, min_count):
    freq = df[col].value_counts()
    frequent_labels = freq[freq >= min_count].index
    return df[df[col].isin(frequent_labels)]

def preprocess_text(text):
    if pd.isna(text):
        return ""  
    text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha()]
    return ' '.join(tokens)

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

    print("Dropping columns...")
    df = drop_col(df, col_to_drop)

    print("Performing fuzzy matching in parallel...")
    title_frequency = df['job_title'].value_counts()
    percentile = 0.995
    cutoff = title_frequency.quantile(percentile)
    top_titles = title_frequency[title_frequency >= cutoff].index.tolist()

    threshold = 65  
    num_processes = cpu_count()  
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(fuzzy_match, [(title, top_titles, threshold) for title in df['job_title']]), total=len(df), desc="Matching job titles"))

    matches = {title: match for title, match in results}
    df['job_title'] = df['job_title'].map(matches)

    for col in tqdm(['job_title', 'job_location', 'company'], desc="Processing columns"):
        df[col] = df[col].apply(preprocess_text)
        df = drop_rare_labels(df, col, 50)
        print(f"Detailed analysis for {col}:\n", df[col].value_counts(), "\n")

    df.to_csv("processed_dataset.csv", index=False)
    
if __name__ == "__main__":
    main()
