import argparse
import pandas as pd
def main():
# set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("job_skills",
                        # default="job_skills.csv",
                        help="filename for job skills")
    # parser.add_argument("--job_summary",
    #                     # default="job_summary.csv",
    #                     help="filename for job summary")
    parser.add_argument("job_postings",
                        # default="linkedin_job_postings.csv",
                        help="filename for webscrapped linkedin job postings")
    parser.add_argument("combined_data",
                        # default="combined_data.csv",
                        help="filename for output csv")
    args = parser.parse_args()
    
    job_skills = pd.read_csv(args.job_skills)
    job_postings = pd.read_csv(args.job_postings)
    
    result = pd.merge(job_skills, job_postings, on='job_link',how='inner')
    result.to_csv(args.combined_data, index=False)
    
if __name__ == "__main__":
    main()
