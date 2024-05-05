import pandas as pd
from joblib import dump, load
from collections import Counter

# takes a job title and the dataset as input
# outputs dataset rows where the job title
# is identical to the given title
def getSubset(title, data):
	indices = []
	# Gets indices where job title matches input
	for i, sample_title in enumerate(data["job_title"]):
		if(sample_title == title):
			indices.append(i)
	# Takes a subset of our data dfs based on indices
	print(type(data))
	data_subset = data.iloc[indices]
	return data_subset


# takes subsetted data as input and desired number
# of companies and outputs the k companies with
# the most openings of that job
def getKCompanies(data, k):
	company_counts = Counter(data["company"])
	common_company = company_counts.most_common(k)
	k_companies = []
	for companies in common_company:
		k_companies.append(companies[0])
	return k_companies

# takes subsetted data as input and desired number
# of companies and outputs the k cities with
# the most openings of that job
def getKLocations(data, k):
	location_counts = Counter(data["job_location"])
	common_location = location_counts.most_common(k)
	k_locations = []
	for location in common_location:
		k_locations.append(location[0])
	return k_locations

def saveClf(clf, fileName):
	dump(clf, fileName)

def loadClf(fileName):
	return load(fileName)

def main():
	k = 5
	data = pd.read_csv("linkedin_job_postings.csv")
	data_subset = getSubset("Shift Manager", data)
	locations = getKLocations(data_subset, k)
	companies = getKCompanies(data_subset, k)
	print(locations)
	print(companies)


if __name__ == "__main__":
    main()
