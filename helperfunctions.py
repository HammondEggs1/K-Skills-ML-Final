import pandas as pd
from joblib import dump, load
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

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

# input skills as a comma separated String
# fileName is the file name of an existing classifier
def modelPredict(skills, fileName):
	prediction = {}
	vectorizer = loadClf("vectorizer_bot50.joblib")
	skills = pd.Series(skills)
	xTest = vectorizer.transform(skills)
	print(xTest)
	clf = loadClf(fileName)
	y = clf.predict(xTest)
	encoder = loadClf("label_encoder_bot50.joblib")
	prediction["title"] = encoder.inverse_transform(y)
	data = pd.read_csv("dataset_bot50_p99.csv")
	data_subset = getSubset(prediction["title"][0], data)
	k = 3
	companies = getKCompanies(data_subset, k)
	prediction["company"] = getKLocations(data_subset, k)
	prediction["location"] = getKCompanies(data_subset, k)
	return prediction

def main():
	skills1 = "SQL, Azure, JQuery, HTML5, CSS3, Kotlin, C#, Visual Studio"
	skills2 = "Restaurant Experience, Communicaation skills, Team Work, Food Preparation"
	skills3 = "Microsoft Office Suite, Adobe, Accounting, Payroll, Time management, Organization"
	prediction = modelPredict(skills3, "dt_bot50.joblib")
	print(prediction["title"][0])
	print(prediction["company"])
	print(prediction["location"])


if __name__ == "__main__":
    main()
