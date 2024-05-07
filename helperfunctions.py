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
	vectorizer = loadClf("vectorizer_fpd.joblib")
	skills = pd.Series(skills)
	xTest = vectorizer.transform(skills)
	clf = loadClf(fileName)
	y = clf.predict(xTest)
	y = [int(x) for x in y]
	encoder = loadClf("label_encoder_fpd.joblib")
	prediction["title"] = encoder.inverse_transform(y)
	data = pd.read_csv("further_processed_dataset.csv")
	data_subset = getSubset(prediction["title"][0], data)
	k = 3
	companies = getKCompanies(data_subset, k)
	prediction["company"] = getKLocations(data_subset, k)
	prediction["location"] = getKCompanies(data_subset, k)
	return prediction

def main():
	skills1 = "Java, C, SQL, PostGreSQL, Database Systems, Kotlin, Python, Machine Learning, Artificial Intelligence, Quantum Computing, CSS, HTML, HTML5, CSS3, Web Development, Android Studio, Visual Studio, Android Development, App Development, Android SDKs, Agile methodologies, Debugging, Frameworks, Git, Unit Testing, Mobile App Development, APIs"
	skills2 = "Food Safety, Internal Communication, Inventory Management, Daily Maintenance, Cleanliness, Quality Food Production, Exceptional Customer Service, Safety, Security, Scheduling, Training, Leadership, Restaurant, Retail, Hospitality, English, High School"
	skills3 = "Medical, Documentation, Patient Care, PCU experience, BLS, ACLS, EPIC, Mentoring, Fall prevention documentation, Stroke documentation, Physician communication, Appointment scheduling, Skin assessments, Care Partners, LPN/LVN"
	"""
	print("KNN:")
	prediction = modelPredict(skills, "knn_fpd.joblib")
	print(prediction["title"][0])
	print(prediction["company"])
	print(prediction["location"])
	print("XGB:")
	prediction = modelPredict(skills, "XGB_fpd.joblib")
	print(prediction["title"][0])
	print(prediction["company"])
	print(prediction["location"])
	print("DT:")
	prediction = modelPredict(skills, "DT_fpd.joblib")
	print(prediction["title"][0])
	print(prediction["company"])
	print(prediction["location"])
	"""
	print("NN, skills1:")
	prediction = modelPredict(skills1, "mlp_fpd.joblib")
	print(prediction["title"][0])
	print(prediction["company"])
	print(prediction["location"])
	print("NN, skills2:")
	prediction = modelPredict(skills2, "mlp_fpd.joblib")
	print(prediction["title"][0])
	print(prediction["company"])
	print(prediction["location"])
	print("NN, skills3:")
	prediction = modelPredict(skills3, "mlp_fpd.joblib")
	print(prediction["title"][0])
	print(prediction["company"])
	print(prediction["location"])
	

if __name__ == "__main__":
    main()
