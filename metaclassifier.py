import argparse
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

""" 
Input: k binary classifications as a dataframe
where column indices are models and values
are binary classifications of each dataset row

Output: a single list of scores from 0-1 based
on the number of models that classfied the value
as of the desired job type
"""
def generateScores(classifications):
	scores = []
	n = len(df.columns)
	divisor = n ** 1.5
	# I chose the function 1-classifications^1.5 / models^1.5
	# because it grows at a low but increasing rate
	for i, row in df.itertupels():
		scores.append((sum(row)**1.5)/divisor)
	return scores

""" 
Input: 
Scores: scores from 0-1 describing degree of
accordance with criteria
Data: skills data dataframe
Threshhold: Minimum score for inclusion in
subset

Output: a subset of the dataset and scores
that have scores that are above the threshhold
"""
def getSubset(scores, data, threshhold):
	indices = []
	# Gets indices where scores are above the threshhold
	for i, score in enumerate(scores):
		if(score>treshhold):
			indices.append(i)
	# Takes a subset of our score and data dfs based on indices
	data_subset = data.iloc(indices)
	scores_subset = [scores[i] for i in indices]
	return scores_subset, data_subset

""" 
Input: 
Data: subset of skills data dataframe
kmultiplier: the multiplier that will multiply
k to generate the number of candidate skills
to be counted and considered for the weighted
average

Output: dataframe with columns for k*multiplier
skills and rows representing the binary presence
of the columns skill in that row of the dataset
"""
def hasSkills(data, k, kmultiplier):
	# Flatten the 'job_skills' column in the data DataFrame 
	# into a single list of all individual skills
	flattened_list = [item for sublist in data['job_skills'] for item in sublist.split(',')]
	# Count the frequency of each skill in the flattened list
	skill_counts = Counter(flattened_list)
	# Get the k * kmultiplier most common skills
	candidate_skills = skill_counts.most_common(int(k*kmultiplier))
	candidate_dict = {}
	for i, skill in enumerate(candidate_skills):
		# Use vectorization to generate binary presence of 
		# a given skill in each row
        vectorizer = CountVectorizer(vocabulary=[skill])
        job_skills = vectorizer.transform(data['job_skills']).toarray().tolist()
        job_skills = [item for sublist in test_data for item in sublist]
        candidate_dict[skill] = pd.Series(job_skills).clip(upper=1)
	return pd.DataFrame(candidate_dict)

""" 
Input: 
Skillclass: dataframe with columns for k*multiplier
skills and rows representing the binary presence
of the columns skill in that row of the dataset
scores: scores from 0-1 describing degree of
accordance with criteria

Output: list of k most desirable skills based on
a weighted average of binary classifications and
scores
"""
def getKBest(skillclass, scores, k):
	skillScores = {}
	# totals weighted scores for each skill
	for i, column in skillclass.columns():
		total = 0
		for j, val in column:
			total = total + (val * scores[j])
		skillScores[column] = total
	# returns k highest scoring skills
	return [k for k, v in sorted(skillScores.items(), key=lambda x: x[1], reverse=True)[:k]]

def main():
	k = 2
	scores = generateScores(classifications)
	scores_subset, data_subset = getSubset(scores, data, 0.2)
	skillclass = hasSkills(data_subset, k, 2)
	print(getKBest(skillclass, scores_subset, k))

if __name__ == "__main__":
    main()

