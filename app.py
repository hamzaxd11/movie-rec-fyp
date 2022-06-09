import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

import joblib

import streamlit as st

import warnings 

from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic
from surprise import SVD
import heapq
from collections import defaultdict
from operator import itemgetter


warnings.filterwarnings('ignore')

@st.cache
def load_and_preprocess_data():
	movies = pd.read_csv('data/movies_preprocessed.csv')
	ratings = pd.read_csv('data/ratings_preprocessed.csv')

	return movies, ratings



def get_title_from_index(movie_id):
    title = movies[movies['movieId'] == movie_id]
    title = list(title['title'])
    return title[0]



def get_index_from_title(title):
    movie_id = movies[movies['title'] == title]
    movie_id = list(movie_id['movieId'])
    return movie_id[0]


def insert_ratings(ratings_dict, ratings):

	ids = []

	for movie_titles in ratings_dict.keys():
	    x = get_index_from_title(movie_titles)
	    ids.append(x)
	    
	id_list = [user_id] * len(ratings_dict)

	user_ratings = list(zip(id_list,ids, ratings_dict.values()))

	user_ratings = pd.DataFrame(user_ratings, columns=ratings.columns)

	ratings = pd.concat([ratings, user_ratings])

	return ratings, ids




def svd(ratings):

	np.random.seed(1)

	reader = Reader(rating_scale=(1, 5))
	data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

	svd = SVD(verbose=True)

	full_trainset = data.build_full_trainset()

	svd.fit(full_trainset)

	movie_ids = list(movies['movieId'])
	movies_not_rated = np.setdiff1d(movie_ids,ids)

	unrated = []

	for i in movies_not_rated:
	    rating = svd.predict(uid=999, iid=i)
	    unrated.append(rating[3])

	new_ratings_for_user = list(zip(movies_not_rated, unrated))

	new_ratings_for_user = pd.DataFrame(new_ratings_for_user, columns=['movieId', 'Predicted Rating'])
	new_ratings_for_user = pd.merge(movies,new_ratings_for_user,on='movieId')


	new_ratings_for_user.sort_values(by='Predicted Rating', ascending=False, inplace=True)

	new_ratings_for_user.reset_index(inplace=True,drop=True)

	new_ratings_for_user.drop(['movieId', 'Predicted Rating'], axis = 1, inplace=True)

	return new_ratings_for_user.head(20)



def userCF(ratings):
	reader = Reader(rating_scale=(1, 5))
	data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

	k = 10

	trainSet = data.build_full_trainset()

	sim_options = {'name': 'cosine',
	               'user_based': True
	               }

	model = KNNBasic(sim_options=sim_options)
	model.fit(trainSet)
	simsMatrix = model.compute_similarities()

	testUserInnerID = trainSet.to_inner_uid(user_id)
	similarityRow = simsMatrix[testUserInnerID]

	testUserInnerID = trainSet.to_inner_uid(user_id)
	similarityRow = simsMatrix[testUserInnerID]

	similarUsers = []
	for innerID, score in enumerate(similarityRow):
	    if (innerID != testUserInnerID):
	        similarUsers.append( (innerID, score) )

	kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])


	# Get the stuff they rated, and add up ratings for each item, weighted by user similarity
	candidates = defaultdict(float)
	for similarUser in kNeighbors:
	    innerID = similarUser[0]
	    userSimilarityScore = similarUser[1]
	    theirRatings = trainSet.ur[innerID]
	    for rating in theirRatings:
	        candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore


	# Build a dictionary of stuff the user has already seen
	watched = {}
	for itemID, rating in trainSet.ur[testUserInnerID]:
	    watched[itemID] = 1



	# Get top-rated items from similar users:
	recommendations = []
	scores = []

	pos = 0
	for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
	    if not itemID in watched:
	        movieID = trainSet.to_raw_iid(itemID)
	        title = get_title_from_index(int(movieID))
	        recommendations.append(title)
	        scores.append(ratingSum)        
	        #print(title, ratingSum)
	        pos += 1
	        if (pos > 20):
	            break


	recos = pd.DataFrame(columns=['title','scores'])

	recos['title'] = recommendations
	recos['scores'] = scores

	recos = pd.merge(movies,recos,on='title')


	recos.reset_index(inplace=True,drop=True)

	recos.sort_values(by='scores', inplace=True, ascending=False)

	recos.drop(['movieId', 'scores'], axis = 1, inplace=True)

	return recos


def userCF_Mean(ratings):

	combined = pd.merge(movies,ratings,on='movieId')

	moviemat = combined.pivot_table(index='userId',columns='title',values='rating').fillna(0)
	moviemat_sparse = csr_matrix(moviemat.values)
	cosine_sim = cosine_similarity(moviemat_sparse)

	k = 10

	#moviemat 
	recommender_df = pd.DataFrame(cosine_sim, 
	                              columns=moviemat.index,
	                              index=moviemat.index)


	## Item Rating Based Cosine Similarity
	cosine_df = pd.DataFrame(recommender_df[user_id].sort_values(ascending=False))
	cosine_df.reset_index(level=0, inplace=True)
	cosine_df.columns = ['userId','cosine_sim']
	similar_usr = list(cosine_df['userId'][1:k+1].values)
	similarities = list(cosine_df['cosine_sim'][1:k+1].values)

	sims_dict = dict(zip(similar_usr, similarities))

	similar_usr_df = moviemat.T[similar_usr].fillna(0)

	for i, j in sims_dict.items():
	    similar_usr_df[i] = similar_usr_df[i] * j

	similar_usr_df['mean rating'] = similar_usr_df[list(sims_dict.keys())].mean(numeric_only=True,axis=1)
	similar_usr_df.sort_values('mean rating', ascending=False,inplace = True)

	watched = list(ratings_dict.keys())

	similar_usr_df = similar_usr_df[~similar_usr_df.index.isin(watched)]

	titles = similar_usr_df.index
	mean_rating = list(similar_usr_df['mean rating'])

	recos = pd.DataFrame(columns=['title','mean rating'])
	recos['title'] = titles
	recos['mean rating'] = mean_rating

	recos = pd.merge(movies,recos,on='title')

	recos.drop('movieId', axis = 1, inplace=True)

	recos.sort_values(by='mean rating', ascending = False, inplace=True)
	recos.reset_index(drop=True, inplace=True)


	return recos.head(20)




@st.cache
def itemCF(movies,ratings, movie_name):

	combined = pd.merge(movies, ratings, on='movieId')
	pivot = combined.pivot_table(index='title',columns='userId',values='rating').fillna(0)
	#pivot = joblib.load('models/pivot.sav')

	#movie_titles = pivot.reset_index()
	#movie_titles = movie_titles['title'].to_frame()
	movie_titles = pd.read_csv('data/movie_titles.csv')

	#movie_matrix = csr_matrix(pivot.values)
	movie_matrix = joblib.load('models/movie_matrix.sav')

	#model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_jobs=-1)
	#model_knn.fit(movie_matrix)
	model_knn = joblib.load('models/model_knn.sav')

	record = movie_titles[movie_titles['title'] == movie_name]
	movie_index = list(record.index) 
	movie_index = movie_index[0]


	distances, indices = model_knn.kneighbors(pivot.iloc[movie_index,:].values.reshape(1, -1), n_neighbors = 20)

	names = []

	distances = distances.flatten()

	for i in range(1, len(distances)):
		names.append(pivot.index[indices.flatten()[i]])


	similar_movies = pd.DataFrame()
	similar_movies['title'] = names
	similar_movies['distances'] = distances[1:len(distances)]

	similar_movies = pd.merge(movies,similar_movies,on='title')
	similar_movies.sort_values(by='distances', ascending=True, inplace=True)

	similar_movies.drop('movieId', axis = 1, inplace=True)

	similar_movies.reset_index(drop=True, inplace=True)

	return similar_movies



if __name__ == '__main__':

	st.title("Movie Recommender System")

	st.sidebar.markdown('[![Muhammad Hamza Adnan]\
                    (https://img.shields.io/badge/Author-@hamzaxd11-gray.svg?colorA=gray&colorB=dodgerblue&logo=github)]\
                    (https://github.com/hamzaxd11/movie-rec/)')


	reco_types = ['Home','Personalized Recommendations', 'Find Similar Movies', 'Taste Match']

	choice = st.sidebar.selectbox("Type of Recommendations" ,reco_types)

	st.markdown("Rate movies and get personalized recommendations or find movies similar to the ones you liked")

	movies, ratings = load_and_preprocess_data()

	movie_titles = movies['title']
	movie_titles = list(movie_titles.sort_values())

	user_id = 999

	rating_scale = [1,2,3,4,5]

	if choice == 'Home':
		st.markdown('**For Personalized Recommendations:**')

		st.text('1) Select the Personalized Recommendations option in the sidebar.')
		st.text('2) Select 5 movies youve already seen.')
		st.text('3) Rate them on a scale of 1 to 5.')
		st.text('4) Press the Submit ratings button at the bottom.')
		st.text('5) Enjoy your personalized recommendations!')

		st.markdown('**To find movies similar to the ones you liked:**')

		st.text('1) Select the Find Similar Movies option in the sidebar.')
		st.text('2) Select a movie.')
		st.text('4) Enjoy the recommendations!')

		st.markdown('**To check how similar your movie taste is with another person:**')

		st.text('1) Select the Taste Match option in the sidebar.')
		st.text('2) You should rate 5 movies with another person.')
		st.text('4) Check your similarity score!')		

	elif choice == 'Personalized Recommendations':

		movie_1 = st.selectbox("Movie 1",movie_titles)

		rating_1 = st.select_slider(
		'Movie 1 rating', rating_scale)


		movie_2 = st.selectbox("Movie 2",movie_titles)

		rating_2 = st.select_slider(
		'Movie 2 rating', rating_scale)


		movie_3 = st.selectbox("Movie 3",movie_titles)

		rating_3 = st.select_slider(
		'Movie 3 rating', rating_scale)


		movie_4 = st.selectbox("Movie 4",movie_titles)

		rating_4 = st.select_slider(
		'Movie 4 rating', rating_scale)


		movie_5 = st.selectbox("Movie 5",movie_titles)

		rating_5 = st.select_slider(
		'Movie 5 rating', rating_scale)


		user_movies = [movie_1,movie_2,movie_3,movie_4,movie_5]
		user_ratings = [rating_1,rating_2,rating_3,rating_4,rating_5]


		ratings_dict = dict(zip(user_movies, user_ratings))

		new_ratings, ids = insert_ratings(ratings_dict, ratings)



		if st.button('Submit ratings'):
			st.write(ratings_dict)

			#userCF_recs = userCF(new_ratings)

			svd_recs = svd(new_ratings)

			userCF_recs_Mean = userCF_Mean(new_ratings)


			#col1, col2,col3 = st.beta_columns([5, 5, 5])

		#with col1:
			st.header("SVD")
			st.write('Matrix Factorization. Predicts what rating a user will give to a movie by learning a users features from their ratings')
			st.write(svd_recs)

		#with col2:
			#st.header("User CF")
			#st.write('Finds the n most similar users to you and scores movies based on user similarity and also the number of occurences of the movie')
			#st.write(userCF_recs)

		#with col3:
			st.header("User based Collaborative Filtering")
			st.write('Finds the n most similar users to you and scores movies based on user similarity and takes the mean score of each movie')
			st.write(userCF_recs_Mean)



	elif choice == 'Find Similar Movies':

		st.header('How this algorithm works:')
		st.write('People who rated a movie highly, also rated these movies similarly')

		movie_name = st.selectbox("Movie",movie_titles)

		recos = itemCF(movies,ratings,movie_name)

		st.write(recos)



	elif choice == 'Taste Match':

		rating_scale2 = [0,1,2,3,4,5]

		st.header('Taste Match:')
		st.write('Tells how similar the tastes in movies of two people are using Cosine Similarity')

		col1, col2, col3 = st.beta_columns([5, 5, 5])

		with col1:
			st.header("Movies")

			movie_1 = st.selectbox("Movie 1",movie_titles)
			movie_2 = st.selectbox("Movie 2",movie_titles)
			movie_3 = st.selectbox("Movie 3",movie_titles)
			movie_4 = st.selectbox("Movie 4",movie_titles)
			movie_5 = st.selectbox("Movie 5",movie_titles)



		with col2:
			st.header("User 1")

			user1_rating1 = st.select_slider(
			'Movie 1 rating', rating_scale2)

			user1_rating2 = st.select_slider(
			'Movie 2 rating', rating_scale2)

			user1_rating3 = st.select_slider(
			'Movie 3 rating', rating_scale2)

			user1_rating4 = st.select_slider(
			'Movie 4 rating', rating_scale2)

			user1_rating5 = st.select_slider(
			'Movie 5 rating', rating_scale2)



		with col3:
			st.header("User 2")


			user2_rating1 = st.select_slider(
			'Movie1 rating', rating_scale2)

			user2_rating2 = st.select_slider(
			'Movie2 rating', rating_scale2)

			user2_rating3 = st.select_slider(
			'Movie3 rating', rating_scale2)

			user2_rating4 = st.select_slider(
			'Movie4 rating', rating_scale2)

			user2_rating5 = st.select_slider(
			'Movie5 rating', rating_scale2)


		if st.button('Submit ratings'):

			user1_ratings = [user1_rating1, user1_rating2, user1_rating3, user1_rating4, user1_rating5]
			user2_ratings = [user2_rating1, user2_rating2, user2_rating3, user2_rating4, user2_rating5]

			#st.write(user1_ratings)
			#st.write(user2_ratings)

			user1_ratings = np.array(user1_ratings)
			user2_ratings = np.array(user2_ratings)

			user1_ratings = user1_ratings.reshape(1,5)
			user2_ratings = user2_ratings.reshape(1,5)


			#st.write(user1_ratings)
			#st.write(user2_ratings)

			st.write('Similarity Score is:')
			st.write(cosine_similarity(user1_ratings, user2_ratings))





	else:
		pass