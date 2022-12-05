### Problem Definition

A popular music application like Spotify has data about millions of users and millions of songs which only keeps growing with time. With a dataset of this size, it is impractical to use a centralized server system to store and process this data. This emanates the need for a distributed architecture wherein we can add new nodes to the system as per requirement without needing to migrate from the old system. We have designed a distributed Music Recommendation System using Spark framework.

### Introduction and Motivation
* Horizontal and Vertical Scalability - Centralized client server architecture provides vertical scalability, that is, it supports increasing the compute of existing nodes in the system. There is a threshold to how much this compute can be increased. Also, there is a greater risk of system going down and it is expensive to implement. Whereas, in distributed architecture, along with adding compute to the existing nodes, we can also add new nodes to the system.

* Robustness - It is able to function even if one or more of its software or hardware components fail

* No single point of failure - In the absence of a centralized server entity, distributed systems are fault tolerant.

### Spark Overview
![](Spark_Distributed_Architecture.png)

We build a distributed recommendation system over Apache Spark. All applications run on spark as independent processes. Spark consists of a master node and multiple worker nodes. The master node consists of a driver program which is connected to a cluster manager which is responsible for allocating resources to the worker nodes, creating tasks and distributing them to worker nodes. These tasks are then executed on the worker nodes and results are returned to the driver program which consolidates them and gives back the result.


### Million Song Dataset
The [Million Song Dataset](http://millionsongdataset.com/) consists of audio features and metadata for a million contemporary popular music tracks.
It has information about tracks, song id, song title, artist id, artist name, danceability, year of release, user id, song play count etc.

* **1,019,318** unique users

* **384,546** unique MSD songs

* **48,373,586** user - song - play count triplets

* **5 GB** Dataset Size (excluding audio tracks)

All the files are in HDF5 format. They are first converted to CSV format. To get the data for our use, we joined the metadata file and triplets file on the song id. We used the entire 5GB dataset for training the model.

### Data snippet

|  **userID**   |  **SongID**    |  **Rating**  |  **trackID**  |  **artistName**              |  **songTitle**                         |
|-----------|------------|----------|-----------|--------------------------|------------------------------------|
|  250..48  |  SOAK..95  | 1        |  TRIQ..D  |  Jack Johnson            |  The Cove                          |
|  250..48  |  SOFR..C0  | 1        |  TRDM..1  |  Harmonia                |  Sehr kosmisch                     |
|  250..48  |  SOJT..3D  | 1        |  TRVM..4  |  Gipsy Kings             |  Vamos A Bailar                    |
|  250..48  |  SOYH..7F  | 5        |  TRVP..8  |  Jack Johnson            |  Moonshine                         |
|  250..48  |  SOAX..A1  | 1        |  TRHK..F  |  Florence + The Machine  |  Dog Days Are Over (Radio   Edit)  |
|  250..48  |  SODJ..CE  | 1        |  TRVC..6  |  Train                   |  Hey_ Soul Sister                  |
|  250..48  |  SONY..C9  | 5        |  TROA..3  |  OneRepublic             |  Secrets                           |
|  250..48  |  SORM..95  | 1        |  TRJY..F  |  Jorge Gonzalez          |  Esta Es Para Hacerte Féliz        |

### Technology Stack
* Used Spark framework over Hadoop MapReduce - which is an iterative algorithm. Each iteration requires a hadoop job which involves continuous reading and writing to the disk which becomes a bottleneck. Spark loads user-song matrix in memory and caches it. 

* Used Amazon Cloud Storage S3  over HDFS for storing the dataset because of its elasticity, better availability and durability, 2X performance and its lower cost.

### Solution Proposal

We present three methods for implementing the recommendation system :

* Alternating Least Squares
* Cosine Similarity using MapReduce
* Friend Based Collaborative Filtering

### ALS Architecture
![](ALS%20Architecture.png)

"Alternating least squares (ALS)" is a distributed matrix factorization method that allows for faster and more efficient computations. The major goal of creating such an algorithm is to overcome some of the disadvantages that kNN-based approaches have, such as popularity bias, cold-start concerns, scalability challenges, sparseness in the matrix, and long calculation times.

Spark is a widely used distributed framework that is both dependable and fault-tolerant. As a result, the ALS algorithm was built using the Apache Spack ML package and is intended for large-scale collaborative filtering. It has a basic, elegant design that scales well to enormous datasets.

In ALS, L2 regularization is employed to lower two loss functions alternatively. In an MF technique, when a matrix R is partitioned into smaller matrices U and V, R = UV', the cost function is a non-convex function. The problem becomes a linear regression problem with an "Ordinary Least Squares (OLS)" solution if one of them, say U, is fixed.

Alternating least squares accomplishes just that. It's a two-step iterative optimization approach. It initially fixes U and then solves for V, then fixes V and then solves for V in each cycle. The algorithm is called "Alternating" because it solves matrices in different ways. Because the OLS solution is unique and guarantees a low MSE, the cost function can decrease or remain unchanged in each step, but it never rises. The cost function is lowered by switching back and forth between the two processes until convergence. It is certain that it will only converge to a local minimum and that the initial values of U or V will ultimately define it.

In 2014, Spotify used an ALS-based approach called full-gridify. The users are divided into small clusters along the rows, as indicated by rectangular boxes. 

In addition, all of the goods rated by the users are organized into columns. The user groups are now transmitted to individual worker nodes, together with the needed item vectors for the ALS calculation, once such a grouping has been established. Even if the item vectors are replicated across worker nodes, the grouping avoids the worker nodes from storing all of the items in memory. Furthermore, because it only stores information for a specific set of users, it requires less memory, and all of the data may be cached for speedier access. Furthermore, because it only stores information for a specific set of users, it requires less memory, and all of the data may be cached for speedier processing. Because all data is stored in a worker node, there is no need for data shuffling, reducing the IPC overhead of distributed systems.

### More about ALS
<img src="ALS-1.png" width="500">

* Dataset for music recommendation is an implicit feedback data -   a major reason for use to try LS is because the data is implicit. Implicit feedback doesn’t directly reflect the interest of the user but is the information produced after observing the users' behavior. In explicit we have negative feedback, in implicit we don’t (listen count one can imply liked or didn’t like. The model treats implicit data as a binary ratings matrix. It uses the listen count as a confidence value which is incorporated in the loss function which needs to be minimized.  As the listen count grows, we have a stronger indication that the user indeed likes the song.
* All users haven’t heard all songs hence user-song matrix is sparse - For each user, we have number of songs plays for only a subset of the songs as any user could not have heard all possible songs, so the user song matrix that we have is sparse. Goal is to predict the number of listens for a user song pair in the sparse matrix. With ALS, the idea is to approximate the matrix by factorizing it as the product of two matrices: one that describes properties of each user, and one that describes properties of each song.
* Predicts number of listens using Matrix Factorization – These two lower order matrices can be used to predict values in the original matrix. We want to select these two matrices such that the error for the users/songs pairs where we know the correct number of plays is minimized.
* It is an iterative algorithm (alternates back and forth between user and song vectors for solving) – To get the values for these weights, or features,  Alternating Least Squares algorithm does this by first randomly filling the users matrix with values and then optimizing the value of the songs such that the error is minimized. Then, it holds the songs matrix constant and optimizes the value of the user's matrix. This alternation between which matrix to optimize is the reason for the "alternating" in the name. Alternate back and forth until we converge and the product of the two matrices approximates the original one as closely as possible. When we feed the entire original sparse user-song matrix into spark, it is partitioned into chunks and sent to different worker nodes. Each partition contains the required data to calculate the values of the user vectors present in that partition. Only item vectors required for that block are sent to this particular worker where the ratings are cached and all the processed happens on this cached data. Since all user specific data exists on this node, no shuffling of data over the wire takes place for grouping.


<img src="RMSE_vs_Nodes.png" width="300">

|Parameter|Value|
|-------|----------|
|Rank (latent factors)| 16|
|Regularization Parameter| 0.25|
|Max Iterations| 20|
|alpha| 40|

We used crossvalidator and grid search to run multiple iterations and find the optimal parameters for our model. 

After trying different rank values - which is the number of latent factors - rank 16 produced the best results.


### Recommendations from ALS

| UserID | Actual top 5 songs | Artist Name               | Recommended Songs  | Artist Name               |
|--------|--------------------|---------------------------|--------------------|---------------------------|
| 7179   | Heartbreak Warfare | John Mayer                | Sehr kosmisch      | Harmonia                  |
| 7179   | The Cove           | Jack Johnson              | Moonshine          | Jack Johnson              |
| 7179   | Sehr kosmisch      | Harmonia                  | Heartbreak Warfare | John Mayer                |
| 7179   | Country Road       | Jack Johnson / Paula Fuga | Country Road       | Jack Johnson / Paula Fuga |
| 7179   | Holes To Heaven    | Jack Johnson              | Sun Hands          | Local Natives             |

If we look at the predictions given by ALS for one particular user, we can see that the top 5 predicted songs very closely resemble the actual most listened songs by that user. Even if the prediction doesn’t exactly match, it recommends similar songs by the same artist or genre. We didn’t didn’t remove the songs the user has already heard from our recommendation so that we could evaluate the performance.

### Cosine Similarity using MapReduce

This approach uses item-item collaborative filtering to provide the recommendations for a user.  All the compute intensive tasks are split between mapper nodes and the data is collected and collated by the master or reducer code. Hence, Map reduce. The same is implemented using spark.

We used item-item collaborative filtering over user-user collaborative filtering for a number of reasons:

* Performance: It performs better than user-user similarity and hence is a popular choice for high load services.
* Cold start problem: Item-item collaborative filtering handles the cold start problem better as when a new user enters into the system, he can be asked to choose a few songs he finds interesting and based on our pre-computed data for song-song similarity we can recommend similar songs to the user. 
* Number of computations: The number of songs is lesser than the number of users in our dataset. Hence, the number of computations is much lesser for item-item collaborative filtering approach  
* Item-item similarity remains more constant as opposed to user-user similarity which changes frequently. 
* Accuracy: In item-item approach as the recommendations are more accurate.
* Security: Resistance of this approach to shilling attack. Shilling attack is where the attackers try to manipulate recommendations by adding user-rating content maliciously. Here again its based on the user’s choice himself it is resistant to shilling attacks. 


### Cosine Similarity using MapReduce - Architecture
![](Cosine%20Similarity%20using%20mapreduce%20-%20architecture.png)

The basic methodology includes two steps: 
* data transformation
* cosine similarity algorithm application
 
We started with the data set consisting of user, song and rating information. We calculated all the pairs of songs listened by the users and the corresponding ratings of songs. This is done for all the users and all the songs they have listened to. Once we have this list consisting of song pairs and rating pairs, for each song pair we form a vector of ratings pairs collected by a number of users. Next the cosine similarity algorithm is applied on this vector to find the similarity score of the song pair.	

While providing the recommendation for a user, we consider the user’s top songs and recommend other songs which are similar to his listening history. Further, we apply other filters like similarity score greater than certain threshold, song pair appearance > certain count to make the recommendations more relevant. We are using implicit data that is the song count and normalizing it. Ideally cosine similarity works better with explicit data, however due to lack of dataset with this information we used implicit data.


### More about Cosine Similarity

#### Data shuffling issue
In item-item collaborative filtering each mapper node contains information about a subset of items. Hence, during different item-item calculations it  requires shuffling of item data over different worker nodes. This data shuffling is a very expensive operation and hence slows down the process.

![](Cosine%20Similarity%20using%20mapreduce.png)

### Cosine Similarity code snippets
* Data Transformation

```
rdd = sc.textFile(data_path)\
            .map(lambda x: x.split(",")).map(lambda x : ((x[1]),(x[2],x[3]))) 
songpair = rdd.join(rdd)
songpair_withoutdups = songpair.filter(remove_duplicates)
just_songpairs = songpair_withoutdups.map(justsongpairs)
groupOfRatingPairs = just_songpairs.groupByKey()
songPairsAndSimilarityScore = groupOfRatingPairs.mapValues(findCosineSimilarity)
```

* Cosine Similarity Algorithm

```
    for pair in rdd:
        X, Y = pair
        ratingX = float(X)
        ratingY = float(Y)
        xxsum = xxsum + ratingX*ratingX
        yysum = yysum + ratingY*ratingY
        xysum = xysum + ratingX*ratingY
        num_of_pairs = num_of_pairs+1
    
    denom = 0.0 + math.sqrt(xxsum)*math.sqrt(yysum)

    result = 0.0 + xysum/denom
    return (result,num_of_pairs)
```

* Algorithm to find top recommendations for a user

```
    for song in top5_songs:
        song_id = song
        filteredsongSimilarity = songPairsAndSimilarityScore.filter(filterOnThreshold)
        top10 = filteredsongSimilarity.take(10)
        top50.extend(top10)
    top50.sort(key = lambda x: x[1][0])
    top50 = top50[0:20]
    already_displayed.clear()
    for song in top5_songs:
        song_id = song
        displayTop10(top50,song_id)
```

### Recommendations from Cosine Similarity
![](mp_recommendations.png)

This the recommendation results which we obtained from map reduce approach for the same user which we considered in ALS approach. We observed that the recommendations we obtained from both the approaches were similar in terms of artists, genre etc. And we got 3 out of top 5 recommendations to be exactly similar.  


### Friend based collaborative filtering

As observed from previous experiments and results, user-based and item-based Collaborative Filtering is computationally expensive, hence this enabled us to explore other solutions for large scale collaborative filtering.

Friend-Based CF is based on the assumption that an individuals taste/liking is strongly influence by the people around him. It is more likely that an individuals taste in music is more similar to his friend rather than a stranger in a different country. Hence, if we can define these relations and form smaller cluster then it is possible to use CF to compare an individual only to his friends and connections in order to determine similarity. This would reduce the computation time as the number of people per cluster would be significantly less and not every user needs to be compared to every other user. Hence, a friend based collaborative filtering would be more efficient, accurate and scalable. 


### Friend based collaborative filtering - Architecture
![](friend_based_architecture.png)

The main component here is the Clustering algorithm - We have used a graph algorithm (BFS) to identify an users 2nd degree connections from the dataset and formed smaller clusters. Once the smaller clusters are formed, the complete set of users in a cluster and the songs they listen to can be sent to individual worker nodes to calculate the user-similarity matrices per cluster and then sent back to the master for recommendations.

Advantages of Friend-Based Collaborative Filtering:

1) Efficient: Since the computations are done for a smaller cluster now, the computation time per user is very low as there are very few user-user comparisons. Additionally, since the complete set of users per cluster along with their music ratings is cached on the worker node, there is no data shuffling between the workers and hence IPC time is saved. 

2) Cold Start Problem: When a new user with limited data is added to the system, he can be recommended songs based on what his/her friends like. This would help to address the cold-start problem.

Hence, this method overcomes the cost problems like that of normal Memory-based CF and also overcomes the cold start problem of ALS and hence should be a more viable and efficient solution.

![](friend_based.png)

### Friend based collaborative filtering - Algorithm

* Create a list of all artist count
* Remove all artists that the user has already seen
* Initialize all artists weight count to 0
* Get top 10 users most similar to the particular user
* For each of the 10 users –
  - Get all artists listened to by the user
  - Multiply the listen count for each artist with the similarity score of the user
  - Add this user weight to the overall weight of the artist
* After amplifying the listen count using all closest users, fetch the top 5 artists with the highest updated weight

```
graph = dict()

def addLinks(df, graph):
    graph[df['userID'].iloc[0]] = set(df['friendID'])
    return

df.groupby('userID').apply(addLinks, graph = graph)


def get_all_connected_groups(graph):
    already_seen = set()
    result = []
    for node in graph:
        if node not in already_seen:
            connected_group, already_seen = get_connected_group(node, already_seen, graph)
            result.append(connected_group)
    return result


def get_connected_group(node, already_seen, graph):
    result = []
    already_seen.add(node)
    result.append(node)
    nodes = graph[node]
    while nodes:
        node = nodes.pop()
        if node in already_seen:
            continue
        already_seen.add(node)
        result.append(node)
        nodes = nodes | graph[node]
    return result, already_seen


def get_second_degree(graph, userID):
    nodes = graph[userID]
    result = set()
    r = set()
    for node in nodes:
      result.update(graph[node])
      r.update(graph[node])
    return result
```

### Evaluation Results

#### ALS Experimentation Results

<img src="ALS_TrainTime.png" width="400">

<img src="RMSE%20Calculation%20Time.png" width="400">

We can see how the training time of the algorithm decreases with increasing number of nodes in the cluster.

The second graph shows the time taken to calculate the RMSE on the test data vs the number of nodes in the cluster. It is visible here as well how the time taken decreases along with the increasing number of nodes in the cluster.

<img src="numCores.PNG" width="500">

<img src="sparkUIcapture.PNG" width="500">

The above images show the execution of the tasks performed by spark along with the number of cores used and how the tasks are parallelized.

#### Cosine Similarity Experimentation Results

<img src="MapReduce%20Exp%20results.png" width="407">

| No of Nodes | Time Taken(sec) | Time Taken(sec) |
|-------------|-----------------|-----------------|
|             | 48M rows        | 1M rows         |
| 4           | -               | 101             |
| 8           | ~6300           | 57              |

#### Result Analysis

Fig: Training + prediction times for 10 million rows on 2,4,6 and 8 nodes on cluster
The entire dataset - 48M rows couldn’t be processed on our cluster with 2, 4 or 6 nodes (8GB RAM each) because of the executor running out of memory. This  was leading to the cluster disconnecting abruptly.
We could only calculate the item-item similarity on the entire dataset by using 8 nodes of 16GB RAM each which took a high computation time of around 6300s.
The results do not linearly scale up when compared to a smaller dataset of 1M rows because of polynomial increase in number of shuffles and comparisons (n^2 comparisons) and increased IPC overhead during data shuffling. The join operation time increases quadratically (n^2) with dataset size.  


#### ALS Vs MapReduce

|                     | Time taken for 8 nodes |
|---------------------|------------------------|
| ALS                 | ~791                   |
| Cosine - Similarity | ~6308                  |

#### Friend based Experimentation Results

We carried out a small POC to prove that the accuracy of recommendations remain unhampered  even if not all users are used in the similarity matrix computation.

We took a user-artist dataset, which includes rating of an user for different artist. This dataset also included a user-friend datafile, which described the friends a particular user has. We used this data file to create small clusters using our graph algorithm.

Once the smaller clusters were obtained, we carried out the following experiment:

For user 128, we computed the similarity matrix considering all the users of the dataset (2000 users).
The top 10 most similar users were - 1210, 1866, 374, 1643, 1209, 428, 1585, 176, 196, 788

For user 128, we computed the similarity matrix considering only the users in his cluster (250 users).
The top 10 most similar users were - 1210, 1866, 374, 1643, 1209, 428, 1585, 176, 196, 666

10-most similar users in both scenarios:

![](fb-1.png)

As observed from the Venn Diagram above - 9 out of 10 users are similar in both the cases. Hence, we can say that a smaller subset cluster would be better/equivalent to using the whole dataset consisting all users.

General recommendations for different cases using Friend-Based CF

![](fb-table.png)

Disadvantages of Friend-Based CF

People with limited friends i.e., smaller clusters still face the cold-start problem (as seen above for the hard case).
Several other factors like geographic location, state of mind etc. can influence a person’s taste of songs.

### Conclusion

We found out that ALS based on Matrix Factorization is one of the most efficient Distributed algorithm available which works well for implicit data and explicit data as well. But the main disadvantage it suffers from is the Cold-Start problem for new users.

So, we recommend a Social Recommendation approach using Friend-Based Collaborative Filtering. This significantly reduces the number of user-user comparisons and using an ALS like architecture, which reduces the IPC cost, it makes this approach more efficient, accurate and more scalable.

### Lessons Learned

* Handling Huge Datasets - We worked with BigData and a dataset of size 5GB for the first time. We dealt with problems of data cleaning and combing different parts of the dataset to obtain a usable dataset.
* Learnt about different algorithms used in Collaborative Filtering (Both Memory based and model based).
* Worked/learned about distributed architecture for Recommendation Systems.
* Understood how Spark and mapReduce frameworks internally work.
* Learned working with Spark clusters on cloud (AWS).
* Learned about different clustering techniques to form smaller clusters.
* Finally, enjoyed collaborating with team, the fruitful discussion and collectively working towards achieving a common goal.

### Future Work

We would like to:
* Carry out more experiments to prove efficacy of Friend-Based CF in terms of training time on larger datasets.
* Explore Hierarchical Agglomerative Clustering to form clusters instead if simple graph based algorithms.
* Look into different metrics to quantify results to compare efficiencies of different approaches.

### References

[1] Rohit and A. K. Singh, "Scalable recommender system based on MapReduce framework," 2017 IEEE International Conference on Power, Control, Signals and Instrumentation Engineering (ICPCSI), 2017, pp. 2892-2895, doi: 10.1109/ICPCSI.2017.8392251.

[2] Efthalia Karydi and Konstantinos Margaritis. 2016. Parallel and Distributed Collaborative Filtering: A Survey. <i>ACM Comput. Surv.</i> 49, 2, Article 37 (June 2017), 41 pages. https://doi.org/10.1145/2951952

[3] S. S. Agrawal, G. R. Bamnote, and S. L. Satarkar. 2016. A Hybrid Clustering Based Collaborative Filtering (CF) Approach. In <i>Proceedings of the Second International Conference on Information and Communication Technology for Competitive Strategies</i> (<i>ICTCS '16</i>). Association for Computing Machinery, New York, NY, USA, Article 21, 1–5. https://doi.org/10.1145/2905055.2905079

[4] Kumar, G.Senthil & Subramani, Anuja & Ravikumar, Keerthana & Bhatnagar, Avani. (2020). LOCATION-BASED RECOMMENDER SYSTEM USING COLLABORATIVE FILTERING AND CLUSTERING. Xi'an Jianzhu Keji Daxue Xuebao/Journal of Xi'an University of Architecture & Technology. XII. 2878. 

[5] https://medium.com/@varunabhi86/movie-recommendation-using-apache-spark-1a41e24b94ba

[6] https://towardsdatascience.com/uncovering-how-the-spotify-algorithm-works-4d3c021ebc0

[7] https://medium.com/analytics-vidhya/model-based-recommendation-system-with-matrix-factorization-als-model-and-the-math-behind-fdce8b2ffe6dStrategies</i> (<i>ICTCS '16</i>). Association for Computing Machinery, New York, NY, USA, Article 21, 1–5. https://doi.org/10.1145/2905055.2905079
