
from lshash import LSHash
import pandas as pd
import numpy as np

k = 10 # hash size
L = 5  # number of tables
d = 64
#4096

lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=L)

# load your data. in my case, it's a map with keys being file-paths, and values being d-dimensional vectors.
#features = load_features(...)

features =pd.read_csv("/home/kapitsa/pyCharm/segmentation/Convolutional-Encoder-Decoder-for-Hand-Segmentation-master/paper2/test_images//1.csv")
#del features.row.name

features.reset_index(inplace=True) # Resets the index, makes factor a column
features.drop("index",axis=1,inplace=True) # drop factor from axis 1 and make changes permanent by inplace=True

#print(features)

# indexing
# for path, vec in features.iteritems():
#     lsh.index(vec, extra_data=path)

for vec in features.iterrows():

    vec=np.random.randint(10)*np.random.random(d)
    #print "\n\t  vec=\n",vec
    lsh.index(vec)

# query a vector q_vec
response = lsh.query(vec,num_results=3,distance_func='euclidean')
#print("\n\t response=",response)

print "\n\t query",vec
for indx,ele in enumerate(response):
    print "\n\t indx=",indx,"\n\t\t ele=",ele[0] #,"\t type=",type(ele[0])
    print "\n\t\t indx=>",indx,"\n\t\t dist=",ele[1]#,"\t type=",type(ele[0])


