
### import all the variables we need for this code to run


```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
from utilities import percentage

from tensorflow.contrib.tensorboard.plugins import projector
```

### set the logdir for tensorboard, this helps us to visualise the word as a multidimensional vector and lets us see close words


```python
current_path=os.path.dirname(os.path.realpath(sys.argv[0]))
parser=argparse.ArgumentParser()
parser.add_argument(
    "--log_dir",
    type=str,
    default=os.path.join(current_path,"log"),
    help="log directory for tensorboard"
)
FLAGS,unparsed=parser.parse_known_args()

if not os.path.exists(FLAGS.log_dir):
    os.path.mkdirs(FLAGS.log_dir)
```

### downloading the dataset, this is whaat we will use to train our model to recognize similarities between words


```python
# step-1 download the data
url="http://mattmahoney.net/dc/"

def maybe_download(filename,expected_bytes):
    """download a file and confirm the size"""
    local_filename=os.path.join(gettempdir(),filename)
    if not os.path.exists(local_filename):
        local_filename,_=urllib.request.urlretrieve(url+filename,local_filename)
        
    statinfo=os.stat(local_filename)
    if statinfo.st_size==expected_bytes:
        print("Found and verified",filename)
    else:
        print(statinfo.st_size)
        raise Exception("Failed to verify "+local_filename+". can you get it with a browser")
    return local_filename


filename=maybe_download('text8.zip',31344016)
```

    Found and verified text8.zip
    


```python
# read the data into a list of strings
def read_data(filename):
    """etract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        return tf.compat.as_str(f.read(f.namelist()[0])).split()
```

### step 2:build the dictionary and replace rare words with UNK token while replacing common words with the first position in whihc they are encountered in



```python
vocabulary_size=50000
vocabulary=read_data(filename)

def build_dataset(words,n_words):
    "process raw inputs  into a dataset"
    unk_count=0
    count=[('UNK',-1)]
    count.extend(collections.Counter(words).most_common(n_words-1))
    dictionary=dict()
    for word,_  in count:
        dictionary[word]=len(dictionary)
    data=list()
    
    for word in words:
        index=dictionary.get(word,0)
        if index==0:
            unk_count+=1
        data.append(index)
    count[0]=("UNK",unk_count)
    reversed_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reversed_dictionary
```


```python
data,count,dictionary,reverse_dictionary=build_dataset(vocabulary,vocabulary_size)
del vocabulary
print("most common words (+unk)",count[5:10])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
data_index=0
```

    most common words (+unk) [('in', 372201), ('a', 325873), ('to', 316376), ('zero', 264975), ('nine', 250430)]
    Sample data [5234, 3081, 12, 6, 195, 2, 3134, 46, 59, 156] ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']
    

### write a function to generate batches of train data since it is too arge to load them all in memory


```python
def generate_batch(batch_size,num_skips,skip_window):
#     create global variables
    global data_index
    assert num_skips<=(skip_window*2)
    assert batch_size%num_skips==0
    
# instantialize local variables
    batch=np.ndarray(shape=(batch_size),dtype=np.int32)
    labels=np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span=(2*skip_window)+1
    buffer=collections.deque(maxlen=span)
    
    
#     conditionals for data_index
    if data_index+span>len(data):
        data_index=0
        
#     initialize buffer
    buffer.extend(data[data_index:data_index+span])
    data_index+=span
    
#     major loop for batch generation
    for i in range(batch_size//num_skips):
        context_words=[word for word in range(span) if word!=skip_window]
        words_to_use=random.sample(context_words,num_skips)
        
#         inner loop to put context words and labels in batch&labels
        for j,word_to_use in enumerate(words_to_use):
            batch[(i*num_skips)+j]=buffer[skip_window]
            labels[(i*num_skips)+j,0]=buffer[word_to_use]
        
#         check the length of buffer array to make sure theres still space
        if data_index==len(data):
            buffer.extend(data[0:span])
            data_index=span
        else:
            buffer.append(data[data_index])
            data_index=data_index+1
    
#     backtrack data index a little to avoid missing words at the end of a batch
    data_index=(data_index+len(data)-span)%len(data) 
    return batch,labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
        reverse_dictionary[labels[i, 0]])
```

    3081 originated -> 5234 anarchism
    3081 originated -> 12 as
    12 as -> 3081 originated
    12 as -> 6 a
    6 a -> 195 term
    6 a -> 12 as
    195 term -> 2 of
    195 term -> 6 a
    

### defination  of  Parameters


```python

batch_size=128
embedding_size=128 #size of the dense embeddding vector
skip_window=1 #number of times to reuse each word
num_skips=2 #number of words to consider both sides
num_sampled=64 #how many negative examples to sample

valid_size=10
valid_window=1000
valid_examples=np.random.choice(valid_window,valid_size,replace=False)
```

### define the graph which we will use for the model


```python
graph=tf.Graph()


with graph.as_default():
    
#     input_data
    with tf.name_scope('inputs'):
        train_inputs=tf.placeholder(dtype=tf.int32,shape=[batch_size])
        train_labels=tf.placeholder(dtype=tf.int32,shape=[batch_size,1])
        valid_dataset=tf.constant(valid_examples,dtype=tf.int32)
        
    with tf.name_scope("embeddings"):
#         look up embeddings for inputs
        embeddings=tf.Variable(
            tf.random_uniform([vocabulary_size,embedding_size],-1.0,1)
        )
        embed=tf.nn.embedding_lookup(embeddings,train_inputs)
        
#         contruct the variables for the NCE loss
    with tf.name_scope("weights"):
        nce_weights=tf.Variable(
            tf.random_normal(
                shape=[vocabulary_size,embedding_size],
                stddev=1.0/math.sqrt(embedding_size)
            ))
    with tf.name_scope("biases"):
        nce_biases=tf.Variable(tf.zeros(shape=[vocabulary_size]))
        
    with tf.name_scope("loss"):
        loss=tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size
            )
        )
    tf.summary.scalar('loss',loss)
    
    with tf.name_scope('optimizer'):
        
        optimizer=tf.train.AdamOptimizer(0.0001).minimize(loss)
#         creatingasaver
    init=tf.global_variables_initializer()
    saver=tf.train.Saver()
    
    norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keepdims=True))
    normalized_embeddings=embeddings/norm
    
    valid_embeddings=tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
    
    similarity=tf.matmul(
        valid_embeddings,
        normalized_embeddings,
        transpose_b=True
    )
    
    merged=tf.summary.merge_all()
```

### train the model, and at each step randomly pick a word and show us the closest word to it


```python
num_steps=100001
percent=0

with tf.Session(graph=graph) as session:
    writer=tf.summary.FileWriter(FLAGS.log_dir,session.graph)
    saver.restore(session,"/temp/tf_models/embedding.ckpt")
#     init.run()
    print("initialized")
    
    average_loss=0
    
    for step in xrange(num_steps):
        batch_inputs,batch_labels=generate_batch(batch_size,num_skips,skip_window)
        feed_dict={train_inputs:batch_inputs,train_labels:batch_labels}
        run_metadata=tf.RunMetadata()
        _,summary,loss_val=session.run([optimizer,merged,loss],feed_dict=feed_dict,run_metadata=run_metadata)
        saver.save(session,"/temp/tf_models/embedding.ckpt")
        average_loss+=loss_val
        percentage(percent,200)
        percent=percent+1
        writer.add_summary(summary,step);
        
        if step ==(num_steps-1):
            writer.add_run_metadata(run_metadata,'step%d'%step)
        
        if not step%200:
            percent=0
            print("epoch:{}  loss:{}".format(step,loss_val))
            similarity_vector=session.run(similarity)
            for i,word in enumerate(valid_examples):
                print("top 5 closest to word to:{} are {}".format(reverse_dictionary[word],[reverse_dictionary[word_index] for word_index in (-similarity_vector[i]).argsort()[1:5+1]]))
        
            
```

    INFO:tensorflow:Restoring parameters from /temp/tf_models/embedding.ckpt
    

    INFO:tensorflow:Restoring parameters from /temp/tf_models/embedding.ckpt
    

    initialized
    epoch:0  loss:42.226627349853516
    top 5 closest to word to:set are ['country', 'beginning', 'settled', 'astronomy', 'theme']
    top 5 closest to word to:see are ['abu', 'articles', 'becomes', 'practice', 'era']
    top 5 closest to word to:movie are ['country', 'final', 'scholars', 'senses', 'duty']
    top 5 closest to word to:language are ['notably', 'whig', 'art', 'banquet', 'rand']
    top 5 closest to word to:al are ['torture', 'delays', 'discoveries', 'mortal', 'god']
    top 5 closest to word to:related are ['classical', 'win', 'secret', 'philosophy', 'discovered']
    top 5 closest to word to:described are ['powers', 'statistical', 'soon', 'result', 'inhabited']
    top 5 closest to word to:won are ['mid', 'help', 'almost', 'troy', 'promote']
    top 5 closest to word to:thomas are ['irritated', 'athenians', 'prime', 'works', 'defeat']
    top 5 closest to word to:close are ['help', 'studied', 'independence', 'behind', 'elected']
    93.50%

#### after training your model, we use its embeddings to generate a word cloud and vsualise it using this function


```python
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom')

  plt.savefig(filename)


try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(
      perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)
```

### we are using a similarty function (cosine similarity to find vectors close to each other), this helps us find words close to ech other based on embeddings
