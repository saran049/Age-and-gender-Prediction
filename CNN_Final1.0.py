# Predict age and gender using Convolutional neural networks.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

img(face_location(2):face_location(4),face_location(1):face_location(3),:))
[age,~]=datevec(datenum(wiki.photo_taken,7,1)-wiki.dob);

# Building the CNN
# Importing the Keras libraries and packages
import numpy
from keras.datasets import cifar100
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = cifar100.load_data()

def read_data(): 
    mat = sio.loadmat(mat_path)
    mdata = mat['wiki'] 
    photo_taken = mdata['photo_taken'][0,0] 
    full_path = mdata['full_path'][0,0] 
    gender = mdata['gender'][0,0] 
    name = mdata['name'][0] 
    dob = mdata['dob'][0,0]
    
# Convert the matlab date format in to python 
p_datetime = [] 
for i in range(0,len(dob[0])): 
    mat_date = int(dob[0,i]) 
    c_date = (date.fromordinal(mat_date) + timedelta(days=mat_date%1) - timedelta(days = 366)) 
    p_datetime.append(c_date)

# store the age for each image in an array.
age = []
for i in range (0, len(p_datetime)):
    age. append(((date(photo_taken[0,i],7,1) - p_datetime[i]).days)/365)
    
def exclude_missing(filenames , gender , age):
    predict_missing = np.argwhere(np.isnan(gender)) 
    data = []
    gen = [] 
    age_d = []
    for i in range(len(filenames)):
        if((i in predict_missing)or(age[i]<0)or(age[i]>110)):
            continue
        else:
            data.append(filenames[i])
            gen.append(gender[i]) 
            age_d.append(age[i])
    return(data , gen, age_d, predict_missing)

# Preparation of training and testing dataset
X_data_train = X_data [0:30000]
X_data_test = X_data [30000:40000]
y_data_gender_train = y_data_gender [0:30000]
y_data_gender_test = y_data_gender [30000:40000]
y_data_age_train = y_data_age [0:30000] 
y_data_age_test = y_data_age [30000:40000]

# Defining parameter for neural network
batch_size = 100
training_epochs = 500 
train_size, _ , _ = X_data_train.shape 
iteration = int(train_size/batch_size) 
test_size,_,_ = X_data_test.shape

# Design CNN
#For Age:
def deepnn_age(x):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1,32])
    b_conv1 = bias_variable([32]) 
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    
    # Pooling layer - down samples by 2X.
    h_pool2 = max_pool_2x2(h_conv2) 
    W_fc1 = weight_variable([7 * 7 * 64, 1024]) 
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) 
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # Dropout - controls the complexity of the model, prevents co-adaptation of 
    # features. 
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # Map the 1024 features to 111 classes, one for each image 
    W_fc2 = weight_variable([1024, 111]) 
    b_fc2 = bias_variable([111]) 
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 
    return y_conv, keep_prob

def conv2d(x, W):
   # """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    #"""max_pool_2x2 down samples a feature map by 2X.""" 
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape): 
    #"""weight_variable generates a weight variable of a given shape.""" 
    initial = tf.truncated_normal(shape, stddev=0.1) 
    return tf.Variable(initial)

def bias_variable(shape): 
    #"""bias_variable generates a bias variable of a given shape.""" 
    initial = tf.constant(0.1, shape=shape) 
    return tf.Variable(initial)

#For Gender
def deepnn_gender(x): 
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    
    #First convolutional layer - maps one grayscale image to 32 feature maps. 
    W_conv1 = weight_variable([5, 5, 1,32]) 
    b_conv1 = bias_variable([32]) 
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    
    # Pooling layer - down samples by 2X. 
    h_pool1 = max_pool_2x2(h_conv1)
    
    # Second convolutional layer -- maps 32 feature maps to 64. 
    W_conv2 = weight_variable([5, 5, 32, 64]) 
    b_conv2 = bias_variable([64]) 
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    
    #Second pooling layer. 
    h_pool2 = max_pool_2x2(h_conv2) 
    W_fc1 = weight_variable([7 * 7 * 64, 1024]) 
    b_fc1 = bias_variable([1024]) 
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) 
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    #Dropout - controls the complexity of the model, prevents co-adaptation of features.
    keep_prob = tf.placeholder(tf.float32) 
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    #Map the 1024 features to 2 classes, one for each image 
    W_fc2 = weight_variable([1024, 2]) 
    b_fc2 = bias_variable([2])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 
    return y_conv, keep_prob

def conv2d(x, W): 
    #"""conv2d returns a 2d convolution layer with full stride.""" 
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x): 
    #"""max_pool_2x2 downsamples a feature map by 2X.""" 
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape): 
    #"""weight_variable generates a weight variable of a given shape.""" 
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape): 
    #"""bias_variable generates a bias variable of a given shape.""" 
    initial = tf.constant(0.1, shape=shape) 
    return tf.Variable(initial)

#Training and Saving the Model

#For Age Prediction
with tf.Graph().as_default() as g: 
    x = tf.placeholder(tf.float32, shape=[None, 28, 28]) 
    y_gender = tf.placeholder(tf.float32, [None, 2]) 
    y_age = tf.placeholder(tf.float32, [None, 111])
    y_conv_age, keep_prob_age = deepnn_age(x)
    cross_entropy_age = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_age, logits=y_conv_age))
    train_step_age = tf.train.AdamOptimizer(0.001).minimize(cross_entropy_age) 
    correct_prediction_age = tf.equal(tf.argmax(y_conv_age, 1), tf.argmax(y_age, 1)) 
    accuracy_age = tf.reduce_mean(tf.cast(correct_prediction_age, tf.float32))
    saver = tf.train.Saver() 
    model_path = 'model_age.ckpt'
    
    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        for e in range(training_epochs):
            idx = np.random.permutation(train_size) 
            x_random = X_data_train[idx]
            y_age_random = y_data_age_train[idx]
            for i in range(iteration):
                xs = x_random[i*batch_size:(i+1)*batch_size] 
                yas = y_age_random[i * batch_size:(i + 1) * batch_size]
                train_accuracy_age = sess.run(accuracy_age, feed_dict={x: xs, y_age: yas, keep_prob_age: 1.0})
                print('step %d,batch %d,training accuracy age %g' % (e,i,train_accuracy_age)) 
                train_step_age.run(feed_dict = {x:xs,y_age:yas,keep_prob_age:0.5})
                
        #saving the model saver.save (sess,model_path) 
        print('Training on age Completed........')
        
#For Gender Prediction
with tf.Session().as_default() as g: 
    x = tf.placeholder(tf.float32, shape=[None, 28, 28]) 
    y_gender = tf.placeholder(tf.float32, [None, 2]) 
    y_age = tf.placeholder(tf.float32, [None, 111])
    y_conv_gen_gender, keep_prob_gen_gen = deepnn_gender(x)
    cross_entropy_gender = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_gender, logits=y_conv_gen_gender))
    train_step_gender = tf.train.AdamOptimizer(0.001).minimize(cross_entropy_gender)
    correct_prediction_gender = tf.equal(tf.argmax(y_conv_gen_gender, 1), tf.argmax(y_gender, 1))
    accuracy_gender = tf.reduce_mean(tf.cast(correct_prediction_gender, tf.float32))
    saver = tf.train.Saver() 
    model_path = 'gender_model.ckpt'
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        for e in range(training_epochs):
            idx = np.random.permutation(train_size) 
            x_random = X_data_train[idx]
            y_gender_random = y_data_gender_train[idx] 
            for i in range(iteration):
                xs = x_random[i*batch_size:(i+1)*batch_size] 
                ygs = y_gender_random[i*batch_size:(i+1)*batch_size]
                train_accuracy_gender = sess.run(accuracy_gender,feed_dict={x: xs, y_gender: ygs, keep_prob_gen_gen: 1.0})
                print('step %d,batch %d, training accuracy gender %g' % (e,i, train_accuracy_gender))
                print('step %d,batch %d, training accuracy gender %g' % (e,i, train_accuracy_gender))
        #saving the model 
        saver.save(sess,model_path)
        print('Training on gender Completed........')
        
#Restore Age model
def r_age(): 
    print(' Age Prediction Started..........................')
    with tf.Graph().as_default() as g: 
        x = tf.placeholder(tf.float32, shape=[None, 28, 28])
        y_age = tf.placeholder(tf.float32, [None, 111]) 
        y_gender = tf.placeholder(tf.float32, [None, 2])
        y_conv_age, keep_prob_age = deepnn_age(x)
        prediction_age = tf.argmax(y_conv_age, 1)
        saver = tf.train.Saver() 
        model_path = 'model_age.ckpt'
        
        with tf.Session() as sess: 
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path_age) 
            print('****** Age Model Restored *****************************')
            age_g = sess.run(prediction_age, feed_dict={x: file_matrrix, keep_prob_age: 0.5}) 
            print(age_g) 
            return(age_g)
        
#Restore Gender model
def r_gender(): 
    print('Gender Prediction Started..........................') 
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, 28, 28]) 
        y_gender = tf.placeholder(tf.float32, [None, 2])
        y_conv_gender, keep_prob_gen = deepnn_gender(x) 
        prediction_gender = tf.argmax(y_conv_gender, 1)
        saver = tf.train.Saver() 
        model_path = 'gender_model.ckpt'
        with tf.Session() as sess: 
            sess.run(tf.global_variables_initializer())
            saver.restore(sess,model_path_gender) 
            print('****** Gender Model Restored *******************************')
            gen = sess.run(prediction_gender,feed_dict={x: file_matrrix ,keep_prob_gen: 0.5}) 
            return gen
    
    
        

    
    

