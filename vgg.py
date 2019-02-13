import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
from sklearn.model_selection import train_test_split
import numpy as np
import tflearn.datasets.oxflower17 as oxflower17
X, Y= oxflower17.load_data(one_hot=True)


x_train, x_test_pre, y_train, y_test_pre = train_test_split(X, Y, test_size=0.20, random_state=42)
x_test, x_validation, y_test, y_validation = train_test_split(x_test_pre, y_test_pre, test_size=0.1)



learning_rate = 0.001
training_iters = 200000


n_input = 224*224*3
n_classes = 17
dropout = 0.75 

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, 17])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


batch_size = 2
epochs = 1
progress = 40
n_classes = 17

def conv2d(name,x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x,name=name)  


def maxpool2d(name,x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME',name=name)    




weights = {
    'wb11': tf.Variable(tf.random_normal([3, 3, 3, 64])),
    'wb12': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'wb21': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wb22': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'wb31': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wb32': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'wb33': tf.Variable(tf.random_normal([1, 1, 256, 256])),
    'wb41': tf.Variable(tf.random_normal([3, 3, 256, 512])),
    'wb42': tf.Variable(tf.random_normal([3, 3, 512, 512])),
    'wb43': tf.Variable(tf.random_normal([1, 1, 512, 512])),
    'wb51': tf.Variable(tf.random_normal([3, 3, 512, 512])),
    'wb52': tf.Variable(tf.random_normal([3, 3, 512, 512])),
    'wb53': tf.Variable(tf.random_normal([1, 1 ,512, 512])),
    'wd1': tf.Variable(tf.random_normal([7*7*512, 4096])),
    'wd2': tf.Variable(tf.random_normal([4096, 4096])),
    'wd3': tf.Variable(tf.random_normal([4096, 1000])),
    'out': tf.Variable(tf.random_normal([1000, n_classes]))
}
biases = {
    'bb11': tf.Variable(tf.random_normal([64])),
    'bb12': tf.Variable(tf.random_normal([64])),
    'bb21': tf.Variable(tf.random_normal([ 128])),
    'bb22': tf.Variable(tf.random_normal([ 128])),
    'bb31': tf.Variable(tf.random_normal([ 256])),
    'bb32': tf.Variable(tf.random_normal([ 256])),
    'bb33': tf.Variable(tf.random_normal([ 256])),
    'bb41': tf.Variable(tf.random_normal([ 512])),
    'bb42': tf.Variable(tf.random_normal([ 512])),
    'bb43': tf.Variable(tf.random_normal([ 512])),
    'bb51': tf.Variable(tf.random_normal([ 512])),
    'bb52': tf.Variable(tf.random_normal([ 512])),
    'bb53': tf.Variable(tf.random_normal([ 512])),
    'bd1': tf.Variable(tf.random_normal([ 4096])),
    'bd2': tf.Variable(tf.random_normal([4096])),
    'bd3': tf.Variable(tf.random_normal([ 1000])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def vgg_net(x, weights,biases,droput):
    x = tf.reshape(x, shape = [-1,224,224,3])
    #input_shape=(224,224,3)
    #block 1
    conv1_1 = conv2d('conv1_1',x, weights['wb11'],biases['bb11'])
    conv1_2 = conv2d('conv1_2',conv1_1, weights['wb12'],biases['bb12'])
    pool1 = maxpool2d('pool1',conv1_2,k=2) #pooling 
#                                             ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
    #block 2
    conv2_1 = conv2d('conv2_1',pool1, weights['wb21'],biases['bb21'])
    conv2_2 = conv2d('conv2_2',conv2_1, weights['wb22'],biases['bb22'])
    pool2 = maxpool2d('pool2',conv2_2,k=2) #pooling 
    
    #block 3
    conv3_1 = conv2d('conv3_1',pool2, weights['wb31'],biases['bb31'])
    conv3_2 = conv2d('conv3_2',conv3_1, weights['wb32'],biases['bb32'])
    conv3_3 = conv2d('conv3_3',conv3_2, weights['wb33'],biases['bb33'])
    pool3 = maxpool2d('pool3',conv3_3,k=2) #pooling

    #blocl 4
    conv4_1 = conv2d('conv4_1',pool3, weights['wb41'],biases['bb41'])
    conv4_2 = conv2d('conv4_2',conv4_1, weights['wb42'],biases['bb42'])
    conv4_3 = conv2d('conv4_3',conv4_2, weights['wb43'],biases['bb43'])
    pool4 = maxpool2d('pool4',conv4_3,k=2) #pooling

    #block 5
    conv5_1 = conv2d('conv5_1',pool4, weights['wb51'],biases['bb51'])
    conv5_2 = conv2d('conv5_2',conv5_1, weights['wb52'],biases['bb52'])
    conv5_3 = conv2d('conv5_3',conv5_2, weights['wb53'],biases['bb53'])
    pool5 = maxpool2d('pool5',conv5_3,k=2) #pooling

    print('*----------------*-*-*-*--*',pool5.shape)

    flattened_shape = np.prod([s.value for s in pool5.get_shape()[1:]])

    fc = tf.reshape(pool5, [-1, flattened_shape], name="flatten")
    
    # Full 1
    
    print('*----------------*-*-*-*--*',fc.shape)
    print('*----------------*-*-*-*--*',weights['wd1'].shape)
    fc1 =tf.add(tf.matmul(fc, weights['wd1']),biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # dropout
    fc1=tf.nn.dropout(fc1,dropout)

    # Full 2
    fc2 =tf.add(tf.matmul(fc1, weights['wd2']),biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # dropout
    fc2=tf.nn.dropout(fc2,dropout)

    # Full 3
    fc3 =tf.add(tf.matmul(fc2, weights['wd3']),biases['bd3'])
    fc3 = tf.nn.relu(fc3)
    # dropout
    fc3=tf.nn.dropout(fc3,dropout)

    out = tf.add(tf.matmul(fc3, weights['out']) ,biases['out'])
    return out


pred = vgg_net(x, weights, biases, keep_prob)


# (6) Define model's cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# (7) Defining evaluation metrics
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy_pct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100


init = tf.global_variables_initializer()

epochs = 1
with tf.Session(config = config) as sess:
    sess.run(init)
    
    print("Training for", epochs, "epochs.")
    
    # looping over epochs: 
    for epoch in range(epochs):
        # To monitor performance during training
        avg_cost = 0.0 
        avg_acc_pct = 0.0
        
        # loop over all batches of the epoch- 1088 records  
        # batch_size = 128 is already defined
        n_batches = int(1088 / batch_size) 
        counter = 1
        for i in range(n_batches):
            print(i)
            # Get the random int for batch
            random_indices = np.random.randint(1088, size=batch_size) # 1088 is the no of training set records
            
            feed = {
                x: x_train[random_indices],
                y: y_train[random_indices]
            }
            
            # feed batch data to run optimization and fetching cost and accuracy: 
            _, batch_cost, batch_acc = sess.run([optimizer, cost, accuracy_pct], 
                                                   feed_dict=feed)
            # Print batch cost to see the code is working (optional)
            # print('Batch no. {}: batch_cost: {}, batch_acc: {}'.format(counter, batch_cost, batch_acc))
            # Get the average cost and accuracy for all batches: 
            avg_cost += batch_cost / n_batches
            avg_acc_pct += batch_acc / n_batches
            counter += 1
        
        # Get cost and accuracy after one iteration
        test_cost = cost.eval({x: x_test, y: y_test})
        test_acc_pct = accuracy_pct.eval({x: x_test, y: y_test})
        # output logs at end of each epoch of training:
        print("Epoch {}: Training Cost = {:.3f}, Training Acc = {:.2f} -- Test Cost = {:.3f}, Test Acc = {:.2f}"\
              .format(epoch + 1, avg_cost, avg_acc_pct, test_cost, test_acc_pct))

   