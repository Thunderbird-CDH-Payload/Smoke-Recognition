import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import image_processing_functions as IPF

#number of color channels
CH = 3
BS = 1
UNIT_STRIDE = [1,1,1,1]
MAX_STEPS = 100
KERNEL_SHAPES = [(5,5,CH,3),(5,5,3,3),(5,5,3,1)]
LR_KEY = 'learning_rate'
LR_VALUE = 0.001

def placeholder_inputs(imageX,imageY):
    image_placeholder = tf.placeholder(tf.float32,(BS,imageX,imageY,CH))
    target_placeholder = tf.placeholder(tf.bool,(BS,imageX,imageY))
    return image_placeholder,target_placeholder

def fill_feed_dict(image,target,image_pl,target_pl):
    #assert image.shape[0] == target.shape[0]
    #assert image.shape[1] == target.shape[1]
    feed_dict = {
            image_pl : image,
            target_pl : target
            }
    
    return feed_dict

def init_vars():
    kernel_0 = tf.Variable(tf.zeros(KERNEL_SHAPES[0],dtype=tf.float32),name='k1')
    kernel_1 = tf.Variable(tf.zeros(KERNEL_SHAPES[1],dtype=tf.float32),name='k2')
    kernel_2 = tf.Variable(tf.zeros(KERNEL_SHAPES[2],dtype=tf.float32),name='k3')
    
    model_parameters = kernel_0,kernel_1,kernel_2
    return model_parameters

def inference(image,model_parameters):
    kernel_0,kernel_1,kernel_2 = model_parameters
    
    def transform(state , kernel):
        conv_out = tf.nn.conv2d(state,kernel,UNIT_STRIDE,'SAME')
        output = tf.nn.elu(conv_out)
        return output
    
    hidden_0 = transform(image,kernel_0)
    hidden_1 = transform(hidden_0,kernel_1)
    hidden_2 = transform(hidden_1,kernel_2)
    output = tf.sigmoid(hidden_2)
    return output

def loss(output,target):
    output = tf.squeeze(output,squeeze_dims=[3])
    all_true_probability = output
    all_false_probability = tf.sub(tf.constant(1,dtype=tf.float32),output)
    tf.squeeze(target,squeeze_dims=[0])
    actual_probability = tf.select(target,all_true_probability,all_false_probability)
    log_probability = tf.log(actual_probability)
    total_log_prob = tf.reduce_sum(log_probability,name='log_loss')
    total_log_loss = tf.neg(total_log_prob)
    
    return total_log_loss

def train(loss,learning_params):

    learning_rate = learning_params()[LR_KEY]
    print(learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    
    return train_op

def learning_params():
    d = dict()
    d[LR_KEY] = tf.constant(LR_VALUE,dtype=tf.float32)
    return d


def run_training(data):
    image,target = data

    with tf.Graph().as_default():
        images_placeholder, target_placeholder = placeholder_inputs(image.shape[1],image.shape[2])

        feed_dict = fill_feed_dict(image,target,images_placeholder,target_placeholder)

        sess = tf.Session()

        model_parameters = init_vars()
        
        probability_distribution = inference(images_placeholder,model_parameters)
        log_loss = loss(probability_distribution,target_placeholder)
        train_op = train(log_loss,learning_params)
        
        saver = tf.train.Saver()

        init = tf.initialize_all_variables()
        sess.run(init)

        for step in range(MAX_STEPS):
            out = sess.run(train_op,feed_dict=feed_dict)
            print(step)

        save_path = saver.save(sess, "model.ckpt")
        print("Model saved in file: %s" % save_path)

def run_inference(data):
    image,target = data

    with tf.Graph().as_default():
        images_placeholder, target_placeholder = placeholder_inputs(image.shape[1],image.shape[2])

        feed_dict = fill_feed_dict(image,target,images_placeholder,target_placeholder)

        sess = tf.Session()

        model_parameters = init_vars()
        
        probability_distribution = inference(images_placeholder,model_parameters)
        log_loss = loss(probability_distribution,target_placeholder)

        saver = tf.train.Saver()
        saver.restore(sess, "model.ckpt")

        pd = sess.run(probability_distribution,feed_dict=feed_dict)
        print(type(pd))
        print(sess.run(log_loss,feed_dict=feed_dict))

        return pd


        
        

def prep_data(data_address,target_address):
    #assert(len(data,targets))
    image = np.expand_dims(np.array(IPF.getImage(data_address),dtype='float32'),axis=0)
    target = np.expand_dims(np.array(IPF.getLabelImage(target_address),dtype='float32'),axis=0)
    print(image.shape)
    print(target.shape)
    return image,target
    
######################################################################
######################################################################
######################################################################
a = '../MODISfires/data/1.jpg'
b = '../MODISfires/targets/1.png'
data = prep_data(a,b)
run_training(data)
pd = run_inference(data)
img = np.squeeze(pd)
print(pd)
print(img.shape)

imgplot = plt.imshow(img)
plt.show()
