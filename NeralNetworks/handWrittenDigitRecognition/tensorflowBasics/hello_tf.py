import tensorflow as tf

x1 = tf.constant([5]) #designing the computation graph
x2 = tf.constant([6]) #designing the computation graph

print(x1*x2) #abstract tensor with no value computed

with tf.Session() as sess: #starting a session to perform computation and close that session when computation is finished
    output = sess.run(tf.multiply(x1,x2)) #valid one dimensional multiplication..'output' is a python variable that is accessible from outside of the session
    print(output)

print(output) #accessing as a python variable from outside of session


