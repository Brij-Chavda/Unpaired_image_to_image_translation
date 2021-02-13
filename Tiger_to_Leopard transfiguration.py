#!/usr/bin/env python
# coding: utf-8

# # Generator

# In[41]:


import random
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Flatten
import matplotlib.pyplot as plt


# In[42]:


IMG_WIDTH = 256
IMG_HEIGHT = 256


# In[3]:


tiger, lion, leopard = [], [], []
for i in range(1018):
    try:
        file_name = 'tiger_c/'+str(i)+'.jpg'
        img = cv2.imread(file_name)
        tiger.append(img)
    except:
        continue


# In[4]:


for i in range(1114):
    try:
        file_name = 'lion_c/'+str(i)+'.jpg'
        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        lion.append(img)
    except:
        continue


# In[5]:


for i in range(820):
    try:
        file_name = 'leopard_c/'+str(i)+'.jpg'
        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        leopard.append(img)
    except:
        continue


# In[6]:


print(np.shape(leopard))
print(np.shape(lion))
print(np.shape(tiger))


# In[33]:


train_tiger, test_tiger = tiger[:820], tiger[820:-1]
train_lion = lion[:820]
train_leopard = leopard[:820]


# In[11]:


def random_crop(image):
    cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image


# In[13]:


def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


# In[20]:


def random_jitter(image):
    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)

    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image


# In[15]:


def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize(image)
    return image
def preprocess_image_test(image):
    image = tf.image.resize(image, [256, 256],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = normalize(image)
    return image


# In[34]:


train_data, test_data = [], []
for i in range(820):
    train_data.append((preprocess_image_train(train_tiger[i]), preprocess_image_train(train_lion[i]), preprocess_image_train(train_leopard[i])))

for i in range(len(test_tiger)):
    test_data.append(preprocess_image_test(test_tiger[i]))

train_tiger, test_tiger, train_lion, train_leopard, tiger, lion, leopard = [], [], [], [], [], [], []

# In[40]:



tiger_sample1 = test_data[12]
tiger_sample2 = test_data[15]
tiger_sample3 = test_data[23]


# In[44]:


def generate_images(model, test_input, j):
    test_input = tf.reshape(test_input, [1,256,256,3])
    prediction1 = model(test_input)
    #prediction2 = model(test_input)

    fig = plt.figure(figsize=(16, 16))

    display_list = [test_input[0], prediction1[0]]
    title = ['Input Image', 'Predicted Image1', 'Predicted Image2']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        tmp = np.array((display_list[i] * 0.5 + 0.5))
        plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
        #plt.imshow(display_list[i] * 0.5 + 0.5)
    file_name = 'output/full_figure'+str(j)+'.png'
    fig.savefig(file_name)
        


# In[45]:


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(tfa.layers.InstanceNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


# In[46]:


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(tfa.layers.InstanceNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result


# In[51]:


from tensorflow.keras.layers import concatenate


# In[80]:


def Generator():
    inputs_img = tf.keras.layers.Input(shape=[256,256,3])
    
    down_stack = [
    downsample(64, 4, apply_batchnorm=False), 
    downsample(128, 4), 
    downsample(256, 4), 
    downsample(512, 4), 
    downsample(512, 4), 
    downsample(512, 4), 
    downsample(512, 4), 
    downsample(512, 4), 
    ]

    up_stack = [
    upsample(512, 4, apply_dropout=True), 
    upsample(512, 4, apply_dropout=True), 
    upsample(512, 4, apply_dropout=True), 
    upsample(512, 4), 
    upsample(256, 4), 
    upsample(128, 4), 
    upsample(64, 4), 
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh') 
    x = inputs_img
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs_img , outputs=x)


# In[81]:

LAMBDA = 10
OUTPUT_CHANNELS = 3
BATCH_SIZE = 1
generator_g = Generator()
generator_f = Generator()


# In[ ]:





# # Discriminator

# In[3]:





# In[58]:


def Discriminator(in_shape = (256,256,3), n_classes = 3):
    initializer = tf.random_normal_initializer(0., 0.02)
    image = tf.keras.layers.Input(shape=in_shape)
    down1 = downsample(64, 4, False)(image) 
    down2 = downsample(128, 4)(down1) 
    down3 = downsample(256, 4)(down2) 
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) 
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1) 
    norm1 = tfa.layers.InstanceNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) 
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2) 
    return tf.keras.Model(inputs=image, outputs = last)


# In[59]:


discriminator_x = Discriminator()
discriminator_y = Discriminator()

# In[60]:


loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# In[61]:


def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


# In[62]:


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


# In[63]:


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss1


# In[64]:


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


# In[65]:


from tensorflow.keras.losses import categorical_crossentropy


# In[66]:


def classification_loss(true_label, predicted_label):
    loss = tf.keras.losses.categorical_crossentropy(true_label, predicted_label)
    return loss


# In[67]:


generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)#change to two
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)#change to two


# In[68]:


checkpoint_path = "./checkpoints_normal/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')


# In[ ]:





# In[89]:


# label_x must be target label and labe_y must be label of tiger [0,0,1]
def reshape_(x):
    x = tf.reshape(x, [1,1,1,3])
    return x

def _reshape_(x):
    x = tf.reshape(x, [1,3])
    return x

label_y, label_x1, label_x2, null_label = [0.0,0.0,1.0], [0.0,1.0,0.0], [1.0,0.0,0.0], [0.0,0.0,0.0]
label_y, label_x1, label_x2, null_label = reshape_(label_y), reshape_(label_x1), reshape_(label_x2), reshape_(null_label)
label_y_, label_x1_, label_x2_, null_label_ = [0.0,0.0,1.0], [0.0,1.0,0.0], [1.0,0.0,0.0], [0.0,0.0,0.0]
label_y_, label_x1_, label_x2_, null_label_ = _reshape_(label_y), _reshape_(label_x1), _reshape_(label_x2), _reshape_(null_label)

@tf.function
def train_step(real_x, real_y2, real_y, label_f1, label_f2):
  
    
    with tf.GradientTape(persistent=True) as tape:
        
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)
        


        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)
        #same_y2 = generator_g([real_y2, label_x2], training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)
        #disc_real_y2 = discriminator_y([real_y2, label_x2_], training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        #disc_fake_x2 = discriminator_x([fake_x2, label_y_], training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)
        #disc_fake_y2 = discriminator_y([fake_y2, label_x2_], training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        #gen_g_loss = gen_g_loss + generator_loss(disc_fake_y2)
        gen_f_loss = generator_loss(disc_fake_x)
        #gen_f_loss = gen_f_loss + generator_loss(disc_fake_x2)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y) 
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)  
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)  

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))
    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss


# In[90]:

def gen_label(r):
    if r == 1:
        return label_y_
    elif r == 2:
        return label_x1_
    elif r == 3:
        return label_x2_

import time
EPOCHS = 100
for epoch in range(EPOCHS):
    start = time.time()
    g_loss, f_loss, x_loss, y_loss = [], [], [], []
    n = 0
    for image_x, image_y, image_z in (train_data):
        r1 = random.randint(1,3)
        r2 = random.randint(1,3)
        label_f1 = gen_label(r1)
        label_f2 = gen_label(r2)
        image_x = tf.expand_dims(image_x, 0)
        image_y = tf.expand_dims(image_y, 0)
        image_z = tf.expand_dims(image_z, 0)
        g, f, x, y  = train_step(image_x, image_y, image_z, label_f1, label_f2)
        if n % 200 == 0:
            print ('.', end = ' ')
        n+=1
        g_loss.append(g)
        f_loss.append(f)
        x_loss.append(x)
        y_loss.append(y)
    print('\n')

    if (epoch + 1) % 5 == 0:
        generate_images(generator_g, tiger_sample1, 1)
        generate_images(generator_g, tiger_sample2, 2)
        generate_images(generator_g, tiger_sample3, 3)
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

    with open('loss.txt', 'a') as file:
        file.write(f'{np.mean(g_loss)}')
        file.write('\n')
        file.write(f'{np.mean(f_loss)}')
        file.write('\n')
        file.write(f'{np.mean(x_loss)}')
        file.write('\n')
        file.write(f'{np.mean(y_loss)}')
        file.write('\n')

# In[43]:



