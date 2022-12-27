import tensorflow as tf

def gen_block(x):
    x_ = x
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.5)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.5)(x)

    return x + x_

def get_generator():
    input = tf.keras.Input(shape=(None,None,3))
    output = tf.keras.layers.Conv2D(64, 9, padding='same')(input)
    output = tf.keras.layers.PReLU(shared_axes=[1,2])(output)
    output_ = output

    for _ in range(5): output = gen_block(output)

    output = tf.keras.layers.Conv2D(64, 3, padding='same')(output)
    output = tf.keras.layers.BatchNormalization(momentum=0.5)(output) + output_

    output = tf.keras.layers.Conv2D(256, 3, padding='same')(output)
    output = tf.keras.layers.PReLU(shared_axes=[1,2])(output)
    output = tf.nn.depth_to_space(output, 2)

    output = tf.keras.layers.Conv2D(256, 3, padding='same')(output)
    output = tf.keras.layers.PReLU(shared_axes=[1,2])(output)
    output = tf.nn.depth_to_space(output, 2)

    output = tf.keras.layers.Conv2D(3, 9, padding='same')(output)
    
    return tf.keras.models.Model(input, output)

def dis_block(x,k,n,s):
  x= tf.keras.layers.Conv2D(n, k, strides=s, padding='same')(x)
  x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
  x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
  
  return x

def get_discriminator():
    input = tf.keras.Input(shape=(224,224,3))
    output = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(input)
    output = tf.keras.layers.LeakyReLU(alpha=0.2)(output)

    output = dis_block(output,3,64,2)
    output = dis_block(output,3,128,1)
    output = dis_block(output,3,128,2)
    output = dis_block(output,3,256,1)
    output = dis_block(output,3,256,2)
    output = dis_block(output,3,512,1)
    output = dis_block(output,3,512,2)

    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dense(1024)(output)
    output = tf.keras.layers.LeakyReLU(alpha=0.2)(output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)
    
    return tf.keras.Model(input, output)

def get_feature_extractor(include_preprocess = False, out_layer = 20):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg = tf.keras.Model(vgg.input, vgg.layers[out_layer].output)
    input = tf.keras.Input((None,None,3))
    
    if include_preprocess: input = tf.keras.applications.vgg19.preprocess_input(input)
    output = vgg(input)

    return tf.keras.Model(input, output)