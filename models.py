import tensorflow as tf


def gen_block(x, include_bn=True, CNN=tf.keras.layers.Conv2D):
    x_ = x
    x = CNN(64, 3, padding='same')(x)

    if include_bn: x = tf.keras.layers.BatchNormalization(renorm=True)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
    x = CNN(64, 3, padding='same')(x)

    if include_bn: x = tf.keras.layers.BatchNormalization(renorm=True)(x)
    x = CNN(64, 3, padding='same')(x)
    x = tf.keras.layers.Add()([x_,x])

    return x


def get_generator(include_bn=True, separable_cnn=False):

    if separable_cnn:
        CNN = tf.keras.layers.SeparableConv2D
    else:
        CNN = tf.keras.layers.Conv2D

    input = tf.keras.Input(shape=(None,None,3))
    output = CNN(64, 9, padding='same')(input)
    output = tf.keras.layers.PReLU(shared_axes=[1,2])(output)
    output_ = output

    for _ in range(5): output = gen_block(output, include_bn, CNN)

    output = CNN(64, 3, padding='same')(output)
    if include_bn: output = tf.keras.layers.BatchNormalization(renorm=True)(output)

    output = tf.keras.layers.Add()([output, output_])
    output = tf.keras.layers.Conv2D(256, 3, padding='same')(output)
    output = tf.nn.depth_to_space(output, 2)
    output = tf.keras.layers.PReLU(shared_axes=[1,2])(output)

    output = tf.keras.layers.Conv2D(256, 3, padding='same')(output)
    output = tf.nn.depth_to_space(output, 2)
    output = tf.keras.layers.PReLU(shared_axes=[1,2])(output)
    output = tf.keras.layers.Conv2D(3, 9, padding='same')(output)
    
    return tf.keras.models.Model(input, output)


def dis_block(x,k,n,s,include_bn=True):
  x= tf.keras.layers.Conv2D(n, k, strides=s, padding='same')(x)

  if include_bn: x = tf.keras.layers.BatchNormalization(renorm=True)(x)
  x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
  
  return x


def get_discriminator(include_bn=True):
    input = tf.keras.Input(shape=(96,96,3))
    output = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(input)
    output = tf.keras.layers.LeakyReLU(alpha=0.2)(output)

    output = dis_block(output,3,64,2,include_bn)
    output = dis_block(output,3,128,1,include_bn)
    output = dis_block(output,3,128,2,include_bn)
    output = dis_block(output,3,256,1,include_bn)
    output = dis_block(output,3,256,2,include_bn)
    output = dis_block(output,3,512,1,include_bn)
    output = dis_block(output,3,512,2,include_bn)

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