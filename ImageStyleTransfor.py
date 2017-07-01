from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import model
import reader
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from preprocessing import preprocessing_factory

app = Flask(__name__)
app.config['SECRET_KEY'] = '123456'
app.static_folder = 'static'
UPLOAD_FOLDER = 'static/images/uploadImage/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

LOSS_MODEL="vgg_16"  #使用的卷积神经网络

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploadImage")
def uploadImage():
    return render_template("uploadImage.html");

@app.route('/transform', methods=['GET', 'POST'])
def dealImage():
    models_dict = {'cubist': 'cubist.ckpt-done',
                   'denoised_starry': 'denoised_starry.ckpt-done',
                   'feathers': 'feathers.ckpt-done',
                   'mosaic': 'mosaic.ckpt-done',
                   'scream': 'scream.ckpt-done',
                   'udnie': 'udnie.ckpt-done',
                   'wave': 'wave.ckpt-done',
                   'painting': 'painting.ckpt-2000',
                   }
    if request.method == 'POST':
        file = request.files['image']
        style = request.form['style']
        if file and allowed_file(file.filename):
            if os.path.exists(app.config['UPLOAD_FOLDER']) is False:
                os.makedirs(app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            model_file = 'wave.ckpt-done'
            if style != '':
                if models_dict[style] != '':
                    model_file = models_dict[style]
            style_transform(style, 'models/' + model_file, os.path.join(app.config['UPLOAD_FOLDER']) + file.filename,
                            style + '_' + file.filename)
            return render_template('result.html', original='images/uploadImage/' +file.filename ,
                               result='images/result/' + style + '_' + file.filename)
        return 'transform error:file format error'
    return 'transform error:method not post'

def style_transform(style, model_file, img_file, result_file):
    height = 0
    width = 0
    with open(img_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if img_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(LOSS_MODEL,is_training=False)
            image = reader.get_image(img_file, height, width, image_preprocessing_fn)
            image = tf.expand_dims(image, 0)
            generated = model.transform_network(image, training=False)
            generated = tf.squeeze(generated, [0])
            saver = tf.train.Saver(tf.global_variables())
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            model_file = os.path.abspath(model_file)
            saver.restore(sess, model_file)

            start_time = time.time()
            generated = sess.run(generated)
            generated = tf.cast(generated, tf.uint8)
            end_time = time.time()
            print('Elapsed time: %fs' % (end_time - start_time))
            generated_file = 'static/images/result/' + result_file
            if os.path.exists('static/images/result') is False:
                os.makedirs('static/images/result')
            with open(generated_file, 'wb') as img:
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                print('Done. Please check %s.' % generated_file)

if __name__ == '__main__':
    app.run(debug=True)