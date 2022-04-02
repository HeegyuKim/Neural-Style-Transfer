import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image as Image


@st.cache(allow_output_mutation=True)
def get_transfer_model():
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
    return hub_module

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def load_img(data, max_dim = 512):
    img = tf.image.decode_image(data, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):
    st.subheader(title)
    st.image(image)
    
st.title("Neural Style Transfer Demo")

image_types = ["png", "jpg"]
base_file = st.file_uploader("Content 이미지 선택", type=image_types)

if base_file is None:
    BASE_EXAMPLE = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
    base_file = tf.io.read_file(BASE_EXAMPLE)
else:
    base_file = base_file.read()
    
content_image = load_img(base_file)
imshow(tensor_to_image(content_image), "Content Image")

    
style_file = st.file_uploader("Style 이미지 선택", type=image_types)
if style_file is None:
    STYLE_EXAMPLE = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
    style_file = tf.io.read_file(STYLE_EXAMPLE)
else:
    style_file = style_file.read()
style_image = load_img(style_file)
imshow(tensor_to_image(style_image), "Style Image")


with st.spinner("Transferring..."):
    hub_module = get_transfer_model()
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    imshow(tensor_to_image(stylized_image), "Style Transfer Result")
