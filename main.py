import cv2
from deepface import DeepFace
import streamlit as st
from PIL import Image
import numpy as np

### PIL型 => OpenCV型　の変換関数
def pil2opencv(in_image):
    out_image = np.array(in_image, dtype=np.uint8)
    if out_image.ndim == 2:
        pass
    elif out_image.shape[2] == 3:
        out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
    return out_image

### OpenCV型 => PIL型　の変換関数
def opencv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    new_image = Image.fromarray(new_image)
    return new_image

uploaded_file = st.file_uploader("choose an image...",type='JPEG')
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img,caption='uploaded image.' ,use_column_width=True)

    img2 = pil2opencv(img)

    result = DeepFace.analyze(img2,actions=['emotion'])

    st.write(img2.shape)
    st.write(img2.shape[0])
    img3 = img2.copy()
    for i in range(len(result)):
        x = result[i]['region']['x']
        y = result[i]['region']['y']
        w = result[i]['region']['w']
        h = result[i]['region']['h']

        top = 0 if y-h/2<0 else int(y-h/2)
        bottom = img2.shape[0] if y+h+h/2 > img2.shape[0] else int(y+h+h/2)
        left = 0 if x-w/2 <0 else int(x-w/2)
        right = img2.shape[1] if x+w+w/2 > img2.shape[1] else int(x+w+w/2)

        img3 = cv2.rectangle(img3, (x,y), (x+w,y+h), (0, 255, 0), 3)

        # 一人ずつ分析
        res = result[i]['emotion']
        emotion = sorted(res.items(), key=lambda x:x[1], reverse=True)
        st.write('emotion ranking')
        st.write('1st:' ,emotion[0] ,',')
        st.write('2nd:' ,emotion[1] ,',')
        st.write('3rd:' ,emotion[2])

        # 一人ずつ切り出す
        # img[top : bottom, left : right]
        img4 = img2[top: bottom, left: right]
        img5 = opencv2pil(img4)

        # 一人ずつのやつまとめ
        st.image(img5,caption='person.' ,use_column_width=True)

    # 全員分をまとめて表示
    img3 = opencv2pil(img3)
    st.image(img3,caption='uploaded image.' ,use_column_width=True)
