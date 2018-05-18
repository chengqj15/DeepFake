"""
    Deal with Model
"""
from model import autoencoder_A
from model import autoencoder_B
from model import encoder, decoder_A, decoder_B
import numpy 

videopath = "../../../data/faceswap/video/"
modelpath = "../../../data/faceswap/"

encoder  .load_weights(modelpath + "models/encoder.h5"   )
decoder_A.load_weights(modelpath + "models/decoder_A.h5" )
decoder_B.load_weights(modelpath + "models/decoder_B.h5" )

def convert_one_image( autoencoder, image ):
    assert image.shape == (256,256,3)
    crop = slice(48,208)
    face = image[crop,crop]
    face = cv2.resize( face, (64,64) )
    face = numpy.expand_dims( face, 0 )
    new_face = autoencoder.predict( face / 255.0 )[0]
    new_face = numpy.clip( new_face * 255, 0, 255 ).astype( image.dtype )
    new_face = cv2.resize( new_face, (160,160) )
    new_image = image.copy()
    new_image[crop,crop] = new_face
    return new_image


"""
    deal with video
"""
import cv2
import imageio

face_cascode = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascode.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        return (x, y, w, h)
    return (-1, -1, -1, -1)

reader = imageio.get_reader(videopath + "source.mp4")
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer(videopath + "target.mp4", fps=fps)

for i, frame in enumerate(reader):
    (x, y, w, h) = detect(frame)
    if x == -1:
        writer.append_data(frame)
    else:      
        face = frame[y:y+h, x:x+w]  
        original_shape = (face.shape[0], face.shape[1])
        face = cv2.resize(face, (256, 256))
        face = convert_one_image(autoencoder_B, face)
        face = cv2.resize(face, original_shape)
        frame[y:y+h, x:x+w] = face
        writer.append_data(frame)
        print(i)
    cv2.imwrite('{}{}'.format(videopath, "image.png"), frame)

writer.close() 
reader.close()
print("completed")