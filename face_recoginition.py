import cv2 
import os 

# Loading the cascades 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detect_faces(srcpath, dstpath, filename):
    image = cv2.imread('{}/{}'.format(srcpath, filename))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    # Extract when just one face is detected
    print('{}/{} {}'.format(srcpath, filename, len(faces) == 1))
    if (len(faces) == 1):
        (x, y, w, h) = faces[0]
        image = image[y:y+h, x:x+w]
        image = cv2.resize(image, (256, 256))
        cv2.imwrite('{}/{}'.format(dstpath, filename), image)


def main():
    for srcpath, _, files in os.walk('../../../data/faceswap/original/'):
        if len(_):
            continue
        dstpath = srcpath.replace('original', 'face')
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        for filename in files:
            if filename.startswith('.'):
                continue
            try:
                detect_faces(srcpath, dstpath, filename)
            except:
                continue

if __name__ == '__main__':
    main()