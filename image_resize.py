import cv2 
import os 


def img_resize(srcpath, filename):
    image = cv2.imread('{}/{}'.format(srcpath, filename))
    image = cv2.resize(image, (256, 256))
    cv2.imwrite('{}/{}'.format(srcpath, filename), image)
    print(filename)


def main():
    for srcpath, _, files in os.walk('../../../data/faceswap/face/'):
        if len(_):
            continue
        for filename in files:
            if filename.startswith('.'):
                continue
            try:
                img_resize(srcpath, filename)
            except:
                continue

if __name__ == '__main__':
    main()