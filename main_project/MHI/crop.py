import cv2  
import os

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if (img is not None):
            images.append(img)
    return images

images = load_images("main_project/MHI/data/test/Pushup")

for input_image in images:   
    input_image_cpy = input_image.copy()
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    ret, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)
    inverted_binary_img = ~ binary_img

    contours_list, hierarchy = cv2.findContours(inverted_binary_img,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE) 

    i = 0
    print("len: ", len(contours_list))
    for c in contours_list:
        x, y, w, h = cv2.boundingRect(c)
        #print("Coordinate ", i, ": ", x, x+w, y, y+h)
        #print("Area ", i, ": ", cv2.contourArea(c))

        if (cv2.contourArea(c) > 50000  and cv2.contourArea(c) < 2000000 and w>200 and h> 200):
            
            cv2.rectangle(input_image_cpy, (x-50, y-50), (x + w+50, y + h+50), (0, 0, 255), 5)
            print("Coordinate ", i, ": ", x-50, x+w+50, y-50, y+h+50)
            print("Area ", i, ": ", cv2.contourArea(c))
            i+=1

    cv2.imshow('Skeleton bounding box', input_image_cpy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()