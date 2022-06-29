import cv2
import uuid
import os
import time

IMAGES_PATH = os.path.join('data', 'images') # 현재 위치에 data/images폴더 미리 만들어 주세요
labels = ['scrollup', 'scrolldown'] # 학습시킬 classes
number_imgs = 5 # label 당 몇 장의 사진을 찍을 것인지

cap = cv2.VideoCapture(0) # 웹캠을 받아옴

# labels를 순환
for label in labels:
    print('Collecting images for {}'.format(label))
    time.sleep(5) # 실제로 사진이 찍히기 전, 잠깐 멈춤

    # 설정한 number_imgs만큼 사진 수집
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))

        # frame 읽어오기
        ret, frame = cap.read()

        # 유니크한 id를 주고 미리 지정한 IMAGES_PATH로
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')

        # 읽어들인 frame을 지정한 path에 저장
        cv2.imwrite(imgname, frame)

        # 어떻게 찍혔는지 보여줌
        cv2.imshow('Data Collection', frame)

        # 2초 대기
        time.sleep(2)

        # q 누르면 나감
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
# 자원 해제
cap.release()
cv2.destroyAllWindows()
