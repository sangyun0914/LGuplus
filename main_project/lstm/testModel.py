from configs import *
from model_v3 import Model
import extraFeatures
import count
from count import action_count_length
from extract_v3 import getParts

seq_length = config['seq_length']
data_dim = config['data_dim']

font = cv2.FONT_HERSHEY_SIMPLEX


model_path = '/Users/jaejoon/LGuplus/main_project/lstm/model/model_mk10_20frame_8.5_100_0.0104_0.0005.pt'


def initModel(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(model_path, map_location=device)
    print(model)
    print('model initialization complete')
    return model


def testModel(model, test_data):
    # start = time.time()
    model.eval()
    with torch.no_grad():
        out = model(test_data)
        out = np.squeeze(out)
        out = F.softmax(out, dim=0)
        # print(actions[out.numpy().argmax()])
    # m, s = divmod(time.time() - start, 60)
    # print(f'Inference time: {m:.0f}m {s:.5f}s')
    # print(out)
    # print(out.numpy().argmax())
    return out[out.numpy().argmax()], actions[out.numpy().argmax()]


def getActionSequence(video, model):  # 비디오를 인풋으로 받아서 카운트한 액션을 순서대로 리스트에 넣어서 반환해줌
    extract = np.empty((1, data_dim))
    action_count = []
    action_sequence = []

    print('testing')
    cap = cv2.VideoCapture(video)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        i = 0
        sum = 0
        SFPS = ""
        while cap.isOpened():
            start_t = timeit.default_timer()
            success, image = cap.read()
            if not success:
                print(video, 'test complete')
                break

            # 미디어파이프를 이용하여 스켈레톤 추출
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

            temp = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten(
            ) if results.pose_world_landmarks else np.zeros(132)

            angles = extraFeatures.extractAngles(results)

            temp2 = getParts(temp, angles)
            extract = np.append(extract, [temp2], axis=0)
            extract = extract.astype(np.float32)

            image = cv2.flip(image, 1)

            if(extract.shape[0] > seq_length):
                extract = np.delete(extract, (0), axis=0)
                prob, action = testModel(model, torch.Tensor(extract))
                prob = prob.item()

                if prob > config['threshold']:
                    action_count.append(action)
                    if len(action_count) > action_count_length:
                        action_count.pop(0)

                    res, detected_action = count.countAction(action_count)
                    if res:
                        print(detected_action, 'action detected!')
                        action_sequence.append(detected_action)

                    cv2.putText(image, action, (50, 100),
                                font, 2, (0, 0, 255), 3)
                    cv2.putText(image, str(prob), (50, 300),
                                font, 2, (0, 0, 255), 3)

            cv2.putText(image, 'squat : {}, lunge : {}, pushup : {}'.format(count.cnt['squat'], count.cnt['lunge'], count.cnt['pushup']),
                        (0, 600), font, 2, (0, 255, 0), 2)

            # fps 계산
            terminate_t = timeit.default_timer()
            FPS = 1./(terminate_t - start_t)
            sum = FPS+sum
            if SFPS == "":
                SFPS = str(FPS)
            if i == 10:
                sum = sum/10
                sum = round(sum, 4)
                SFPS = str(sum)
                i = 0
                sum = 0
            i += 1
            cv2.putText(image, "FPS : "+SFPS, (640, 60), 0, 1, (255, 0, 0), 3)

            cv2.imshow('Testing', image)

    cap.release()
    return action_sequence


def getTestLabel(filename):
    label = filename[-7:-4]
    return list(label)


def test():
    model = initModel(model_path)
    test_videos_num = 0
    positive = 0
    test_videos_path = '/Users/jaejoon/LGuplus/main_project/lstm/test_videos'
    for filename in os.listdir(test_videos_path):
        test_videos_num += 1
        video_path = os.path.join(test_videos_path, filename)
        label = getTestLabel(filename)
        predicted = getActionSequence(video_path, model)

        print('label :', label, 'model prediction :', predicted)

        if predicted == label:
            positive += 1
            print('True')
        else:
            print('False')

    print(positive, test_videos_num, positive/test_videos_num)


def main():
    test()


if __name__ == '__main__':
    main()
