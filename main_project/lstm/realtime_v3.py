from configs import *
from model_v3 import Model
import test
import extraFeatures

seq_length = config['seq_length']
data_dim = config['data_dim']

model = test.initModel()

extract = np.empty((1, 88))

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    i = 0
    sum = 0
    SFPS = ""
    while cap.isOpened():
        start_t = timeit.default_timer()
        success, image = cap.read()
        if not success:
            continue

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
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

        temp = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten(
        ) if results.pose_world_landmarks else np.zeros(132)

        angles = extraFeatures.extractAngles(results)

        left_upper = np.hstack(
            [temp[0:12], temp[16:20], temp[48:52], angles[0:1], angles[2:3]])
        right_upper = np.hstack(
            [temp[0:8], temp[12:16], temp[20:24], temp[52:56], angles[1:2], angles[3:4]])
        left_lower = np.hstack(
            [temp[0:4], temp[48:60], temp[64:68], angles[4:5], angles[6:7]])
        right_lower = np.hstack(
            [temp[4:8], temp[48:56], temp[60:64], temp[68:72], angles[5:6], angles[7:8]])

        temp2 = np.hstack(
            [left_upper, right_upper, left_lower, right_lower])
        extract = np.append(extract, [temp2], axis=0)
        extract = extract.astype(np.float32)

        image = cv2.flip(image, 1)

        if(extract.shape[0] > seq_length):
            extract = np.delete(extract, (0), axis=0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            prob, action = test.testModel(model, torch.Tensor(extract))
            prob = prob.item()
            cv2.putText(image, action, (50, 100),
                        font, 2, (255, 0, 0), 2)
            cv2.putText(image, str(prob), (50, 300),
                        font, 2, (255, 0, 0), 2)

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

        if cv2.waitKey(5) > 0:
            break


cap.release()
