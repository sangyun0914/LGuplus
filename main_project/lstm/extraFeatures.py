import numpy as np

# ab, ac 벡터 사이 각도 구하기 - radian 값으로 리턴해줌


def getAngle(landmarks, a, b, c):
    A = landmarks[a]
    B = landmarks[b]
    C = landmarks[c]
    ab = B - A
    ac = C - A
    ab_len = np.linalg.norm(ab)
    ac_len = np.linalg.norm(ac)
    cos = np.dot(ab, ac) / (ab_len * ac_len)
    angle = np.array([np.arccos(cos)])
    return angle


def landmark2nparray(landmark):
    return np.array([landmark.x, landmark.y, landmark.z, landmark.visibility])


# input : mediapipe의 pose.process()의 results
# output : 8개의 관절 사이 각도로 이루어진 1차원 넘파이 배열 (8,)
def extractAngles(results):
    temp = np.array(results.pose_world_landmarks.landmark)
    landmarks = np.array([landmark2nparray(x) for x in temp])
    out = getAngle(landmarks, 13, 11, 15)
    out = np.append(out, getAngle(landmarks, 14, 12, 16))
    out = np.append(out, getAngle(landmarks, 11, 13, 23))
    out = np.append(out, getAngle(landmarks, 12, 14, 24))
    out = np.append(out, getAngle(landmarks, 23, 11, 25))
    out = np.append(out, getAngle(landmarks, 24, 12, 26))
    out = np.append(out, getAngle(landmarks, 25, 23, 27))
    out = np.append(out, getAngle(landmarks, 26, 24, 28))
    # print(out.shape)
    return out
