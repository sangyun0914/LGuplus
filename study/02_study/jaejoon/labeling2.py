equivalancyList = {}


def labeling2(frame):
    label = 0

    y = frame.shape[0]
    x = frame.shape[1]

    for i in range(y):
        for j in range(x):
            if frame[i][j] > 0:
                if i > 0:
                    if j > 0:
                        if frame[i-1][j] == 0 and frame[i][j-1] == 0:
                            label += 1
                            equivalancyList[label] = label
                            frame[i][j] = label
                        elif frame[i-1][j] != 0 and frame[i][j-1] == 0:
                            frame[i][j] = frame[i-1][j]
                        elif frame[i-1][j] == 0 and frame[i][j-1] != 0:
                            frame[i][j] = frame[i][j-1]
                        else:
                            if(frame[i-1][j] < frame[i][j-1]):
                                equivalancyList[frame[i][j-1]] = frame[i-1][j]
                            else:
                                equivalancyList[frame[i-1]
                                                [j-1]] = frame[i][j-1]
                    else:
                        if frame[i-1][j] == 0:
                            label += 1
                            equivalancyList[label] = label
                            frame[i][j] = label
                        else:
                            frame[i][j] = frame[i-1][j]
                else:
