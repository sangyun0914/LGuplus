dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]

count = 0


def DFS(x, y, graph):
    m = graph.shape[1]
    n = graph.shape[0]

    if x < 0 or x >= m or y < 0 or y >= n:
        return False

    #print(x, y)
    if graph[y][x] > 0:
        global count
        count += 1
        graph[y][x] = 0
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            DFS(nx, ny, graph)
        return True
    return False


def labeling(frame):
    graph = frame
    m = frame.shape[1]
    n = frame.shape[0]
    num = []
    count = 0
    result = 0

    for i in range(m):
        for j in range(n):
            if DFS(i, j, graph) == True:
                num.append(count)
                result += 1
                count = 0

    num.sort()

    return num
