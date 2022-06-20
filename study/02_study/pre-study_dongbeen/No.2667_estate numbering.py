def estate(maze, check, i, j, a):
    count = 0
    ok = 0
    #print('0:', i, j)
    if check[i][j] == 0:
        ok = 0
    else:
        ok = 1
        count += 1
        check[i][j] = 0
        if i-1 >= 0 and check[i-1][j] == 1:
            #print('1:',i-1, j, maze[i][j], check[i-1][j])
            count = count + estate(maze, check, i-1, j, a)
        if i+1 < a and check[i+1][j] == 1:
            #print('2:',i+1, j,maze[i][j], check[i+1][j])
            count = count + estate(maze, check, i+1, j, a)
        if j-1 >= 0 and check[i][j-1] == 1:
            #print('3:',i, j-1,maze[i][j], check[i][j-1])
            count = count + estate(maze, check, i, j-1, a)
        if j+1 < a and check[i][j+1] == 1:
            #print('4:', i, j+1,maze[i][j], check[i][j+1])
            count = count + estate(maze, check, i, j+1, a)

    return count

def main():
    a = int(input())

    maze = [list(map(int, input())) for _ in range(a)]

    check = [[1]*a for i in range(a)]

    for i in range(a):
        for j in range(a):
            if maze[i][j] == 0:
                check[i][j] = 0

    estateCount = 0
    estateM = []

    for i in range(a):
        for j in range(a):
            if check[i][j] == 1:
                count = estate(maze, check, i, j, a)
                #print(count)
                if count > 0:
                    estateCount += 1
                    estateM.append(count)
                


    '''for i in range(a):
        for j in range(a):
            print(check[i][j], end="")
        print("")'''
    estateM.sort()
    
    print(estateCount)
    for i in range(estateCount):
        print(estateM[i])

if __name__ == "__main__":
    main()