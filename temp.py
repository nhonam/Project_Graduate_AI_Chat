def chunk_feature(str, arrPolarityTerm):
    # Maximum_Matching
    vTerm = []
    strRemain = ""
    start = 0
    isTerm = False
    isStop = False

    str = str.lower()
    str = str.lstrip(" ").rstrip(" ")
    WordList = str.split(" ")
    stop = len(WordList)

    while (isStop == False and stop >= 0):
        for num in range(start, stop):
            strRemain = strRemain + WordList[num] + " "

        strRemain = strRemain.lstrip(" ").rstrip(" ").lower()
        isTerm = False
        for cha in range(0, len(arrPolarityTerm)):
            arr = arrPolarityTerm[cha]
            if (arr == strRemain):
                vTerm.append(strRemain)
                isTerm = True
                if (start == 0):
                    isStop = True
                else:
                    stop = start
                    start = 0

        if (isTerm == False):
            if (start == stop):
                stop = stop - 1
                start = 0
            else:
                start += 1

        strRemain = ""
    strRemain = ""
    for stt in range(0, len(vTerm)):
        strRemain = strRemain + " " + vTerm[stt]

    return vTerm

arrTerm = ['hôm nay', 'hôm', 'nay', 'thi tốt', 'thi tốt nghiệp', 'tốt nghiệp']
strtemp = 'hôm nay thi tốt nghiệp'
arr = chunk_feature(strtemp, arrTerm)

print(arr)
