def strrmatch(strr, pattern, n, m):
	if (m == 0): 
		return (n == 0) 
	lookup = [[False for i in range(m + 1)] for j in range(n + 1)] 
	lookup[0][0] = True
	for j in range(1, m + 1): 
		if (pattern[j - 1] == '*'): 
			lookup[0][j] = lookup[0][j - 1] 
	for i in range(1, n + 1): 
		for j in range(1, m + 1): 
			if (pattern[j - 1] == '*'): 
				lookup[i][j] = lookup[i][j - 1] or lookup[i - 1][j] 
			elif (pattern[j - 1] == '?' or strr[i - 1] == pattern[j - 1]): 
				lookup[i][j] = lookup[i - 1][j - 1] 
			else: 
				continue#lookup[i][j] = False
	return lookup[n][m]



def aster(samples):
    cnt=0
    for sample in samples:
        cnt=cnt+1
        strr = ["fuck","kutta","sex","prick","bastard","penis","cunt","balls",
                "shit","witch","whore","arse","suck","lodu","loda","madarchod",
                "maadarchod","bhenchod","lunn","lund","penchod","painchod",
                "poop","boob","tits","ass","chutia","chut"]
        #pattern=input()
        for i in strr:
            if (strrmatch(i, sample, len(i),len(sample))): 
                samples[cnt-1]=i
                break
            else: 
                continue
    return samples
'''       '''
