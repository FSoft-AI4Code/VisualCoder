def f(L, m, start, step): 
    L.insert(start, m) 
    for x in range(start-1, 0, -step): 
        start -= 1
        L.insert(start, L.pop(L.index(m)-1)) 
    return L

assert f([1, 2, 7, 8, 9], 3, 3, 2) == [1, 2, 7, 3, 8, 9]