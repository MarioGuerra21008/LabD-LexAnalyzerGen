def scan(tokenYal):
    if tokenYal == 'ws':
        return tokenYal
    if tokenYal == 'id':
       return ID
    if tokenYal == '+':
       return PLUS
    if tokenYal == '*':
       return TIMES
    if tokenYal == '(':
       return LPAREN
    if tokenYal == ')':
       return RPAREN
    return tokenYal

def outputScanner(scannerList):
    for token, element in zip(scannerList[0], scannerList[1]):
        if token == '':
            print(f'Simbolo {element} -> Token no definido')
        else:
            scanSymbol = scan(token)
            print(f'Simbolo {element} -> Token {scanSymbol}')
