from Definitions import *

def scan(tokenYal):
    if tokenYal == 'ws':
        return tokenYal
    if tokenYal == 'id':
        try:
           return ID
        except NameError:
            print('Valor de retorno no definido.')
    if tokenYal == '+':
        try:
           return PLUS
        except NameError:
            print('Valor de retorno no definido.')
    if tokenYal == '*':
        try:
           return TIMES
        except NameError:
            print('Valor de retorno no definido.')
    if tokenYal == '(':
        try:
           return LPAREN
        except NameError:
            print('Valor de retorno no definido.')
    if tokenYal == ')':
        try:
           return RPAREN
        except NameError:
            print('Valor de retorno no definido.')
    return tokenYal

def outputScanner(scannerList):
    for token, element in zip(scannerList[0], scannerList[1]):
        if token == '':
            print(f'Simbolo {element} -> Token no definido')
        else:
            scanSymbol = scan(token)
            print(f'Simbolo {element} -> Token {scanSymbol}')
