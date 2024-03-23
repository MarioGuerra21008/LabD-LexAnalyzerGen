def scan(tokenYal):
    if tokenYal == 'ws':
        try:
            return
        except NameError:
            return f'Token {tokenYal} no definido en archivo'
    if tokenYal == 'id':
        try:
            return ID
        except NameError:
            return f'Token {tokenYal} no definido en archivo'
    if tokenYal == '+':
        try:
            return PLUS
        except NameError:
            return f'Token {tokenYal} no definido en archivo'
    if tokenYal == '*':
        try:
            return TIMES
        except NameError:
            return f'Token {tokenYal} no definido en archivo'
    if tokenYal == '(':
        try:
            return LPAREN
        except NameError:
            return f'Token {tokenYal} no definido en archivo'
    if tokenYal == ')':
        try:
            return RPAREN
        except NameError:
            return f'Token {tokenYal} no definido en archivo'
