#Universidad del Valle de Guatemala
#Diseño de Lenguajes de Programación - Sección: 10
#Mario Antonio Guerra Morales - Carné: 21008
#ScanFrame para generar el Scanner.py

from lexicalAnalyzerGen import *
import sys

def createScanner(regexTokens):
    i = 0
    with open("Scanner.py", "w") as file:
        file.write("def scan(tokenYal):\n")
        while i < len(regexTokens):
            token = regexTokens[i]
            code = regexTokens[i+1]
            file.write(f"    if tokenYal == '{token}':\n")
            file.write("        try:\n")
            if code == ' ':  # Verificar si el código consiste únicamente en espacios en blanco
                file.write("            return\n")
            else:
                file.write(f"            {code}\n")
            file.write("        except NameError:\n")
            file.write("            return f'Token {tokenYal} no definido en archivo'\n")
            i += 2
        file.close
