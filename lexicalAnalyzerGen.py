#Universidad del Valle de Guatemala
#Diseño de Lenguajes de Programación - Sección: 10
#Mario Antonio Guerra Morales - Carné: 21008
#Analizador Léxico por medio de lectura de un archivo .yal (YALex)

#Importación de módulos y librerías para mostrar gráficamente los autómatas finitos.
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from collections import deque
from collections import defaultdict
from lexicalAnalyzer import *
import sys

#Declaración de archivos Yalex proporcionados para el laboratorio.
yalexArchive1 = "slr-1.yal"
yalexArchive2 = "slr-2.yal"
yalexArchive3 = "slr-3.yal"
yalexArchive4 = "slr-4.yal"

#Algoritmo Shunting Yard para pasar una expresión infix a postfix.

def insert_concatenation(expression): #Función insert_concatenation para poder agregar los operadores al arreglo result.
    result = [] #Lista result para agregar los operadores.
    operators = "+|*()?" #Operadores en la lista.
    i = 0
    while i < len(expression): #Por cada elemento según el rango de la variable expression:
        token = expression[i]
        
        if token.isdigit():
            number = token
            while i+1 < len(expression) and expression[i+1].isdigit():
                number += expression[i+1]
                i += 1
                token = expression[i]
            result.append(number)
        else:   
            result.append(token) #Se agrega caracter por caracter al arreglo.
        if i + 1 < len(expression): #Si la expresión es menor que la cantidad de elementos en el arreglo, se coloca en la posición i + 1.
            lookahead = expression[i + 1]
            if isinstance(token, str) or token == 'ε' or token == '#':
                if lookahead not in operators and lookahead != '.' and not (
                        isinstance(lookahead, str) and isinstance(token, str)):
                    result.append('.')
            elif isinstance(token, str) and lookahead == '(':
                result.append('.')
            elif isinstance(token, str) and isinstance(lookahead, str):
                result.append('.')
            elif token == '*' and lookahead == '(':
                result.append('.')
            elif token == '*' and isinstance(lookahead, str):
                result.append('.')
            elif token == '+' and lookahead == '(':
                result.append('.')
            elif token == '+' and isinstance(lookahead, str):
                result.append('.')
            elif token == '?' and lookahead == '(':
                result.append('.')
            elif token == '?' and isinstance(lookahead, str):
                result.append('.')
            elif token == ')' and isinstance(lookahead, str):
                result.append('.')
            elif token == ')' and lookahead == '(':
                result.append('.')
        i += 1

    return result #Devuelve el resultado.

def shunting_yard(expression): #Función para realizar el algoritmo shunting yard.
    precedence = {'|': 1, '.': 2, '*': 3, '+': 3, '?': 3}  # Orden de precedencia entre operadores.
    output_queue = []  # Lista de salida como notación postfix.
    operator_stack = []

    expression = insert_concatenation(expression)
    for token in expression:
        if token.isalnum() or token == '#':
            output_queue.append(token)
        elif token in "|*.+?":
            while (operator_stack and operator_stack[-1] != '(' and
                   precedence[token] <= precedence.get(operator_stack[-1], 0)):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output_queue.append(operator_stack.pop())
            operator_stack.pop()
        elif token == '.':
            while operator_stack and operator_stack[-1] != '(':
                output_queue.append(operator_stack.pop())
            if operator_stack and operator_stack[-1] == '(':
                operator_stack.pop()

    while operator_stack:
        output_queue.append(operator_stack.pop())

    return output_queue

def leer_archivo_yalex():
    with open(yalexArchive1, "r") as yalexArchive:
        content = yalexArchive.read()
        if not content:
            raise ValueError("El archivo .yal está vacío.")
    
    # Verifica si el archivo no contiene bloques 'let' o 'rule'
    if 'let' not in content or 'rule' not in content:
        raise ValueError("El archivo .yal no contiene bloques 'let' o 'rule'.")
    
    # Separar el contenido por 'let' o 'rule' para procesarlo por bloques
    blocks = []
    current_block = ''
    for line in content.splitlines():
        if line.strip().startswith(('let', 'rule')):
            if current_block:
                blocks.append(current_block.strip())
                current_block = ''
        current_block += line + '\n'
    if current_block:
        blocks.append(current_block.strip())
    
    yalexFunctions = []
    yalexRegex = []
    tokensBool = False

    for block in blocks:
        if block.startswith('let'):
            yalexFunctions.append(block[4:]) # Funciones del Yalex
            print("Print del let, ", str(yalexFunctions), "\n")
        elif block.startswith("rule"):
            tokensBool == True #Inicia la reglamentación de los tokens.
            lines = block.split("\n")
            for line in lines[1:]:
                if not line.startswith('(*'):
                    tokens = line.strip().split('{')[0].strip()
                    yalexRegex.extend(tokens.split())
    
    yalexRegex2 = []
    yalexFunctions2 = []

    for n in yalexRegex:
        if len(n) != 0:
            if n.count("'") == 2 or n.count('"') == 2:
                n = n[1:-1]
                ascii_nums = [str(ord(char)) for char in n]
                yalexRegex2.append('.'.join(ascii_nums))
            else:
                # Verifica si se encuentran dos "|" consecutivos
                if n == "|":
                    if yalexRegex2 and yalexRegex2[-1] == "|":
                        raise ValueError("Se encontró un token no definido.")
                yalexRegex2.append(n)
            print("Print de YalexRegex2, ", str(yalexRegex2), "\n")
        if yalexRegex2 and yalexRegex2[0] == "|":
            raise ValueError("La expresión no es válida.")
    
    for function in yalexFunctions:
        variable, definition = function.split("=")
        variable = variable.strip()
        definition = definition.strip()
        # Verifica si la definición de la variable está vacía
        if not definition:
            raise ValueError(f"La variable '{variable}' no tiene una definición después del signo igual (=).")
        arrayForRegex = []
        arrayForDefinition = []
        arrayForRegex.append(variable)
        print("Identificadores de variables: ", str(arrayForRegex))
        elements = ""

        if definition[0] == '[':
            definition = definition[1:-1]
            for element in definition:
                elements += element
                if elements[0] == "'" or elements[0] == '"':
                    if elements.count("'") == 2: #Caso donde la variable está delimitada por ''
                        elements = elements[1:-1]
                        if len(elements) == 2:
                            if elements == "\s":
                                elements = bytes(" ", "utf-8").decode("unicode_escape")
                            else:
                                elements = bytes(elements, "utf-8").decode("unicode_escape")
                            arrayForDefinition.append(ord(elements))
                            print("Abr 1 ", str(arrayForDefinition))
                        else:
                            if elements == " ":
                                elements = bytes(" ", "utf-8").decode("unicode_escape")
                                arrayForDefinition.append(ord(elements))
                                print("Abr 2 ", str(arrayForDefinition))
                            else:
                                arrayForDefinition.append(ord(elements))
                                print("Abr 3 ", str(arrayForDefinition))
                        elements = ""
                    if elements.count('"') == 2: #Caso donde la variable está delimitada por ""
                        elements = elements[1:-1]
                        newElements = ""
                        if chr(92) in elements:
                            for char in elements:
                                newElements += char
                                if newElements.count(chr(92)) == 2:
                                    if newElements[:-1] == "\s":
                                        escapedCharacter = " "
                                    else:
                                        escapedCharacter = newElements[:-1]
                                    elements = bytes(escapedCharacter, "utf-8").decode("unicode_escape")
                                    arrayForDefinition.append(ord(elements))
                                    print("Abr 4 ", str(arrayForDefinition))
                                    newElements = newElements[2:]
                            if len(newElements) != 0:
                                if newElements == "\s":
                                    escapedCharacter = " "
                                else:
                                    escapedCharacter = newElements
                                elements = bytes(escapedCharacter, "utf-8").decode("unicode_escape")
                                arrayForDefinition.append(ord(elements))
                                print("Abr 5 ", str(arrayForDefinition))
                        else:
                            elements = list(elements)
                            for i in range(len(elements)):
                                elements[i] = ord(elements[i])
                            arrayForDefinition.extend(elements)
                else:
                    if elements != '\n':
                        if elements != ' ':
                            if elements != '\t':
                                arrayForDefinition.append(elements)
                    print("Abr 6 ", str(arrayForDefinition))
                    elements = ""
                print("Atributos de las variables como unicode: ", str(arrayForDefinition))
        else:
            tokensArray = []
            token = ""
            for char in definition:
                if "]" in token:
                    charArray = []
                    character = ""
                    charArray.append("(")
                    print("Array de caracteres 1 ", str(charArray))
                    token = token[1:-1]
                    for element in token:
                        character += element
                        if character.count("'") == 2:
                            character = ord(character[1:-1])
                            charArray.append(character)
                            print("Array de caracteres 2 ", str(charArray))
                            charArray.append("|")
                            print("Array de caracteres 3 pero con el OR ", str(charArray))
                            character = ""
                    charArray[len(charArray) - 1] = ")"
                    tokensArray.extend(charArray)
                    token = ""
                if token.count("'") == 2:
                    if "[" not in token:
                        token = token[1:-1]
                        if token.isalnum():
                            tokensArray.append(ord(token))
                        else:
                            tokensArray.append(token)
                        print("TokenArray 1 ", str(tokensArray))
                        token = ""
                if char in ("(", ")", ".", "|", "*", "?", "+"):
                    if "'" not in token:
                        if token:
                            if len(token) == 1:
                                token = ord(token)
                            tokensArray.append(token)
                            print("TokenArray 2 ", str(tokensArray))
                            token = ""
                        if char == '.':
                            tokensArray.append(ord(char))
                        else:
                            tokensArray.append(char)
                        print("TokenArray 3 ", str(tokensArray))
                    else:
                        token += char
                else:
                    token += char
            if token:
                tokensArray.append(token)
                print("TokenArray 4 ", str(tokensArray))
            arrayForDefinition.extend(tokensArray)
        arrayForRegex.append(arrayForDefinition)
        yalexFunctions2.append(arrayForRegex)
    for i in range(len(yalexFunctions2)):
        bool = True
        for op in ["(", ")", "|", "*", "?", "+"]:
            if op in yalexFunctions2[i][1]:
                bool = False
        if bool == False:
            arrayForRegex = []
            for j in yalexFunctions2[i][1]:
                arrayForRegex.append(j)
                arrayForRegex.append('.')
            for k in range(len(arrayForRegex)):
                    if arrayForRegex[k] == "(":
                        if arrayForRegex[k+1] == ".":
                            arrayForRegex[k+1] = ""
                    if arrayForRegex[k] == ")":
                        if arrayForRegex[k-1] == ".":
                            arrayForRegex[k-1] = ""
                    if arrayForRegex[k] == "*":
                        if arrayForRegex[k-1] == ".":
                            arrayForRegex[k-1] = ""
                    if arrayForRegex[k] == "|":
                        if arrayForRegex[k-1] == ".":
                            arrayForRegex[k-1] = ""
                        if arrayForRegex[k+1] == ".":
                            arrayForRegex[k+1] = ""
                    if arrayForRegex[k] == "+":
                        if arrayForRegex[k-1] == ".":
                            arrayForRegex[k-1] = ""
                    if arrayForRegex[k] == "?":
                        if arrayForRegex[k-1] == ".":
                            arrayForRegex[k-1] = ""
            arrayForRegex = [element for element in arrayForRegex if element != ""]
            yalexFunctions2[i][1] = arrayForRegex[:-1]
        else:
            unicode_array = []
            funcArray = []
            if "-" in yalexFunctions2[i][1]:
                for k in range(len(yalexFunctions2[i][1])):
                    if yalexFunctions2[i][1][k] == "-":
                        for n in range(yalexFunctions2[i][1][k - 1], yalexFunctions2[i][1][k + 1] + 1,):
                            unicode_array.append(n)
                for n in unicode_array:
                    funcArray.append(n)
                yalexFunctions2[i][1] = funcArray
            funcArray = []
            for j in yalexFunctions2[i][1]:
                funcArray.append(j)
                funcArray.append("|")
            funcArray = funcArray[:-1]
            yalexFunctions2[i][1] = funcArray
    for function in yalexFunctions2:
        function[1] = ["("] + function[1] + [")"]

    variables = [i[0] for i in yalexFunctions2] + ["|"]
    yalexRegex2 = [
        ord(i) if len(i) == 1 and i not in variables else i
        for i in yalexRegex2
    ]

    yalexRegex3 = []
    for element in yalexRegex2:
        if element != "|":
            yalexRegex3.append("(")
            yalexRegex3.append(element)
            yalexRegex3.append(".")
            yalexRegex3.append("#")
            yalexRegex3.append(")")
        else:
            yalexRegex3.append(element)

    yalexRegex2 = yalexRegex3
    print("Regex de parte de los tokens: ", yalexRegex3)

    yalexRegex4 = getFinalRegex(yalexRegex2, dict(yalexFunctions2))
    return yalexRegex4

def getFinalRegex(yalexRegex, yalexFunctions):
            yalexRegex4 = []
            for el in yalexRegex:
                if el in yalexFunctions:
                    yalexRegex4.extend(getFinalRegex(yalexFunctions[el], yalexFunctions))
                else:
                    yalexRegex4.append(el)
            return yalexRegex4

#Algoritmo de Construcción Directa para convertir una regex en un AFD.

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.num = None
        self.position = 0

def build_syntax_tree(regex):
    regex_postfix = shunting_yard(regex)  # Convertir la expresión regular a formato postfix con '#' al final
    print("Esto es mi expresión en postfix: ", regex_postfix)
    regex_characters = [char if char in ['+', '|', '#', '.', '*', '?'] else chr(int(char)) for char in regex_postfix]
    print("Y estas son las expresiones sin formato ASCII: ", regex_characters)
    stack = []
    nodes_calculated = set()  # Conjunto para rastrear qué nodos ya han sido calculados
    leaf_calculated = set()
    position_counter = 1  # Contador para asignar números de posición
    nodo_position = 1

    for char in regex_postfix:
        if char.isalnum() or char == 'ε' or char == '#':
            node = Node(char)
            node.position = position_counter
            position_counter += 1
            node.num = nodo_position
            nodo_position += 1
            stack.append(node)
            leaf_calculated.add(node)
        elif char in ".|*+?":  # Operadores
            if char == '.':
                if len(stack) < 2:
                    raise ValueError("Insuficientes operandos para la concatenación")
                right = stack.pop()
                left = stack.pop()
                print(f"Concatenando nodos {left.value} y {right.value}")
                node = Node('.')
                node.left = left
                node.position = position_counter
                position_counter += 1
                node.right = right
                stack.append(node)
                nodes_calculated.add(node)
            elif char == '|':
                right = stack.pop()
                left = stack.pop()
                print(f"Creando nodo OR con hijos {left.value} y {right.value}")
                node = Node('|')
                node.left = left
                node.right = right
                node.position = position_counter
                position_counter += 1
                stack.append(node)
                nodes_calculated.add(node)
            elif char == '*':
                child = stack.pop()
                print(f"Creando nodo Kleene con hijo {child.value}")
                node = Node('*')
                node.left = child
                node.position = position_counter
                position_counter += 1
                stack.append(node)
                nodes_calculated.add(node)
            elif char == '+':
                child = stack.pop()
                print(f"Creando nodo Positivo con hijo {child.value}")
                node = Node('+')
                node.left = child
                node.position = position_counter
                position_counter += 1
                stack.append(node)
                nodes_calculated.add(node)
            elif char == '?':
                child = stack.pop()
                print(f"Creando nodo Opcional con hijo {child.value}")
                node = Node('?')
                node.left = child
                node.position = position_counter
                position_counter += 1
                stack.append(node)
                nodes_calculated.add(node)
        elif char == '#':
            if stack:
                child = stack.pop()
                if isinstance(child, Node):
                    node = Node('.')
                    node.left = child
                    node.right = Node('#')
                    node.position = position_counter
                    position_counter += 1
                    node.num = nodo_position
                    node.right.num = nodo_position
                    nodo_position += 1
                    print(f"Creando nodo concatenación con hijo izquierdo y hijo derecho #")
                    stack.append(node)
                    nodes_calculated.add(node)
                else:
                    node = Node('#')
                    node.right = child
                    node.position = position_counter
                    position_counter += 1
                    node.num = nodo_position
                    nodo_position += 1
                    print(f"Creando nodo marcador final con hijo {child.value}")
                    stack.append(node)
                    leaf_calculated.add(node)
            else:
                node = Node('#')
                node.position = position_counter
                position_counter += 1
                node.num = nodo_position
                nodo_position += 1
                print("Creando nodo marcador final sin hijos")
                stack.append(node)
    return stack.pop(), nodes_calculated, leaf_calculated

def visualize_tree(root):
    G = nx.DiGraph()
    build_networkx_graph(root, G)

    # Ajusta el parámetro scale para aumentar la distancia entre los nodos hijos
    pos = nx.kamada_kawai_layout(G, scale=100.0)

    labels = {node: node.value for node in G.nodes()}
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=700, node_color="skyblue", font_size=15, font_weight="bold")
    plt.show()

def build_networkx_graph(root, G):
    if root is not None:
        stack = [root]  # Usamos una pila para realizar DFS

        while stack:
            current_node = stack.pop()
            G.add_node(current_node)

            if current_node.left:
                stack.append(current_node.left)
                G.add_node(current_node.left)
                G.add_edge(current_node, current_node.left)

            if current_node.right:
                stack.append(current_node.right)
                G.add_node(current_node.right)
                G.add_edge(current_node, current_node.right)

            if not current_node.left and not current_node.right:
                G.add_node(current_node)

def get_all_nodes(node):
    nodes = set()

    if node is not None:
        nodes.add(node)
        nodes |= get_all_nodes(node.left)
        nodes |= get_all_nodes(node.right)

    return nodes

def nullable(node):
    if node.value == 'ε':
        return True
    elif node.value == '.':
        return nullable(node.left) and nullable(node.right)
    elif node.value == '|':
        return nullable(node.left) or nullable(node.right)
    elif node.value == '*':
        return True
    elif node.value == '+':
        return nullable(node.left)
    elif node.value == '?':
        return nullable(node.left)
    elif node.value == '#':
        return False
    elif node.value.isalnum():
        return False

def firstpos(node):
    if node.value.isalnum():
        return {node.num}
    elif node.value == 'ε':
        return {0}
    elif node.value == '.':
        if nullable(node.left):
            return firstpos(node.left) | firstpos(node.right)
        else:
            return firstpos(node.left)
    elif node.value == '|':
        return firstpos(node.left) | firstpos(node.right)
    elif node.value == '*':
        return firstpos(node.left)
    elif node.value == '+':
        return firstpos(node.left)
    elif node.value == '?':
        return firstpos(node.left)
    elif node.value == '#':
        return {node.num}

def lastpos(node):
    if node.value.isalnum():
        return {node.num}
    elif node.value == 'ε':
        return {0}
    elif node.value == '.':
        if nullable(node.right):
            return lastpos(node.left) | lastpos(node.right)
        else:
            return lastpos(node.right)
    elif node.value == '|':
        if nullable(node.left) or nullable(node.right):
            return lastpos(node.left) | lastpos(node.right)
        else:
            return lastpos(node.left) | lastpos(node.right)
    elif node.value == '*':
        return lastpos(node.left)
    elif node.value == '+':
        return lastpos(node.left)
    elif node.value == '?':
        return lastpos(node.left)
    elif node.value == '#':
        return {node.num}

def followpos(node):
    if node.value == '.':
        for pos in lastpos(node.left):
            for fp in firstpos(node.right):
                follow_pos[pos].add(fp)
    elif node.value == '*':
        for pos in lastpos(node):
            for fp in firstpos(node):
                follow_pos[pos].add(fp)
    elif node.value == '+':
        for pos in lastpos(node.left):
            for fp in firstpos(node.left):
                follow_pos[pos].add(fp)
    elif node.value == '|':
        pass 
    elif node.value == '?':
        pass  # No se necesita hacer nada para operador opcional

def build_dfa(follow_pos,root,leaf_calculated,expression):
    # Obtener el estado inicial del AFD
    start_state = tuple(firstpos(root))
    state_counter = 0

    # Obtener el alfabeto de la expresión regular
    alphabet = set([str(c) for c in expression if c not in ['+', '|', '*', '?', '.', '(', ')']])
    print("Este es el alfabeto.", alphabet)

    # Inicializar un grafo dirigido para representar el AFD
    dfaDirect = nx.DiGraph()
    # Agregar el estado inicial al AFD
    dfaDirect.add_node(start_state)
    state_counter += 1
    # Inicializar una lista de estados no marcados con el estado inicial del AFD
    unmarked_states = [start_state]

    # Proceso de construcción del AFD
    while unmarked_states:
        # Tomar un estado no marcado del AFD
        current_dfa_direct_state = unmarked_states.pop()
        # Para cada símbolo del alfabeto
        for symbol in alphabet:
            #if symbol in start_state:
                #dfaDirect.add_edge(start_state, start_state, label=symbol)
            # Calcular los estados a los que se llega desde el estado actual del AFD utilizando el símbolo
            target_states = set()
            for node in leaf_calculated:
                for pos in current_dfa_direct_state:
                    if pos == node.num and node.value == symbol:  # Verificar si la posición es igual al símbolo actual
                        target_states |= follow_pos[node.num]
            
            # Filtrar las tuplas vacías
            target_states = [state for state in target_states if state]
            
            target_states_list = list(target_states)
            while () in target_states_list:
                target_states_list.remove(())
            
            target_states_complete = tuple(target_states_list)

            # Convertir los estados obtenidos en una tupla ordenada
            if target_states_complete:
                dfa_direct_target_state = tuple(sorted(target_states_complete))
                print("\n Este es para el AFD ", dfa_direct_target_state)

                # Evitar agregar la tupla vacía al AFD
                if dfa_direct_target_state and dfa_direct_target_state != ():
                    # Si el estado obtenido no está en el AFD, marcarlo como no marcado y agregarlo al AFD
                    if dfa_direct_target_state not in dfaDirect:
                        unmarked_states.append(dfa_direct_target_state)
                        dfaDirect.add_node(dfa_direct_target_state)

                    # Agregar una transición desde el estado actual del AFD al estado obtenido con el símbolo actual
                    if dfaDirect.has_edge(current_dfa_direct_state, dfa_direct_target_state):
                        # Si ya existe una transición hacia el mismo estado, agregar el nuevo símbolo a la etiqueta de la transición
                        edge_data = dfaDirect.get_edge_data(current_dfa_direct_state, dfa_direct_target_state)
                        edge_data['label'] += (symbol,)  # Agregar el símbolo como una tupla de un solo elemento
                    else:
                        # Si no existe una transición hacia el mismo estado, agregar una nueva transición con el símbolo como tupla
                        dfaDirect.add_edge(current_dfa_direct_state, dfa_direct_target_state, label=(symbol,))
    
    # Establecer el estado inicial del AFD
    dfaDirect.graph['start'] = start_state
    # Obtener los estados de aceptación del AFD
    dfa_direct_accept_states = [state for state in dfaDirect.nodes if set(state) & set(lastpos(root)) and state != ()]
    # Establecer los estados de aceptación del AFD
    dfaDirect.graph['accept'] = dfa_direct_accept_states

    # Retornar el AFD construido
    return dfaDirect

def compute_epsilon_closure(dfaDirect, state):
    #Inicializar conjunto de cierre épsilon y pila con el estado inicial
    epsilon_closure = set()
    stack = [state]

    #Recorrer el grafo del AFD por construcción directa
    while stack:
        #Sacar un estado de la pila
        current_state = stack.pop()
        #Agregarlo al cierre épsilon
        epsilon_closure.add(current_state)

        #Recorrer los sucesores del estado actual
        for successor, edge_data in dfaDirect.adj[current_state].items():
            label = edge_data.get('label', None)
            #Si la etiqueta es épsilon y el sucesor no está en el cierre épsilon, agregarlo a la pila
            if label == 'ε' and successor not in epsilon_closure:
                stack.append(successor)
    #Retornar el cierre épsilon
    return epsilon_closure

def move(dfaDirect, state, symbol):
    #Inicializar conjunto de estados destino
    target_states = set()
    
    #Recorrer los sucesores del estado actual
    for successor, edge_data in dfaDirect.adj[state].items():
        label = edge_data.get('label', None)
        #Si la etiqueta coincide con el símbolo, agregar el sucesor al conjunto de estados destino
        if label == symbol:
            target_states.add(successor)
    #Retornar los estados destino
    return target_states

def get_alphabet(dfaDirect):
    #Inicializar conjunto de símbolos del alfabeto
    alphabet = set()
    
    #Recorrer todas las aristas del grafo del AFD por construcción directa
    for _, _, label in dfaDirect.edges(data='label'):
        #Si la etiqueta no es épsilon, agregarla al alfabeto
        if label != 'ε':
            alphabet.add(label)
    #Retornar el alfabeto
    return alphabet

def epsilon_closure(dfaDirect, states):
    #Inicializar cierre épsilon con los estados dados y una pila
    closure = set(states)
    stack = list(states)
    
    #Recorrer la pila
    while stack:
        state = stack.pop()
        #Recorrer los sucesores del estado actual
        for successor, attributes in dfaDirect[state].items():
            label = attributes['label']
            #Si la etiqueta es épsilon y el sucesor no está en el cierre épsilon, agregarlo al cierre y a la pila
            if label == 'ε' and successor not in closure:
                closure.add(successor)
                stack.append(successor)
            #Si la etiqueta es '*', agregar el sucesor al cierre y expandir su cierre épsilon
            elif label == '*':
                closure.add(successor)
                for epsilon_successor in epsilon_closure(afdDirect, {successor}):
                    if epsilon_successor not in closure:
                        closure.add(epsilon_successor)
                        stack.append(epsilon_successor)
    #Retornar el cierre épsilon
    return closure

def check_membership(dfaDirect, s):
    #Inicializar estados actuales con el cierre épsilon del estado inicial
    current_states = epsilon_closure(dfaDirect, {dfaDirect.graph['start']})
    
    #Recorrer los símbolos de la cadena de entrada
    for element in s:
        next_states = set()
        i = 0
        #Recorrer los estados actuales
        for state in current_states:
            #Recorrer los sucesores del estado actual
            for successor, attributes in dfaDirect[state].items():
                for label_element in attributes['label']:
                    if chr(int(label_element)) == element:
                        #Si la etiqueta coincide con el símbolo, agregar el cierre épsilon del sucesor a los estados siguientes
                        next_states |= epsilon_closure(dfaDirect, {successor})
                        print("Estado actual: ",state)
                        print("Posibles caminos: ", dfaDirect[state])
                        print("Lee simbolo: ",element)
            current_states = next_states
        print("Estado actual: ",state)
        print("Posibles caminos: ",dfaDirect[state])
        print("Lee simbolo: ",element)
    #Verificar si algún estado actual es un estado de aceptación
    return any(state in dfaDirect.graph['accept'] for state in current_states)

def encontrar_nodo_posicion_mas_grande(raiz):
    if raiz is None:
        return None

    # Inicializar el nodo con la posición más grande
    nodo_posicion_mas_grande = raiz

    # Recorrer el subárbol izquierdo
    nodo_izquierdo_mas_grande = encontrar_nodo_posicion_mas_grande(raiz.left)
    if nodo_izquierdo_mas_grande is not None and nodo_izquierdo_mas_grande.position > nodo_posicion_mas_grande.position:
        nodo_posicion_mas_grande = nodo_izquierdo_mas_grande

    # Recorrer el subárbol derecho
    nodo_derecho_mas_grande = encontrar_nodo_posicion_mas_grande(raiz.right)
    if nodo_derecho_mas_grande is not None and nodo_derecho_mas_grande.position > nodo_posicion_mas_grande.position:
        nodo_posicion_mas_grande = nodo_derecho_mas_grande

    return nodo_posicion_mas_grande

def remove_unreachable_states(dfa):
    # Encontrar estados alcanzables desde el estado inicial
    reachable_states = set()
    stack = [dfa.graph['start']]

    while stack:
        state = stack.pop()
        if state not in reachable_states:
            reachable_states.add(state)
            stack.extend(successor for successor in dfa.successors(state))
    # Encontrar estados no alcanzables
    unreachable_states = set(dfa.nodes) - reachable_states
    # Remover estados no alcanzables
    dfa.remove_nodes_from(unreachable_states)

#Algoritmo para minimizar un AFD hecho por construcción directa.

def hopcroft_minimization_dfa_direct(dfa_direct):
    # Inicializar particiones con estados de aceptación y no de aceptación
    partitions = [dfa_direct.graph['accept'], list(set(dfa_direct.nodes) - set(dfa_direct.graph['accept']))]
    # Inicializar una lista de trabajo con la partición de estados de aceptación
    worklist = deque([dfa_direct.graph['accept']])

    # Proceso de minimización de Hopcroft
    while worklist:
        partition = worklist.popleft()
        for symbol in get_alphabet(dfa_direct):
            divided_partitions = []
            for p in partitions:
                divided = set()
                for state in p:
                    # Verificar si hay transiciones con el símbolo actual hacia estados en la partición
                    successors = set(dfa_direct.successors(state))
                    if symbol in [dfa_direct.edges[(state, succ)]['label'] for succ in successors]:
                        divided.add(state)
                if divided:
                    divided_partitions.append(divided)
                    if len(divided) < len(p):
                        divided_partitions.append(list(set(p) - divided))
            # Actualizar las particiones si se dividen en particiones más pequeñas
            if len(divided_partitions) > len(partitions):
                if partition in partitions:
                    partitions.remove(partition)
                partitions.extend(divided_partitions)
                worklist.extend(divided_partitions)
    # Crear el DFA minimizado
    min_dfa_direct = nx.DiGraph()
    state_mapping = {}

    # Mapear estados a su representación en la partición
    for i, partition in enumerate(partitions):
        if partition:
            min_state = ', '.join(sorted(str(state) for state in partition))
            state_mapping.update({state: min_state for state in partition})

    # Construir las transiciones del DFA minimizado
    for source, target, label in dfa_direct.edges(data='label'):
        min_source = state_mapping[source]
        min_target = state_mapping[target]
        min_dfa_direct.add_edge(min_source, min_target, label=label)

    # Establecer el estado inicial y los estados de aceptación del DFA minimizado
    min_dfa_direct.graph['start'] = state_mapping[dfa_direct.graph['start']]
    min_dfa_direct.graph['accept'] = [state_mapping[state] for state in dfa_direct.graph['accept'] if state in state_mapping]

    # Remover nodos y aristas no alcanzables del DFA minimizado
    if '()' in min_dfa_direct.nodes:
        min_dfa_direct.remove_node('()')
        for source, target in list(min_dfa_direct.edges):
            if target == '()':
                min_dfa_direct.remove_edge(source, target)
    # Retornar el DFA minimizado
    return min_dfa_direct

if __name__ == "__main__":
    #try:
        regexList = leer_archivo_yalex()

        print("Nuestra expresión regular es la siguiente: ", regexList)

        # Inicializamos una cadena vacía para almacenar los elementos
        regex = ''

        # Recorremos el arreglo y concatenamos cada elemento a la cadena
        for element in regexList:
            regex += str(element)
        
        print("Y esta es nuestra expresión regular: ", regex)

        w = input("Ingrese una cadena: ")
        
        # Construcción directa (AFD).
        
        syntax_tree, nodes_calculated, leaf_calculated = build_syntax_tree(regex)
        print("Árbol Sintáctico:")
        visualize_tree(syntax_tree)

        root = encontrar_nodo_posicion_mas_grande(syntax_tree)

        follow_pos = {node.num: set() for node in leaf_calculated}

        for num, conjunto in follow_pos.items():
            print(f"Posición: {num}, Conjunto: {conjunto}")

        # Calcula firstpos, lastpos y followpos

        for node in nodes_calculated:
            followpos(node)

        print("\nFollowpos:")
        for num, conjunto in follow_pos.items():
            print(f"Posición: {num} : {conjunto}")
        
        # Construye el AFD
        afdDirect = build_dfa(follow_pos,root,leaf_calculated,regexList)

        if ((), ()) in afdDirect.nodes:
            afdDirect.remove_node(((), ()))
            # Asegúrate de también eliminar cualquier arista que apunte a este nodo
            for source, target in list(afdDirect.edges):
                if target == ((), ()):
                    afdDirect.remove_edge(source, target)

        filtered_edges = [(source, target, label) for source, target, label in afdDirect.edges(data='label') if source != () and target != ()]

        # Filtrar los nodos que no son tuplas vacías
        filtered_nodes = [node for node in afdDirect.nodes if node != ()]

        simbolos = set(label for _, _, label in afdDirect.edges(data='label'))
        # Obtener el conjunto de estados iniciales
        estados_iniciales = {nodo for nodo in filtered_nodes if len(list(afdDirect.predecessors(nodo))) == 0}

        estados_aceptacion = set()
        for nodo in filtered_nodes:
            aceptacion = True
            for succ in afdDirect.successors(nodo):
                if succ not in filtered_nodes or succ == ():
                    aceptacion = False
                    break
            if aceptacion:
                estados_aceptacion.add(nodo)

        # Visualiza el AFD
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(afdDirect)
        labels = {edge: label for edge, label in nx.get_edge_attributes(afdDirect, 'label').items()}
        nx.draw(afdDirect, pos, with_labels=True, node_size=200, node_color='blue')
        nx.draw_networkx_edge_labels(afdDirect, pos, edge_labels=labels)
        plt.title("Direct AFD Visualization")
        plt.axis("off")
        plt.show()

        result = check_membership(afdDirect, w)
        if result:
            print(f"'{w}' pertenece al lenguaje L({regex})")
        else:
            print(f"'{w}' no pertenece al lenguaje L({regex})")
    #except Exception as e:
        #print("Error: ", str(e))
        #sys.exit(1)