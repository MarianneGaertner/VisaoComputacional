def operacoes_basicas(a, b):
    soma = a + b
    subtracao = a - b
    multiplicacao = a * b
    # exemplo de uso de operador ternário (abaixo)
    divisao = a / b if b != 0 else "Divisão por zero não é permitida"
    
    return soma, subtracao, multiplicacao, divisao

# Exemplo de uso da função
a = 10
b = 5
r1, r2, r3, r4 = operacoes_basicas(a, b)

print(f"Soma: {r1}")
print(f"Subtração: {r2}")
print(f"Multiplicação: {r3}")
print(f"Divisão: {r4}")