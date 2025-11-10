# Criando um dicionário
aluno = { "nome": "Maria", "idade": 22, "curso": "Engenharia"}

# Acessando valores
print("Nome:", aluno["nome"])

# Adicionando um novo par chave-valor
aluno["nota"] = 9.5
print("Dicionário atualizado:", aluno)

# Removendo um par chave-valor
del aluno["idade"]
print("Dicionário após remoção:", aluno)