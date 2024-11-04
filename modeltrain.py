import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


####  Passo 3: Criar um Conjunto de Dados Simulado
####  Vamos criar um conjunto de dados fictício com características de transações. 
####  O conjunto terá duas características (por simplicidade) e 
####  uma coluna de rótulo (0 para não fraude e 1 para fraude).
# Criar um DataFrame simulado
data = {
    'valor_transacao': [50, 150, 200, 1000, 5000, 75, 800, 25, 450, 7000, 30, 120],
    'tempo_transacao': [1, 2, 1, 5, 10, 1, 6, 1, 3, 15, 1, 2],
    'fraude': [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0]
}

df = pd.DataFrame(data)

print(df)

####  Passo 4: Preparar os Dados
####  Divida os dados em variáveis independentes (X) e dependentes (y) e,
####  em seguida, divida os dados em conjuntos de treinamento e teste:
# Separar as variáveis independentes e dependentes
X = df[['valor_transacao', 'tempo_transacao']]
y = df['fraude']

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

####  Passo 5: Treinar o Modelo
####  Usaremos um classificador Random Forest para treinar o modelo:
# Criar o classificador Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinar o modelo
model.fit(X_train, y_train)


####  Passo 6: Fazer Previsões
####  Agora, podemos fazer previsões com o conjunto de teste:
# Fazer previsões
y_pred = model.predict(X_test)


####  Passo 7: Avaliar o Modelo
####  Avalie o desempenho do modelo usando uma matriz de confusão e um relatório de classificação:
# Avaliar o modelo
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

####  Passo 8: Visualização (Opcional)
####  Você pode visualizar as previsões em um gráfico de dispersão:
# Visualizar as transações
plt.scatter(df['valor_transacao'], df['tempo_transacao'], c=df['fraude'], cmap='coolwarm', edgecolor='k')
plt.xlabel('Valor da Transação')
plt.ylabel('Tempo da Transação')
plt.title('Transações: Fraude vs. Não Fraude')
plt.colorbar(label='Fraude (1) ou Não Fraude (0)')
plt.show()
