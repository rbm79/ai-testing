# Classificação de Transações para Identificação de Fraude


Sobre o ```modeltrain.py```

Este projeto usa um modelo de aprendizado de máquina para classificar transações como fraudulentas ou não fraudulentas. Utilizamos o algoritmo Random Forest para treinar um classificador usando um conjunto de dados simulado, onde cada transação tem um valor e um tempo de duração como características, e um rótulo indicando se é uma fraude.

## Estrutura do Código

1. **Criação do Conjunto de Dados**: Criamos um conjunto de dados simulado com características como `valor_transacao` e `tempo_transacao`, e uma coluna de rótulo `fraude` para indicar se a transação foi fraudulenta.

2. **Preparação dos Dados**: Dividimos os dados em variáveis independentes (`X`) e dependentes (`y`) e os separamos em conjuntos de treino e teste.

3. **Treinamento do Modelo**: Utilizamos o `RandomForestClassifier` do Scikit-Learn para treinar o modelo.

4. **Previsão e Avaliação**: Usamos o conjunto de teste para fazer previsões e avaliamos o desempenho do modelo com uma matriz de confusão e um relatório de classificação.

5. **Visualização**: Visualizamos as transações em um gráfico de dispersão, onde as transações fraudulentas e não fraudulentas são coloridas de forma diferente.

## Bibliotecas Utilizadas

- `numpy`: Para operações numéricas.
- `pandas`: Para manipulação de dados.
- `matplotlib`: Para visualização de dados.
- `scikit-learn`: Para modelagem e avaliação de aprendizado de máquina.
- `joblib`: Para salvar o modelo treinado (opcional).

## Como Executar

1. Instale as bibliotecas necessárias:
   ```bash
   pip install numpy pandas matplotlib scikit-learn joblib

2. Execute o script para criar o modelo e visualizar as transações.


## Estrutura do Código

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Passo 3: Criação do Conjunto de Dados Simulado
data = {
    'valor_transacao': [50, 150, 200, 1000, 5000, 75, 800, 25, 450, 7000, 30, 120],
    'tempo_transacao': [1, 2, 1, 5, 10, 1, 6, 1, 3, 15, 1, 2],
    'fraude': [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0]
}

df = pd.DataFrame(data)

# Passo 4: Preparação dos Dados
X = df[['valor_transacao', 'tempo_transacao']]
y = df['fraude']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Passo 5: Treinamento do Modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Passo 6: Previsões
y_pred = model.predict(X_test)

# Passo 7: Avaliação do Modelo
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Passo 8: Visualização (Opcional)
plt.scatter(df['valor_transacao'], df['tempo_transacao'], c=df['fraude'], cmap='coolwarm', edgecolor='k')
plt.xlabel('Valor da Transação')
plt.ylabel('Tempo da Transação')
plt.title('Transações: Fraude vs. Não Fraude')
plt.colorbar(label='Fraude (1) ou Não Fraude (0)')
plt.show()
```

## Resultados Esperados
Ao executar o código, você verá:

Uma matriz de confusão e um relatório de classificação com as métricas do modelo.
Um gráfico de dispersão visualizando as transações, diferenciando fraudes (1) de não fraudes (0).

## Licença
Este projeto é licenciado sob a MIT License.