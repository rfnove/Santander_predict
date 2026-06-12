# Santander Predict 📊📈

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Prophet](https://img.shields.io/badge/Facebook_Prophet-8A2BE2?style=for-the-badge)

Este repositório contém o pipeline  de engenharia de dados, modelagem preditiva e deploy de microsserviço para projeção do mercado de consórcios no cenário de 2026.

---

## 🧠 Arquitetura do Modelo Preditivo

A escolha do algoritmo **Facebook Prophet** baseou-se na sua robustez para séries temporais de negócios e na alta flexibilidade de sua arquitetura matemática. O Prophet formula o problema de previsão como um Modelo Aditivo Generalizado (GAM):

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

No nosso pipeline, o comportamento de cada componente foi configurado de forma cirúrgica para se adequar à natureza dos dados históricos:

* **Tendência $g(t)$:** Foi definido um crescimento **linear por partes (*piecewise linear*)**. O Prophet varre o histórico e identifica automaticamente pontos de mudança de inclinação (*changepoints*). Isso permite que o modelo compreenda se o mercado acelerou ou desacelerou recentemente e projete o futuro com base na velocidade estrutural mais atual.
* **Sazonalidade $s(t)$ e Feriados $h(t)$:** Como a base histórica consolidada por trimestre apresenta um horizonte curto (12 trimestres no total), as sazonalidades (anual, semanal e diária) e os efeitos de feriados foram desativados (`False`). Isolar esses componentes evitou o sobreajuste (*overfitting*), impedindo o algoritmo de capturar ciclos artificiais onde há poucos pontos amostrais de dados.

> 💡 **Nota Teórica:** Sem os componentes sazonais e de eventos, o Prophet reduziu-se a uma **regressão linear segmentada**. Ao contrário de uma regressão linear clássica (que traça uma linha reta única e rígida do início ao fim), o Prophet adapta de forma maleável a inclinação da reta ao longo do tempo.

---

## 🗄️ Engenharia e Consolidação de Dados

A variável alvo escolhida foi a `Quantidade_total_cotas_ativas`, que se caracteriza como uma variável de **estoque** (o volume acumulado de consórcios ativos simultaneamente no mercado). 

Para alimentar o modelo de forma consistente e evitar redundâncias/duplicações no momento da inferência, os dados brutos foram limpos e agregados estritamente em janelas trimestrais (`ds`), sendo fatiados através de dois pipelines distintos:

1.  **Pipeline por UF:** Consolida uma série temporal única (1 linha por trimestre) para cada um dos 27 estados do Brasil.
2.  **Pipeline por Operação:** Estrutura visões isoladas por **Administradora** e **Código de Segmento** (permitindo análises granulares como a performance da operação do Santander).

---

## 🎛️ Hiperparâmetros do Modelo

Os parâmetros abaixo foram injetados diretamente na instância do objeto `Prophet()` para controlar a complexidade matemática e o comportamento do ajuste das curvas:

| Parâmetro | Configuração | Objetivo Prático |
| :--- | :---: | :--- |
| `growth` | `'linear'` | Define a tendência padrão como linear por partes (*piecewise linear*). |
| `changepoint_prior_scale` | `0.05` | **Regularização da Tendência:** Atua como uma restrição de rigidez ideal. Permite que o modelo se adapte às curvas recentes do mercado, mas bloqueia flutuações atípicas de curtíssimo prazo, garantindo uma extrapolação segura para 2026. |
| `yearly_seasonality` | `False` | Desativada para impedir que o algoritmo tente criar padrões repetitivos anuais complexos a partir de apenas 12 trimestres históricos disponíveis. |
| `weekly_seasonality` | `False` | Desativada. Indicadores de estoque macro trimestrais tornam variações de dias de semana irrelevantes. |
| `daily_seasonality` | `False` | Desativada pela ausência de granularidade diária na base. |

---

## 🚀 Deploy, Serialização e Arquitetura da API

Com os modelos validados com **98,06% de acurácia global (MAPE de 1,94%)** na janela de holdout, o pipeline foi estruturado para produção através de automação em Python, englobando três etapas:

### 1. Treinamento em Massa
Um script automatizado em lote (*batch*) varre as matrizes de dados, instancia, ajusta e treina de forma paralela 27 modelos independentes para as UFs e centenas de variações para as combinações de Administradoras/Segmentos.

### 2. Serialização em JSON
Utilizando as funções nativas de serialização do Prophet (`model_to_json` e `model_from_json`), a estrutura matemática, pesos e coeficientes aprendidos por cada modelo foram "congelados" e exportados em arquivos `.json` locais. Isso mitiga despesas computacionais e elimina a necessidade de re-treinar os algoritmos a cada nova consulta.

### 3. Disponibilização via API (Flask)
Foi construído um microsserviço utilizando o framework **Flask** (com suporte nativo a CORS). 

* **Inicialização:** A API carrega todos os arquivos JSON de modelos diretamente em memória RAM no momento do *startup*.
* **Rotas Dinâmicas:** Disponibiliza endpoints assíncronos como:
    ```http
    GET /api/prever/<estado_ou_global>/<meses>
    GET /api/modelos/status
    ```
* **Inferência:** Ao receber uma requisição vinda do Dashboard Frontend (HTML/JavaScript), a API executa instantaneamente o método `.predict()` do modelo correspondente mapeado em memória, retornando as projeções contendo o cenário base (`yhat`), o cenário otimista (`yhat_upper`) e o pessimista (`yhat_lower`) para o planejamento tático do negócio.

