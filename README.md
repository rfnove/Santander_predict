# Santander_predict

A escolha do Facebook Prophet baseou-se na sua robustez para séries temporais de negócios e na flexibilidade de sua arquitetura matemática.

No nosso pipeline, o comportamento de cada componente foi configurado da seguinte forma:

•	Tendência : Foi definido um crescimento linear por partes (piecewise linear). O Prophet varre o histórico e identifica automaticamente pontos de mudança de inclinação (changepoints). Isso permite que o modelo entenda se o mercado acelerou ou desacelerou recentemente e projete o futuro com base na velocidade mais atual.

•	Sazonalidade e Feriados: Como a base histórica consolidada por trimestre apresenta um horizonte curto (12 trimestres), as sazonalidades (anual, semanal e diária) e os efeitos de feriados foram configurados como False. Desativar esses componentes evitou o sobreajuste (overfitting), impedindo o algoritmo de inventar ciclos artificiais onde existem poucos pontos de dado.

Matematicamente, o Prophet reduziu-se a uma regressão linear segmentada. Ao contrário de uma regressão linear simples, que traça uma linha reta rígida do início ao fim, o Prophet adapta a inclinação ao longo do tempo. 

A variável alvo escolhida foi a Quantidade_total_cotas_ativas, que representa uma variável de estoque (o volume acumulado de consórcios ativos no mercado). Os dados foram agrupados por trimestre (ds) e por suas respectivas chaves geográficas e corporativas:


•	Pipeline por UF: Gerou uma visão limpa de 1 linha por trimestre para cada um dos 27 estados. 
•	Pipeline por Operação: Gerou visões isoladas por Administradora e Código de Segmento.

## Configuração de parâmetros do modelo:
Estes parâmetros foram injetados diretamente na instância do objeto Prophet() para controlar a complexidade matemática e o comportamento do ajuste das curvas:

•	changepoint_prior_scale = 0.05 (Regularização da Tendência): Ele define a flexibilidade da linha de tendência em relação a mudanças bruscas no mercado. O valor 0.05 (que é o padrão recomendado pelo Prophet) atuou como uma restrição de rigidez ideal: permitiu que o modelo se adaptasse de forma saudável às curvas de crescimento recentes de cada estado/administradora, mas impediu que ele sofresse com ruídos de flutuações atípicas de curtíssimo prazo, garantindo uma extrapolação segura para 2026.

•	yearly_seasonality = False (Sazonalidade Anual): Desativada intencionalmente. Como a base histórica consolidada é trimestral (composta por 12 trimestres/pontos no total), permitir que o algoritmo calculasse uma sazonalidade anual complexa causaria sobreajuste (overfitting), fazendo com que o modelo tentasse criar padrões repetitivos artificiais a partir de pouquíssimos dados.

•	weekly_seasonality = False (Sazonalidade Semanal): Desativada. O indicador de "cotas ativas" é medido e fechado em janelas macro (trimestrais), tornando os ciclos de dias de semana irrelevantes para a previsão.

•	daily_seasonality = False (Sazonalidade Diária): Desativada, uma vez que não existem dados com granularidade de dias ou horas na base.

•	growth = 'linear' (Tipo de Crescimento): Mantido no padrão linear por partes (piecewise linear). O modelo assume que o estoque de cotas ativas cresce ou diminui seguindo taxas lineares que podem mudar de velocidade em determinados pontos cruciais do histórico (changepoints).


# DEPLOY, SERIALIZAÇÃO E ARQUITETURA DA API
Com o modelo validado e aprovado, o pipeline foi estruturado para produção através de automação em Python:

Treinamento em Massa: Um loop percorreu a base de dados, instanciou e treinou 27 modelos independentes para os estados e centenas de modelos para as administradoras.

Serialização com JSON: Utilizando as funções nativas model_to_json e model_from_json do Prophet, a estrutura matemática e os coeficientes aprendidos por cada modelo foram "congelados" e salvos em arquivos .json locais, eliminando a necessidade de re-treinar os algoritmos a cada nova consulta.

Disponibilização via API: Foi construído um microsserviço em Flask (com suporte a CORS). Essa API carrega os arquivos JSON em memória na sua inicialização e disponibiliza rotas dinâmicas (ex: /api/prever/<estado>/<meses>). Quando o Dashboard em HTML/JavaScript consome essa rota, a API executa instantaneamente a função .predict() do Prophet, retornando os cenários base, otimista (yhat_upper) e pessimista (yhat_lower) para o planejamento comercial de 2026.

