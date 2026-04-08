# Relatorio Tecnico - Operacionalizacao de Modelos com MLOps

**Disciplina:** Operacionalizacao de Modelos com MLOps
**Dataset:** Loan Approval Classification (45.000 registros, 13 features)

---

## 1. Introducao e Contexto

### Problema de negocio

Uma instituicao financeira recebe milhares de solicitacoes de emprestimo e precisa decidir quais aprovar e quais rejeitar. Cada decisao errada tem custo:

- **Falso Positivo (aprovar caloteiro):** o banco perde o valor do emprestimo.
- **Falso Negativo (rejeitar bom cliente):** o banco perde a receita de juros.

### Objetivo tecnico

Classificacao binaria: prever se uma solicitacao deve ser aprovada (1) ou rejeitada (0), usando 13 caracteristicas do solicitante e do emprestimo.

### Metricas de sucesso

- **Tecnica:** F1-Score - equilibra nao aprovar caloteiros (precision) e nao perder bons clientes (recall).
- **Negocio:** impacto financeiro total (custo dos erros em reais).

### Evolucao do projeto

No projeto anterior, exploramos diferentes modelos num notebook unico. Testamos Perceptron, Decision Trees e Random Forest, e identificamos que o Random Forest otimizado produzia o melhor resultado (F1=0.83).

Neste projeto, o desafio mudou: nao se trata mais de encontrar o melhor modelo, mas de estruturar o trabalho como um projeto de engenharia. Reorganizamos o notebook em scripts modulares, adicionamos configuracoes externas, tracking de experimentos, validacao de qualidade, e monitoramento - transformando uma exploracao pontual num sistema reprodutivel e operacional.

---

## 2. Estruturacao do Projeto

### De notebook para engenharia

No projeto anterior, toda a logica vivia num notebook de 89 celulas. Isso funciona para explorar, mas apresenta limitacoes para operacao:

- Dificil de reutilizar partes do codigo em outros contextos
- Impossivel de testar modulos isoladamente
- Sem separacao entre configuracao e logica
- Sem rastreabilidade de experimentos

Neste projeto, cada responsabilidade tem seu modulo:

| Modulo | O que faz |
|--------|-----------|
| `config.py` | Carrega e valida configuracoes |
| `ingestion.py` | Baixa dados e converte para Parquet |
| `quality_checks.py` | Valida qualidade dos dados |
| `preprocessing.py` | Prepara dados para treinamento |
| `reduction.py` | Reducao de dimensionalidade (PCA/LDA) |
| `train.py` | Treina modelos e registra no MLflow |
| `evaluate.py` | Compara modelos e calcula impacto financeiro |
| `serve.py` | Carrega modelo salvo e faz predicao |
| `monitoring.py` | Detecta mudancas nos dados (drift) |
| `custom_models.py` | Modelos customizados escritos manualmente |

A vantagem dessa modularizacao vai alem da organizacao: cada modulo pode ser reutilizado, substituido ou testado independentemente. Se amanha a estrategia de validacao de qualidade precisar mudar (por exemplo, usar uma biblioteca diferente ou adicionar regras especificas por fonte de dados), basta alterar `quality_checks.py` sem impactar o restante do pipeline.

### Configuracao via YAML + dataclasses

As configuracoes do projeto sao externas ao codigo, definidas em arquivos YAML:

- `data.yaml`: contrato de dados - quais colunas existem, tipos, fonte
- `pipeline.yaml`: parametros de execucao - random_state, proporcao treino/teste
- `quality.yaml`: regras de qualidade - ranges aceitaveis, nulos, categorias
- `experiments_prod.yaml`: modelo que roda em producao
- `experiments_test.yaml`: modelos em fase de teste

Cada YAML e validado por uma dataclass Python. Campos obrigatorios geram erro se faltarem. Campos opcionais tem valor padrao. Isso garante que erros de configuracao aparecem na carga, nao no meio do treinamento.

O professor demonstrou YAML com dicionarios Python. Adicionamos dataclasses como camada de validacao porque e a pratica usada profissionalmente - o YAML continua sendo a interface de configuracao (editavel sem mexer no codigo), e a dataclass garante que a estrutura esta correta.

### Separacao de ambientes

O pipeline opera em dois modos, controlados por um parametro na execucao:

- **make test:** carrega `experiments_test.yaml` (8 modelos), salva no MLflow experiment de experimentacao, roda reducao de dimensionalidade.
- **make prod:** carrega `experiments_prod.yaml` (1 modelo), salva no MLflow experiment de producao, pula reducao.

Dados processados e relatorios ficam em pastas separadas por modo e mes de execucao, sem sobrescrita.

### Makefile como pipeline de CI/CD simulado

O Makefile funciona como um pipeline de CI/CD: cada comando e uma etapa que valida,
treina, compara, promove e monitora. A sequencia completa de deploy:

```
make setup → make configs → make test → make compare → make promote → make prod → make post-deploy
```

| Etapa | Comando | O que faz |
|-------|---------|-----------|
| Setup | `make setup` | Cria ambiente e instala dependencias |
| Gate | `make configs` | Valida configuracoes (se falhar, para) |
| Experimentacao | `make test` | Treina todos os modelos, registra no MLflow |
| Analise | `make compare` | Tabela comparativa + exporta CSV |
| Promocao | `make promote MODEL=nome` | Move modelo escolhido para producao |
| Producao | `make prod` | Treina modelo de producao |
| Pos-deploy | `make post-deploy` | Compara run atual com anterior (detecta degradacao) |
| Inferencia | `make serve` | Inicia servico de inferencia (CLI) |
| Visual | `make mlflow` | Abre interface de resultados |

Para a manutencao mensal, a sequencia e mais curta:

```
make prod → make post-deploy
```

Se `post-deploy` detectar degradacao (F1 caiu mais de 0.02), a recomendacao e rodar `make test` para re-avaliar todos os modelos.

Em producao real, esses comandos seriam etapas de um pipeline de CI/CD automatizado (GitHub Actions, GitLab CI, Jenkins). Cada etapa teria criterios de aprovacao: configs validas → qualidade ok → F1 acima do limiar → deploy automatico.

### Flexibilidade de inputs

Nem sempre e possivel garantir que todas as fontes de dados tenham o mesmo formato. Em cenarios reais, diferentes fontes podem trazer informacoes diferentes - por exemplo, um estado envia dados de licencas com 10 colunas e outro envia com 15. O importante e encontrar um ponto em comum no output: todos os modelos recebem as mesmas features padronizadas e produzem o mesmo tipo de resultado.

A arquitetura que construimos permite isso. O contrato de dados (`data.yaml`) define o formato padrao. Se amanha uma nova fonte de dados precisar ser integrada, basta criar um preprocessamento especifico para essa fonte que converte os dados para o formato do contrato. O modelo nao muda - ele sempre recebe features padronizadas. A complexidade fica na ingestao, nao no treinamento.

---

## 3. Fundacao de Dados

### Dataset

Loan Approval Classification (Kaggle): 45.000 solicitacoes de emprestimo com 14 colunas.

- 8 numericas: idade, renda, experiencia profissional, valor do emprestimo, taxa de juros, percentual da renda comprometido, historico de credito, score de credito
- 5 categoricas: genero, escolaridade, tipo de moradia, finalidade do emprestimo, historico de calote
- 1 alvo: aprovado (22%) ou rejeitado (78%)

### Ingestao

O dataset e baixado e convertido de CSV para Parquet (compressao Snappy). Parquet armazena dados por coluna - permite ler apenas as colunas necessarias e ocupa menos espaco.

Neste projeto usamos Parquet como camada de armazenamento local. Em ambiente de producao real, a escolha dependeria da infraestrutura da empresa - podendo ser um data warehouse, data lake, ou banco relacional, conforme volume e frequencia de acesso.

### Contrato de dados

O `data.yaml` define quais colunas o pipeline espera. Na ingestao, o sistema verifica se todas as colunas existem no dataset. Se o time de dados alterar o formato, o pipeline para com erro claro antes de treinar.

### Validacao de qualidade

34 regras de qualidade definidas no `quality.yaml`:

- Tabela: entre 40.000 e 55.000 linhas
- Numericas: ranges (idade 18-100, score 300-850), sem nulos
- Categoricas: valores permitidos (loan_status so 0/1, historico de calote so Yes/No)

Resultado: 33 regras passaram, 1 falhou - 7 registros com idade acima de 100 anos (outliers reais do dataset). Decidimos manter porque o modelo consegue lidar e sao dados reais.

### Limitacoes do dataset

- **Desbalanceamento:** 78% rejeitados, 22% aprovados. Tratado com StratifiedKFold e F1-Score em vez de accuracy.
- **Sem dados temporais:** nao e possivel analisar evolucao temporal nem drift real.
- **Outliers de idade:** 7 registros com idade acima de 100 anos.

### Pipeline de preprocessamento

Aplicamos transformacoes diferentes por tipo de coluna:
- **Numericas:** RobustScaler - normaliza usando mediana, resistente a valores extremos (renda tem outliers significativos)
- **Categoricas:** OneHotEncoder - transforma categorias em colunas 0/1

Resultado: 13 colunas originais viram 22 apos a transformacao.

O preprocessador aprende as estatisticas apenas dos dados de treino e aplica a mesma transformacao nos dados de teste, prevenindo vazamento de informacao.

---

## 4. Experimentacao

### Dois tipos de modelos

O pipeline suporta dois caminhos para incluir modelos:

**Modelos do scikit-learn (declarativos):** definidos no YAML com classe e parametros. O pipeline instancia automaticamente. O time de dados nao escreve codigo - so configura.

**Modelos customizados (codigo):** escritos em `src/custom_models.py`. Implementam a mesma interface (`fit` e `predict`). Referenciados no YAML da mesma forma. Isso permite logica especifica de negocio que nao existe em bibliotecas prontas.

Ambos passam pelas mesmas etapas: mesmos dados, mesma validacao, mesmo preprocessamento, mesmas metricas, mesmo registro no MLflow.

### Modelo customizado: regras de negocio

Implementamos um classificador deterministico como baseline. Ele aplica regras fixas que um analista de credito usaria manualmente:

- Se o cliente tem historico de calote → rejeitar
- Se a parcela compromete mais de 35% da renda → rejeitar
- Se o score de credito esta abaixo de 600 → rejeitar
- Caso contrario → aprovar

Este modelo nao usa estatistica - demonstra que regras fixas sao insuficientes para o problema (F1=0.00, nao aprovou ninguem corretamente). Isso justifica a necessidade de modelos de machine learning.

### Resultados

| Modelo | Tipo | F1 | Overfitting | Tempo |
|--------|------|:---:|:---:|:---:|
| SklearnDummy | Baseline sklearn | 0.0000 | - | 0.0s |
| CustomRuleBased | Baseline customizado | 0.0000 | - | 0.0s |
| SklearnPerceptron | Linear | 0.6641 | nenhum | 0.0s |
| SklearnDTNoRegularization | Arvore | 0.7722 | severo | 0.1s |
| SklearnDTRegularized | Arvore | 0.7836 | nenhum | 0.1s |
| SklearnDTOptimized | Arvore | 0.8055 | baixo | 5.4s |
| SklearnRandomForest | Ensemble | 0.8053 | baixo | 0.3s |
| **SklearnRFOptimized** | **Ensemble** | **0.8254** | **aceitavel** | **38s** |

### Evolucao

A progressao dos resultados demonstra que:

1. **Regras deterministicas nao funcionam** (F1=0.00): tanto o baseline automatico quanto as regras de negocio falharam em identificar bons clientes. Os dois tem 78% de acuracia (simplesmente rejeitam todo mundo), mas F1=0 - nao aprovaram ninguem.

2. **Modelo linear e insuficiente** (F1=0.66): o Perceptron tenta separar as classes com uma linha reta. Funciona parcialmente, mas o problema nao e linear.

3. **Arvore sem limites decora** (F1=0.77 mas gap=0.23): acerta tudo no treino, erra no teste. Regularizacao resolve (gap cai para 0.01).

4. **Ensemble melhora e estabiliza** (F1=0.83): 200 arvores votando juntas. Mais robusto que arvore unica.

### Rastreamento com MLflow

Cada modelo registra no MLflow: hiperparametros, metricas (F1, accuracy, precision, recall, AUC-ROC), tempo de treino, indicador de overfitting, e o modelo salvo. Modelos sao registrados no Model Registry com versionamento automatico.

---

## 5. Reducao de Dimensionalidade

### Por que testar reducao

Com 22 features apos o preprocessamento, a pergunta e: podemos obter resultados iguais ou melhores usando menos features? Menos features significam treino mais rapido e modelo mais simples.

### Escolha das tecnicas

Dentre PCA, LDA e t-SNE, escolhemos PCA e LDA:

- **PCA:** encontra os eixos onde os dados mais variam. Nao olha os rotulos. Pode ser aplicado em dados novos (`transform()`).
- **LDA:** encontra o eixo que melhor separa as classes. Usa os rotulos. Para problema com 2 classes, gera 1 unico eixo.
- **t-SNE:** descartado porque nao pode ser aplicado em dados novos (nao tem `transform()`). Serve apenas para visualizacao.

Ambas foram integradas como etapas do Pipeline do scikit-learn, garantindo que a reducao aprende apenas com dados de treino.

### Resultados

| Configuracao | Features | F1 | Tempo |
|-------------|:---:|:---:|:---:|
| Sem reducao | 22 | **0.8254** | 38s |
| Com PCA (95% variancia) | 17 | 0.7891 | 3s |
| Com LDA | 1 | 0.6643 | 1s |

### Por que NAO usamos reducao em producao

PCA reduziu de 22 para 17 features mas perdeu 3.6 pontos de F1. O ganho em tempo (38s para 3s) nao compensa a perda de performance neste caso.

LDA comprimiu 22 features em 1 unico numero. Perdeu 16 pontos de F1 - informacao demais descartada.

Com apenas 22 features, a dimensionalidade ja e gerenciavel. Reducao nao se justifica. Por isso, em producao, rodamos o modelo sem reducao. A reducao existe apenas na experimentacao como evidencia de que essa decisao foi tomada com base em dados, nao por omissao.

---

## 6. Selecao do Modelo Final

### Comparativo

| Modelo | F1 | Precision | Recall | Impacto Financeiro |
|--------|:---:|:---:|:---:|---:|
| **SklearnRFOptimized** | **0.8254** | 0.8968 | 0.7645 | **R$2.1M** |
| SklearnDTOptimized | 0.8055 | 0.8746 | 0.7465 | R$2.5M |
| SklearnRandomForest | 0.8053 | 0.9089 | 0.7230 | R$2.0M |
| SklearnPerceptron | 0.6641 | 0.8116 | 0.5620 | R$4.4M |
| CustomRuleBased | 0.0000 | 0.0000 | 0.0000 | R$2.8M |

### Analise financeira

Cada erro do modelo tem um custo diferente, calculado com valores reais do dataset:

| Tipo de Erro | O que acontece | Como calculamos | Formula |
|---|---|---|---|
| Falso Positivo (FP) | Aprovou cliente que deu calote | Banco perde o valor total do emprestimo | soma(loan_amnt) dos FPs |
| Falso Negativo (FN) | Rejeitou cliente que pagaria | Banco perde os juros que teria ganho | soma(loan_amnt x loan_int_rate / 100) dos FNs |

O prejuizo de FP e tipicamente maior que de FN porque:
- FP: perde 100% do valor emprestado (o dinheiro nao volta)
- FN: perde apenas os juros (~10-15% do valor)

Isso justifica a postura conservadora do modelo: e financeiramente mais seguro rejeitar um bom cliente (perder juros) do que aprovar um caloteiro (perder capital).

### Resultados financeiros detalhados

| Modelo | FP (qtd) | FN (qtd) | Prejuizo FP | Receita Perdida FN | Impacto Total |
|--------|:---:|:---:|---:|---:|---:|
| SklearnRandomForest | 145 | 554 | R$1.46M | R$0.54M | R$2.0M |
| SklearnRFOptimized | 176 | 471 | R$1.63M | R$0.47M | R$2.1M |
| SklearnDTOptimized | 214 | 507 | R$1.94M | R$0.52M | R$2.5M |
| SklearnDummy | 0 | 2000 | R$0 | R$2.84M | R$2.8M |
| CustomRuleBased | 0 | 2000 | R$0 | R$2.84M | R$2.8M |
| SklearnPerceptron | 261 | 876 | R$3.60M | R$0.83M | R$4.4M |
| SklearnRFLDA | 646 | 684 | R$6.34M | R$0.74M | R$7.1M |

### Trade-off entre RF Optimized e Random Forest

| Aspecto | RF Optimized | Random Forest |
|---------|:---:|:---:|
| F1 | 0.8254 (melhor) | 0.8053 |
| Recall | 76.45% (mais aprovacoes corretas) | 72.30% |
| Prejuizo FP | R$1.63M | R$1.46M (menor risco) |
| Receita Perdida FN | R$0.47M (menos oportunidade perdida) | R$0.54M |
| Impacto Total | R$2.1M | R$2.0M |

O RandomForest e mais conservador (rejeita mais, menos caloteiros aprovados). O RF Optimized e mais equilibrado (aprova mais bons clientes, mas aceita um pouco mais de risco).

### Justificativa da selecao

RF Optimized selecionado porque:
1. Maior F1 (0.8254) entre todos os modelos
2. A diferenca de impacto financeiro e pequena (R$100k vs RandomForest)
3. Maior recall (76.45% vs 72.30%) significa menos bons clientes rejeitados
4. AUC-ROC 0.975 (excelente capacidade de discriminacao)

Em escala, a diferenca de 2 pontos de F1 representa milhares de decisoes melhores. R$100k a mais de risco e aceitavel quando o modelo identifica significativamente mais bons clientes.

---

## 7. Operacionalizacao

### Persistencia e versionamento

Modelos sao salvos no MLflow e registrados no Model Registry com versionamento automatico. Qualquer versao pode ser recuperada pelo identificador do run.

### Inferencia

O sistema de inferencia carrega o modelo salvo, aplica o mesmo preprocessamento usado no treino, e retorna a decisao (aprovado/rejeitado) com probabilidade.

### Metricas de monitoramento

Definimos metricas em dois niveis:

**Tecnicas:** F1, accuracy, precision, recall, AUC-ROC, indicador de overfitting, tempo de treino.

**Negocio:** custo de emprestimos perdidos (falsos positivos), receita de juros perdida (falsos negativos), impacto financeiro total.

### Deteccao de drift

O sistema monitora mudancas nos dados comparando estatisticas (media) de cada coluna entre dados de treino e dados novos. Se a variacao superar 10%, gera alerta. Tambem verifica se apareceram categorias novas que o modelo nunca viu.

Na simulacao realizada, nenhum drift foi detectado - resultado esperado por usar o mesmo dataset dividido aleatoriamente.

### Analise pos-deploy

Apos cada execucao de producao, o sistema compara automaticamente a run atual com a anterior:

```
make post-deploy

  metric              previous        current         change
                      (032026)        (042026)
  f1_score              0.8254          0.8291        +0.0037
  accuracy              0.9281          0.9297        +0.0016
  STATUS: STABLE
```

Se o F1 cair mais de 0.02, o status muda para DEGRADED e recomenda investigar drift e re-avaliar modelos.

### Estrategia de re-treinamento

Ciclo mensal:
1. Time de dados atualiza a base com novos registros
2. `make prod` roda com dados acumulados (historicos + novos)
3. `make post-deploy` compara com a versao anterior
4. Se estavel, mantem. Se degradou, `make test` re-avalia todos os modelos.

O retreino e completo (nao incremental) porque queremos garantir que o modelo aprenda todos os padroes historicos, nao apenas os mais recentes.

---

## 8. Conclusao

### Principais decisoes

| Decisao | Por que |
|---------|---------|
| Scripts modulares | Cada peca pode ser reutilizada ou substituida independentemente |
| YAML + dataclasses | Mudar configuracao sem mexer no codigo. Validacao automatica. |
| Dois tipos de modelo (sklearn + custom) | Flexibilidade para usar modelos prontos ou implementar logica especifica |
| RobustScaler | Dataset tem outliers significativos em renda |
| PCA + LDA testados, nao usados em producao | Reducao piorou o resultado. Decisao baseada em evidencia. |
| Parquet local | Adequado ao escopo. Em producao: banco de dados ou data lake conforme infraestrutura. |
| SQLite para MLflow | Simples, local. Em producao: PostgreSQL ou similar. |
| Retreino completo mensal | Preserva historico. scikit-learn treina rapido o suficiente. |

### O que mudaria em ambiente de producao

- **Armazenamento:** Parquet local → banco de dados ou data lake
- **Orquestracao:** Makefile → Airflow ou similar para agendamento
- **Inferencia:** Script → API REST com endpoint HTTP
- **Monitoramento:** Script manual → dashboard automatizado
- **Testes:** Adicionar testes unitarios para cada modulo
- **Deploy:** Makefile → CI/CD automatizado (GitHub Actions)
- **Escala:** Local → containers com orquestracao

### Aprendizado

A principal mudanca neste projeto nao e tecnica - e de mentalidade. O foco deixa de ser maximizar uma metrica isolada e passa a ser garantir que o sistema e reprodutivel, rastreavel, e preparado para mudancas. O modelo e apenas uma peca de um sistema que precisa ser mantido, monitorado e atualizado continuamente.
