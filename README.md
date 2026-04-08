# ML Pipeline — Loan Approval Classification

Projeto de operacionalizacao de modelos de ML para classificacao de aprovacao de emprestimos.

Evolucao do [projeto anterior](archive/): notebook monolitico reorganizado em pipeline modular
com tracking de experimentos, validacao de qualidade, reducao de dimensionalidade,
inferencia e monitoramento de drift.

## Estrutura

```
├── config/                     Configuracoes (YAML)
│   ├── data.yaml               Contrato de dados: colunas, tipos, fonte
│   ├── pipeline.yaml           Execucao: paths, random_state, split, drift
│   ├── quality.yaml            Expectations de qualidade
│   ├── experiments_prod.yaml   Modelo de producao
│   └── experiments_test.yaml   Modelos de experimentacao
│
├── src/                        Pipeline (modulos chamados pelo orquestrador)
│   ├── config.py               Dataclasses que validam os YAMLs
│   ├── ingestion.py            Download + CSV → Parquet
│   ├── quality_checks.py       Validacao de qualidade
│   ├── preprocessing.py        ColumnTransformer + train/test split
│   ├── train.py                Treinamento com MLflow tracking
│   ├── evaluate.py             Comparacao de modelos + analise financeira
│   ├── serve.py                Servico de inferencia
│   ├── monitoring.py           Deteccao de drift
│   └── reduction.py            PCA/LDA pipeline helpers
│
├── models/                     Modelos customizados (DS edita aqui)
│   └── custom_models.py        RuleBasedClassifier (exemplo)
│
├── scripts/                    Ferramentas de apoio
│   ├── compare.py              Tabela comparativa no terminal + CSV
│   ├── promote.py              Promove modelo para producao
│   ├── simulate_months.py      Simula execucoes mensais
│   └── post_deploy.py          Compara run atual vs anterior
│
├── docs/                       Documentacao
│   ├── WALKTHROUGH.md          Referencia para apresentacao
│   ├── MANUAL_TESTES.md        Guia para data scientists
│   ├── CONCEITOS_AULA.md       Conceitos aplicados no projeto
│   └── DECISOES_ESTRUTURA.md   Justificativa da organizacao
│
├── archive/                    Projeto anterior preservado
├── run_pipeline.py             Orquestrador principal
├── Makefile                    CI/CD simulado
├── relatorio_tecnico.md        Relatorio tecnico (entregavel)
└── requirements.txt            Dependencias
```

## CI/CD Pipeline (simulado via Makefile)

### Deploy completo (primeira vez)

```bash
make setup                          # instala dependencias
make configs                        # valida configuracoes (gate)
make test                           # treina todos os modelos
make compare                        # compara resultados
make promote MODEL=SklearnRF_Optimized  # promove o melhor
make prod                           # treina em producao
make post-deploy                    # verifica estabilidade
```

### Manutencao mensal

```bash
make prod                           # retrain com dados novos
make post-deploy                    # compara com run anterior
```

### Todos os comandos

| Comando | O que faz |
|---------|-----------|
| `make setup` | Cria venv e instala dependencias |
| `make configs` | Valida todas as configs (gate) |
| `make test` | Pipeline completo em experimentacao (10 modelos) |
| `make prod` | Pipeline de producao (1 modelo) |
| `make compare` | Tabela comparativa (experimentacao) |
| `make compare-prod` | Tabela comparativa (producao) |
| `make promote MODEL=nome` | Move modelo para producao |
| `make post-deploy` | Compara run atual vs anterior |
| `make serve` | Inicia servico de inferencia |
| `make mlflow` | Abre MLflow UI (http://localhost:5000) |
| `make clean` | Apaga dados e resultados gerados |

## Modos de execucao

| Modo | YAML | Modelos | Reducao |
|------|------|:---:|:---:|
| `make test` (experimentacao) | experiments_test.yaml | 10 | PCA + LDA |
| `make prod` (producao) | experiments_prod.yaml | 1 | — |

## Resultados

| Modelo | F1 | Accuracy | AUC-ROC |
|--------|:---:|:---:|:---:|
| SklearnRF_Optimized | **0.8254** | 0.9281 | 0.9753 |
| SklearnDT_Optimized | 0.8055 | 0.9199 | 0.9586 |
| SklearnRandomForest | 0.8053 | 0.9223 | 0.9708 |
| SklearnRF_PCA | 0.7891 | 0.9124 | 0.9624 |
| SklearnPerceptron | 0.6641 | 0.8737 | — |
| CustomRuleBased | 0.0000 | 0.7778 | 0.5000 |

## Tecnologias

- Python 3.11+ / scikit-learn / MLflow / PyArrow / PyYAML
