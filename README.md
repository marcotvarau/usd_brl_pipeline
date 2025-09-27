# 🚀 USD/BRL Machine Learning Pipeline

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](docs/)

Sistema completo de coleta, processamento e estruturação de dados para predição da taxa de câmbio USD/BRL usando machine learning. Pipeline de produção com coleta automática de dados brasileiros e internacionais, feature engineering avançado e validação rigorosa.

## 📋 Índice

- [Características](#-características)
- [Arquitetura](#-arquitetura)
- [Instalação](#-instalação)
- [Configuração](#-configuração)
- [Uso](#-uso)
- [Estrutura de Dados](#-estrutura-de-dados)
- [API Documentation](#-api-documentation)
- [Desenvolvimento](#-desenvolvimento)
- [Troubleshooting](#-troubleshooting)
- [Contribuindo](#-contribuindo)

## ✨ Características

### 📊 Coleta de Dados
- **Fontes Brasileiras Oficiais**:
  - Banco Central do Brasil (BCB): PTAX, SELIC, Focus
  - IBGE: IPCA, indicadores econômicos
  - B3: Dados de mercado

- **Fontes Internacionais**:
  - Federal Reserve (FRED): Fed Funds, inflação US, GDP
  - Yahoo Finance: Commodities, índices, moedas
  - COT Reports: Posicionamento institucional

### 🔧 Feature Engineering
- **150+ features** organizadas em 4 tiers por importância preditiva
- Indicadores econômicos customizados (diferencial de juros real, carry trade)
- Índice de commodities ponderado por exportações brasileiras
- Indicadores técnicos otimizados para forex
- Features temporais e sazonalidade brasileira

### ✅ Qualidade e Confiabilidade
- Validação automática com Great Expectations
- Detecção de data drift e anomalias
- Circuit breaker para proteção contra falhas de API
- Cache Redis com fallback local
- Logging estruturado em JSON

### 🚀 Performance e Escalabilidade
- Processamento paralelo de múltiplas fontes
- Chunked processing para grandes volumes
- Compressão automática de dados
- Suporte a múltiplos formatos (CSV, Parquet, HDF5)

## 🏗 Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Sources Layer                       │
├─────────────────┬─────────────────┬────────────────────────────┤
│   BCB API       │   FRED API      │   Yahoo Finance            │
│   (PTAX, SELIC) │   (Fed, CPI)    │   (DXY, Commodities)       │
└─────────────────┴─────────────────┴────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Collection Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  BCBCollector│  │ FREDCollector│  │YahooCollector│         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                    Circuit Breaker + Retry Logic                │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Processing Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │DataCleaner   │  │FeatureEngineer│ │TechnicalCalc │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Validation Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │QualityValidator│ │DriftMonitor  │  │IntegrityCheck│         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Output Layer                               │
│     CSV / Parquet / HDF5 / PostgreSQL / MongoDB                │
└─────────────────────────────────────────────────────────────────┘
```

## 💻 Instalação

### Pré-requisitos
- Python 3.8+
- Redis (opcional, para cache distribuído)
- PostgreSQL (opcional, para armazenamento)

### Instalação Rápida

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/usd-brl-pipeline.git
cd usd-brl-pipeline

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale dependências
pip install -r requirements.txt

# Configure variáveis de ambiente
cp .env.example .env
# Edite .env com suas credenciais de API
```

### Instalação com Docker

```bash
# Build da imagem
docker build -t usd-brl-pipeline .

# Execute o container
docker run -d \
  --name usd-brl-pipeline \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  usd-brl-pipeline
```

## ⚙️ Configuração

### Credenciais de API

Edite o arquivo `.env`:

```bash
# Federal Reserve Economic Data
FRED_API_KEY=your_fred_api_key_here

# Alpha Vantage (opcional)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# Database
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=usd_brl_pipeline

# Redis Cache
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Email Notifications
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECIPIENT=recipient@example.com
```

### Configuração do Pipeline

Edite `config.yaml` para customizar:

```yaml
collection:
  start_date: "2014-01-01"  # Data inicial
  frequency: "daily"         # Frequência de coleta
  cache_ttl_hours: 1        # TTL do cache

features:
  lag_periods: [1, 2, 5, 10, 22, 44, 66]
  rolling_windows: [5, 10, 22, 44, 66]
  
validation:
  max_missing_pct: 5.0      # Máximo de dados faltantes
  ranges:
    usd_brl:
      min: 1.0
      max: 10.0
```

## 🚀 Uso

### Execução Básica

```bash
# Executar pipeline completo (últimos 5 dias)
python scripts/run_pipeline.py

# Executar para período específico
python scripts/run_pipeline.py --start-date 2023-01-01 --end-date 2023-12-31

# Executar apenas coletores específicos
python scripts/run_pipeline.py --collectors bcb yahoo fred

# Forçar atualização ignorando cache
python scripts/run_pipeline.py --force-refresh
```

### Modos de Execução

```bash
# Modo backfill (dados históricos)
python scripts/run_pipeline.py --mode backfill --start-date 2014-01-01

# Modo validação (verificar qualidade)
python scripts/run_pipeline.py --mode validate

# Modo repair (corrigir problemas)
python scripts/run_pipeline.py --mode repair

# Modo dry-run (teste sem exportar)
python scripts/run_pipeline.py --dry-run
```

### Agendamento Automático

```bash
# Adicionar ao crontab para execução diária às 19h
0 19 * * * /path/to/venv/bin/python /path/to/run_pipeline.py --mode daily

# Ou usar o scheduler interno
python scripts/scheduler.py
```

### Uso Programático

```python
from src.pipeline.orchestrator import PipelineOrchestrator

# Inicializar pipeline
orchestrator = PipelineOrchestrator('config.yaml')

# Executar coleta
results = orchestrator.run_pipeline(
    start_date='2023-01-01',
    end_date='2023-12-31',
    collectors=['bcb', 'fred', 'yahoo']
)

# Acessar dados
data = results['data']
print(f"Shape: {data.shape}")
print(f"Features: {data.columns.tolist()}")
print(f"Date range: {data.index.min()} to {data.index.max()}")
```

## 📊 Estrutura de Dados

### Features por Tier

#### Tier 1 (40% poder preditivo)
- `usd_brl_ptax_close`: Taxa PTAX oficial
- `selic_rate`: Taxa SELIC
- `fed_funds_rate`: Fed Funds Rate
- `dxy_index`: US Dollar Index
- `real_interest_differential`: Diferencial de juros real
- `usd_brl_implied_volatility`: Volatilidade implícita

#### Tier 2 (30% poder preditivo)
- `brazilian_commodity_index`: Índice customizado de commodities
- `ipca_monthly_yoy`: Inflação brasileira
- `us_cpi_yoy`: Inflação americana
- `capital_flows_net`: Fluxo líquido de capital
- `brazil_fiscal_balance`: Balanço fiscal

#### Tier 3 (20% poder preditivo)
- `risk_sentiment_score`: Score de sentimento de risco
- `carry_trade_attractiveness`: Atratividade do carry trade
- `cot_sentiment_aggregate`: Sentiment COT agregado
- `correlation_emerging_markets`: Correlação com emergentes

#### Tier 4 (10% poder preditivo)
- Indicadores técnicos (RSI, MACD, Bollinger Bands)
- Features temporais e sazonais
- Padrões de preço

### Exemplo de Dataset

```csv
date,usd_brl_ptax_close,selic_rate,fed_funds_rate,real_interest_differential,risk_sentiment_score
2023-01-02,5.2194,13.75,4.33,8.21,42.1
2023-01-03,5.2567,13.75,4.33,8.18,43.8
2023-01-04,5.2145,13.75,4.33,8.19,41.9
```

## 📖 API Documentation

### Collectors

```python
# BCB Collector
from src.collectors.bcb_collector import BCBCollector

collector = BCBCollector(config)
data = collector.collect(
    start_date='2023-01-01',
    end_date='2023-12-31',
    series=['selic_daily', 'ipca_monthly']
)
```

### Feature Engineering

```python
from src.processors.feature_engineer import FeatureEngineer

engineer = FeatureEngineer(config)
features = engineer.engineer_all_features(raw_data)

# Features específicas
interest_diff = engineer.create_interest_rate_features(data)
commodity_index = engineer.create_commodity_index(data)
risk_score = engineer.create_risk_sentiment_score(data)
```

### Validation

```python
from src.validators.quality_validator import QualityValidator

validator = QualityValidator(config)
results = validator.validate(data)

if not results['passed']:
    print(f"Validation issues: {results['issues']}")
```

## 🧪 Testes

```bash
# Executar todos os testes
pytest

# Testes com coverage
pytest --cov=src --cov-report=html

# Testes específicos
pytest tests/test_collectors.py
pytest tests/test_feature_engineering.py

# Testes de integração
pytest tests/integration/ --slow
```

## 🛠 Desenvolvimento

### Estrutura do Projeto

```
usd_brl_pipeline/
├── src/
│   ├── collectors/       # Coletores de dados
│   ├── processors/       # Processamento e features
│   ├── validators/       # Validação de qualidade
│   ├── exporters/        # Exportação de dados
│   ├── utils/           # Utilitários
│   └── pipeline/        # Orquestração
├── tests/               # Testes unitários
├── scripts/             # Scripts de execução
├── notebooks/           # Análise exploratória
├── data/               # Dados (git-ignored)
├── logs/               # Logs (git-ignored)
└── docs/               # Documentação
```

### Adicionando Novos Coletores

1. Crie uma classe herdando de `BaseCollector`
2. Implemente os métodos `collect()` e `validate_data()`
3. Adicione ao orchestrator
4. Escreva testes

Exemplo:

```python
from src.collectors.base_collector import BaseCollector

class MyCollector(BaseCollector):
    def _initialize(self):
        # Setup específico
        pass
    
    def collect(self, start_date, end_date):
        # Implementar coleta
        pass
    
    def validate_data(self, data):
        # Implementar validação
        pass
```

## 🐛 Troubleshooting

### Problemas Comuns

#### Erro de API Key
```
Error: FRED API key not found
Solution: Configure FRED_API_KEY in .env file
```

#### Dados faltantes
```
Warning: Excessive missing data
Solution: Check API availability, extend date range, or use --force-refresh
```

#### Memória insuficiente
```
MemoryError during processing
Solution: Reduce chunk_size in config.yaml or process smaller date ranges
```

### Logs

Verifique os logs para diagnóstico:

```bash
# Log principal
tail -f logs/pipeline.log

# Apenas erros
tail -f logs/errors.log

# Validação
tail -f logs/validation.log
```

## 📈 Performance

### Benchmarks

- **Coleta de dados**: ~2-5 min para 1 ano de dados
- **Feature engineering**: ~30s para 10.000 linhas
- **Validação completa**: ~10s para dataset completo
- **Export CSV**: ~5s para 100MB

### Otimizações

- Use cache Redis para ambientes distribuídos
- Configure `chunk_size` baseado na memória disponível
- Use Parquet para datasets grandes
- Habilite compressão para reduzir I/O

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Guidelines

- Siga PEP 8 e use Black para formatação
- Adicione testes para novas funcionalidades
- Atualize a documentação
- Mantenha compatibilidade com Python 3.8+

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- Banco Central do Brasil pela API pública
- Federal Reserve pelo FRED API
- Comunidade Python pelos excelentes pacotes
- Contribuidores do projeto

## 📬 Contato

Para questões, sugestões ou suporte:

- Issues: [GitHub Issues](https://github.com/seu-usuario/usd-brl-pipeline/issues)
- Email: seu-email@example.com

---

**Desenvolvido com ❤️ para a comunidade de Data Science brasileira**