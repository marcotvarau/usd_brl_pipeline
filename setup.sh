#!/bin/bash

# ============================================================================
# USD/BRL Pipeline - Setup Script com Ambiente Virtual
# ============================================================================
# Este script configura todo o ambiente do projeto em um venv isolado
# sem modificar o Python global do sistema
# ============================================================================

set -e  # Exit on error

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para imprimir com cores
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
echo "============================================"
echo "   USD/BRL Pipeline - Setup Automático"
echo "============================================"
echo ""

# 1. Verificar Python
print_info "Verificando instalação do Python..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 não encontrado. Por favor, instale Python 3.8 ou superior."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_success "Python $PYTHON_VERSION encontrado"

# Verificar versão mínima (3.8)
REQUIRED_VERSION="3.8"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python 3.8 ou superior é necessário. Versão atual: $PYTHON_VERSION"
    exit 1
fi

# 2. Criar estrutura de diretórios
print_info "Criando estrutura de diretórios..."

# Diretório principal do projeto
PROJECT_DIR="usd_brl_pipeline"

if [ -d "$PROJECT_DIR" ]; then
    print_warning "Diretório $PROJECT_DIR já existe."
    read -p "Deseja sobrescrever? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Abortando setup."
        exit 0
    fi
    rm -rf "$PROJECT_DIR"
fi

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Criar estrutura completa
mkdir -p src/collectors
mkdir -p src/processors  
mkdir -p src/validators
mkdir -p src/exporters
mkdir -p src/utils
mkdir -p src/pipeline
mkdir -p tests
mkdir -p scripts
mkdir -p notebooks
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/features
mkdir -p data/cache
mkdir -p logs
mkdir -p docs

print_success "Estrutura de diretórios criada"

# 3. Criar ambiente virtual
print_info "Criando ambiente virtual Python..."

python3 -m venv venv

# Ativar venv
source venv/bin/activate

print_success "Ambiente virtual criado e ativado"

# 4. Atualizar pip
print_info "Atualizando pip..."
pip install --upgrade pip setuptools wheel

# 5. Criar arquivo requirements.txt
print_info "Criando arquivo de dependências..."

cat > requirements.txt << 'EOF'
# Core dependencies
pandas==2.1.4
numpy==1.24.3
python-dotenv==1.0.0
pyyaml==6.0.1
sqlalchemy==2.0.23

# Data collection
requests==2.31.0
yfinance==0.2.33
fredapi==0.5.1
python-bcb==0.1.9
beautifulsoup4==4.12.2
lxml==4.9.3

# Data processing
scipy==1.11.4
scikit-learn==1.3.2
statsmodels==0.14.1
ta==0.10.2

# Validation
great-expectations==0.18.7
pandera==0.17.2
pydantic==2.5.3

# Caching and persistence
redis==5.0.1
pyarrow==14.0.2
fastparquet==2023.10.1
tables==3.9.2
openpyxl==3.1.2

# Scheduling
schedule==1.2.0
apscheduler==3.10.4
aiohttp==3.9.1

# Logging
loguru==0.7.2
python-json-logger==2.0.7

# Reliability
tenacity==8.2.3
pybreaker==1.0.1
backoff==2.2.1

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-mock==3.12.0

# Development
black==23.12.1
flake8==6.1.0
ipykernel==6.27.1
jupyter==1.0.0

# Visualization (optional)
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0

# Additional
coloredlogs==15.0.1
EOF

print_success "requirements.txt criado"

# 6. Instalar dependências
print_info "Instalando dependências Python (isso pode levar alguns minutos)..."

pip install -r requirements.txt

if [ $? -eq 0 ]; then
    print_success "Todas as dependências instaladas com sucesso"
else
    print_warning "Algumas dependências podem ter falhado. Verifique os erros acima."
fi

# 7. Criar arquivo .env.example
print_info "Criando arquivo de configuração de ambiente..."

cat > .env.example << 'EOF'
# Federal Reserve Economic Data
# Obtenha sua chave gratuita em: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=your_fred_api_key_here

# Alpha Vantage (opcional)
# Obtenha em: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=usd_brl_pipeline

# Redis Cache (opcional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Email Notifications (opcional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECIPIENT=recipient@example.com

# Slack Notifications (opcional)
SLACK_WEBHOOK=your_slack_webhook_url
EOF

# Copiar .env.example para .env
cp .env.example .env

print_success "Arquivo .env criado (configure suas credenciais)"

# 8. Criar __init__.py files
print_info "Criando arquivos __init__.py..."

touch src/__init__.py
touch src/collectors/__init__.py
touch src/processors/__init__.py
touch src/validators/__init__.py
touch src/exporters/__init__.py
touch src/utils/__init__.py
touch src/pipeline/__init__.py
touch tests/__init__.py

# 9. Criar script de ativação
print_info "Criando scripts auxiliares..."

cat > activate.sh << 'EOF'
#!/bin/bash
# Script para ativar o ambiente virtual

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Ambiente virtual ativado!"
    echo "Python: $(which python)"
    echo "Pip: $(which pip)"
else
    echo "Erro: Ambiente virtual não encontrado."
    echo "Execute ./setup.sh primeiro."
fi
EOF

chmod +x activate.sh

# 10. Criar script de execução rápida
cat > run.sh << 'EOF'
#!/bin/bash
# Script para executar o pipeline rapidamente

# Ativar venv
source venv/bin/activate

# Verificar se o script principal existe
if [ ! -f "scripts/run_pipeline.py" ]; then
    echo "Erro: scripts/run_pipeline.py não encontrado."
    echo "Certifique-se de ter copiado todos os arquivos do projeto."
    exit 1
fi

# Executar pipeline
python scripts/run_pipeline.py "$@"
EOF

chmod +x run.sh

# 11. Criar script de teste
cat > test.sh << 'EOF'
#!/bin/bash
# Script para executar testes

source venv/bin/activate
pytest tests/ -v --cov=src --cov-report=term-missing
EOF

chmod +x test.sh

# 12. Criar Makefile para comandos comuns
cat > Makefile << 'EOF'
.PHONY: help setup activate install run test clean lint format

help:
	@echo "Comandos disponíveis:"
	@echo "  make setup    - Configura o ambiente completo"
	@echo "  make activate - Ativa o ambiente virtual"
	@echo "  make install  - Instala/atualiza dependências"
	@echo "  make run      - Executa o pipeline"
	@echo "  make test     - Executa os testes"
	@echo "  make clean    - Limpa arquivos temporários"
	@echo "  make lint     - Verifica código com flake8"
	@echo "  make format   - Formata código com black"

setup:
	./setup.sh

activate:
	@echo "Execute: source venv/bin/activate"

install:
	source venv/bin/activate && pip install -r requirements.txt

run:
	source venv/bin/activate && python scripts/run_pipeline.py

test:
	source venv/bin/activate && pytest tests/ -v

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf data/cache/*

lint:
	source venv/bin/activate && flake8 src/ tests/

format:
	source venv/bin/activate && black src/ tests/ scripts/
EOF

# 13. Criar .gitignore
print_info "Criando .gitignore..."

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Project specific
data/
logs/
*.log
.env
*.pkl
*.csv
*.parquet
*.h5
*.hdf5

# Testing
.coverage
htmlcov/
.pytest_cache/
.tox/

# Documentation
docs/_build/
*.rst~

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Distribution
dist/
build/
*.egg-info/
EOF

print_success ".gitignore criado"

# 14. Criar script de verificação do ambiente
cat > check_env.py << 'EOF'
#!/usr/bin/env python3
"""Verifica se o ambiente está configurado corretamente"""

import sys
import importlib
import os

def check_python_version():
    """Verifica versão do Python"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ é necessário")
        return False
    print("✅ Python version OK")
    return True

def check_package(package_name):
    """Verifica se um pacote está instalado"""
    try:
        importlib.import_module(package_name.replace('-', '_'))
        return True
    except ImportError:
        return False

def check_dependencies():
    """Verifica dependências principais"""
    required = [
        'pandas', 'numpy', 'requests', 'yfinance',
        'fredapi', 'yaml', 'redis', 'pytest'
    ]
    
    print("\nVerificando dependências:")
    all_ok = True
    
    for package in required:
        if check_package(package):
            print(f"✅ {package}")
        else:
            print(f"❌ {package} não encontrado")
            all_ok = False
    
    return all_ok

def check_env_file():
    """Verifica arquivo .env"""
    print("\nVerificando arquivo .env:")
    
    if os.path.exists('.env'):
        print("✅ .env existe")
        
        # Verificar se FRED_API_KEY está configurada
        with open('.env', 'r') as f:
            content = f.read()
            if 'your_fred_api_key_here' in content:
                print("⚠️  FRED_API_KEY ainda não foi configurada")
                print("   Obtenha sua chave em: https://fred.stlouisfed.org/docs/api/api_key.html")
            else:
                print("✅ FRED_API_KEY configurada")
    else:
        print("❌ .env não encontrado")
        print("   Execute: cp .env.example .env")
        return False
    
    return True

def main():
    print("="*50)
    print("   Verificação do Ambiente USD/BRL Pipeline")
    print("="*50)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_env_file()
    ]
    
    print("\n" + "="*50)
    if all(checks):
        print("✅ Ambiente configurado corretamente!")
        print("\nPróximos passos:")
        print("1. Configure suas chaves de API em .env")
        print("2. Execute: ./run.sh")
    else:
        print("❌ Há problemas na configuração")
        print("\nExecute ./setup.sh para configurar o ambiente")
    
    print("="*50)

if __name__ == "__main__":
    main()
EOF

chmod +x check_env.py

# 15. Criar README de início rápido
cat > QUICK_START.md << 'EOF'
# 🚀 Quick Start - USD/BRL Pipeline

## 1️⃣ Ativação do Ambiente Virtual

```bash
# Ativar o ambiente virtual
source venv/bin/activate

# Ou use o script auxiliar
./activate.sh
```

## 2️⃣ Configuração das APIs

Edite o arquivo `.env` com suas credenciais:

```bash
# Abrir arquivo para edição
nano .env

# Adicionar sua chave FRED (obrigatória)
FRED_API_KEY=sua_chave_aqui
```

📌 **Obter chave FRED gratuita**: https://fred.stlouisfed.org/docs/api/api_key.html

## 3️⃣ Execução do Pipeline

```bash
# Execução básica (últimos 5 dias)
./run.sh

# Ou diretamente com Python
python scripts/run_pipeline.py

# Com parâmetros específicos
./run.sh --start-date 2023-01-01 --end-date 2023-12-31
```

## 4️⃣ Comandos Úteis

```bash
# Verificar ambiente
python check_env.py

# Executar testes
./test.sh

# Limpar cache
make clean

# Ver todas as opções
make help
```

## 5️⃣ Estrutura de Dados

Os dados processados ficam em:
- `data/features/` - Dataset final para ML
- `data/processed/` - Dados processados
- `data/raw/` - Dados brutos
- `logs/` - Logs de execução

## 📊 Visualização dos Dados

```python
# Jupyter notebook para explorar
jupyter notebook

# Ou carregue diretamente
import pandas as pd
df = pd.read_csv('data/features/usd_brl_all_*.csv')
print(df.head())
```

## 🆘 Problemas Comuns

### Ambiente virtual não ativa
```bash
# Recriar ambiente
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Erro de API Key
```bash
# Verificar se .env existe e está configurado
cat .env | grep FRED_API_KEY
```

### Memória insuficiente
```bash
# Executar com período menor
./run.sh --start-date 2023-01-01 --end-date 2023-03-31
```
EOF

# 16. Resumo final
echo ""
echo "============================================"
print_success "Setup concluído com sucesso!"
echo "============================================"
echo ""
print_info "Ambiente virtual criado em: $(pwd)/venv"
print_info "Próximos passos:"
echo ""
echo "  1. Ative o ambiente virtual:"
echo "     ${GREEN}source venv/bin/activate${NC}"
echo ""
echo "  2. Configure suas credenciais de API:"
echo "     ${GREEN}nano .env${NC}"
echo "     (adicione sua FRED_API_KEY)"
echo ""
echo "  3. Verifique a instalação:"
echo "     ${GREEN}python check_env.py${NC}"
echo ""
echo "  4. Execute o pipeline:"
echo "     ${GREEN}./run.sh${NC}"
echo ""
echo "============================================"
print_info "Scripts auxiliares criados:"
echo "  - ${BLUE}activate.sh${NC} - Ativa o ambiente virtual"
echo "  - ${BLUE}run.sh${NC} - Executa o pipeline"
echo "  - ${BLUE}test.sh${NC} - Executa os testes"
echo "  - ${BLUE}check_env.py${NC} - Verifica a configuração"
echo "  - ${BLUE}Makefile${NC} - Comandos make"
echo "============================================"
echo ""

# Salvar caminho do projeto
echo "export USD_BRL_PIPELINE_HOME=$(pwd)" >> venv/bin/activate

print_warning "IMPORTANTE: Este terminal já está com o venv ativado!"
print_info "Para novos terminais, use: source $(pwd)/venv/bin/activate"