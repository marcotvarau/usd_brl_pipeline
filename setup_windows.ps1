# ============================================================================
# USD/BRL Pipeline - Setup Script para Windows (PowerShell)
# ============================================================================
# Este script configura todo o ambiente do projeto em um venv isolado
# sem modificar o Python global do sistema
# ============================================================================

# Habilitar execução de scripts (executar como Admin se necessário)
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Cores para output
function Write-Info {
    Write-Host "[INFO]" -ForegroundColor Blue -NoNewline
    Write-Host " $args"
}

function Write-Success {
    Write-Host "[SUCCESS]" -ForegroundColor Green -NoNewline
    Write-Host " $args"
}

function Write-Warning {
    Write-Host "[WARNING]" -ForegroundColor Yellow -NoNewline
    Write-Host " $args"
}

function Write-Error-Message {
    Write-Host "[ERROR]" -ForegroundColor Red -NoNewline
    Write-Host " $args"
}

# Banner
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   USD/BRL Pipeline - Setup Automático" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Verificar Python
Write-Info "Verificando instalação do Python..."

try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $majorVersion = [int]$matches[1]
        $minorVersion = [int]$matches[2]
        Write-Success "Python $majorVersion.$minorVersion encontrado"
        
        if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 8)) {
            Write-Error-Message "Python 3.8 ou superior é necessário"
            exit 1
        }
    }
} catch {
    Write-Error-Message "Python não encontrado. Por favor, instale Python 3.8+"
    Write-Info "Download: https://www.python.org/downloads/"
    exit 1
}

# 2. Criar estrutura de diretórios
Write-Info "Criando estrutura de diretórios..."

$projectDir = "usd_brl_pipeline"

if (Test-Path $projectDir) {
    Write-Warning "Diretório $projectDir já existe."
    $response = Read-Host "Deseja sobrescrever? (y/n)"
    if ($response -ne 'y') {
        Write-Info "Abortando setup."
        exit 0
    }
    Remove-Item -Recurse -Force $projectDir
}

# Criar diretório principal
New-Item -ItemType Directory -Path $projectDir | Out-Null
Set-Location $projectDir

# Criar estrutura completa
$directories = @(
    "src\collectors",
    "src\processors",
    "src\validators",
    "src\exporters",
    "src\utils",
    "src\pipeline",
    "tests",
    "scripts",
    "notebooks",
    "data\raw",
    "data\processed",
    "data\features",
    "data\cache",
    "logs",
    "docs"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

Write-Success "Estrutura de diretórios criada"

# 3. Criar ambiente virtual
Write-Info "Criando ambiente virtual Python..."

python -m venv venv

if ($?) {
    Write-Success "Ambiente virtual criado"
} else {
    Write-Error-Message "Erro ao criar ambiente virtual"
    exit 1
}

# 4. Ativar venv e instalar dependências
Write-Info "Ativando ambiente virtual..."

& ".\venv\Scripts\Activate.ps1"

# 5. Atualizar pip
Write-Info "Atualizando pip..."
python -m pip install --upgrade pip setuptools wheel

# 6. Criar requirements.txt
Write-Info "Criando arquivo de dependências..."

@'
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

# Caching (Redis opcional no Windows)
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

# Visualization
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0

# Windows specific
pywin32==306
colorama==0.4.6
'@ | Out-File -FilePath "requirements.txt" -Encoding UTF8

Write-Success "requirements.txt criado"

# 7. Instalar dependências
Write-Info "Instalando dependências Python (isso pode levar alguns minutos)..."

pip install -r requirements.txt

if ($?) {
    Write-Success "Dependências instaladas com sucesso"
} else {
    Write-Warning "Algumas dependências podem ter falhado"
}

# 8. Criar .env.example
Write-Info "Criando arquivo de configuração..."

@'
# Federal Reserve Economic Data
# Obtenha sua chave gratuita em: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=your_fred_api_key_here

# Alpha Vantage (opcional)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=usd_brl_pipeline

# Redis Cache (opcional no Windows)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Email Notifications (opcional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECIPIENT=recipient@example.com
'@ | Out-File -FilePath ".env.example" -Encoding UTF8

Copy-Item ".env.example" ".env"
Write-Success "Arquivo .env criado (configure suas credenciais)"

# 9. Criar arquivos __init__.py
Write-Info "Criando arquivos __init__.py..."

$initFiles = @(
    "src\__init__.py",
    "src\collectors\__init__.py",
    "src\processors\__init__.py",
    "src\validators\__init__.py",
    "src\exporters\__init__.py",
    "src\utils\__init__.py",
    "src\pipeline\__init__.py",
    "tests\__init__.py"
)

foreach ($file in $initFiles) {
    New-Item -ItemType File -Path $file -Force | Out-Null
}

# 10. Criar scripts auxiliares

# Script de ativação (activate.bat)
@'
@echo off
echo Ativando ambiente virtual...
call venv\Scripts\activate.bat
echo Ambiente virtual ativado!
echo Python: %where python%
'@ | Out-File -FilePath "activate.bat" -Encoding ASCII

# Script de execução (run.bat)
@'
@echo off
call venv\Scripts\activate.bat
if not exist "scripts\run_pipeline.py" (
    echo Erro: scripts\run_pipeline.py nao encontrado.
    echo Certifique-se de ter copiado todos os arquivos do projeto.
    exit /b 1
)
python scripts\run_pipeline.py %*
'@ | Out-File -FilePath "run.bat" -Encoding ASCII

# Script de teste (test.bat)
@'
@echo off
call venv\Scripts\activate.bat
pytest tests\ -v --cov=src --cov-report=term-missing
'@ | Out-File -FilePath "test.bat" -Encoding ASCII

Write-Success "Scripts auxiliares criados"

# 11. Criar .gitignore
Write-Info "Criando .gitignore..."

@'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Windows
Thumbs.db
ehthumbs.db
Desktop.ini
$RECYCLE.BIN/
*.cab
*.msi
*.msm
*.msp
*.lnk

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Project specific
data/
logs/
*.log
.env
*.pkl
*.csv
*.parquet
*.h5

# Testing
.coverage
htmlcov/
.pytest_cache/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Distribution
dist/
build/
*.egg-info/
'@ | Out-File -FilePath ".gitignore" -Encoding UTF8

Write-Success ".gitignore criado"

# 12. Criar script de verificação
@'
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
        importlib.import_module(package_name.replace("-", "_"))
        return True
    except ImportError:
        return False

def check_dependencies():
    """Verifica dependências principais"""
    required = [
        "pandas", "numpy", "requests", "yfinance",
        "fredapi", "yaml", "pytest"
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
    
    if os.path.exists(".env"):
        print("✅ .env existe")
        
        with open(".env", "r") as f:
            content = f.read()
            if "your_fred_api_key_here" in content:
                print("⚠️  FRED_API_KEY ainda não foi configurada")
                print("   Obtenha sua chave em: https://fred.stlouisfed.org/docs/api/api_key.html")
            else:
                print("✅ FRED_API_KEY configurada")
    else:
        print("❌ .env não encontrado")
        print("   Execute: copy .env.example .env")
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
        print("2. Execute: run.bat")
    else:
        print("❌ Há problemas na configuração")
        print("\nExecute setup_windows.ps1 novamente")
    
    print("="*50)

if __name__ == "__main__":
    main()
'@ | Out-File -FilePath "verify_setup.py" -Encoding UTF8