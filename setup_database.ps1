# HealthGuard AI - Automated PostgreSQL Setup and Bootstrap Script
# =========================================================================
# Usage:
#   .\setup_database.ps1 -DbUser postgres -DbPassword <your_password> -DbName healthcare_fraud_db
# 
# Security Note:
#   Zero credentials or passwords are hardcoded in this script.
#   If -DbPassword is not provided as a parameter or in $env:PGPASSWORD,
#   you will be prompted securely for input.
# =========================================================================

param(
    [string]$DbUser = "postgres",
    [string]$DbPassword = "anok1234",
    [string]$DbHost = "localhost",
    [int]$DbPort = 5432,
    [string]$DbName = "healthcare_fraud_db",
    [string]$JwtSecret = ""
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  HealthGuard AI - Automated Database Bootstrap & Setup" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Resolve PostgreSQL Password securely without hardcoded fallbacks
if ([string]::IsNullOrWhiteSpace($DbPassword)) {
    if (-not [string]::IsNullOrWhiteSpace($env:PGPASSWORD)) {
        $DbPassword = $env:PGPASSWORD
    }
    elseif (-not [string]::IsNullOrWhiteSpace($env:DATABASE_URL)) {
        # Try extracting password from DATABASE_URL
        try {
            $uri = [System.Uri]$env:DATABASE_URL
            if (-not [string]::IsNullOrWhiteSpace($uri.UserInfo)) {
                $parts = $uri.UserInfo.Split(':')
                if ($parts.Length -gt 1) {
                    $DbPassword = $parts[1]
                }
            }
        }
        catch { }
    }
}

if ([string]::IsNullOrWhiteSpace($DbPassword)) {
    Write-Host "[?] PostgreSQL password not supplied in parameters or environment." -ForegroundColor Yellow
    $securePass = Read-Host "Enter PostgreSQL password for user '$DbUser'" -AsSecureString
    $bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePass)
    $DbPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
}

if ([string]::IsNullOrWhiteSpace($DbPassword)) {
    Write-Error "Error: Database password is required. Cannot proceed with setup."
    exit 1
}

# 2. Resolve JWT Secret Key securely (minimum 32 characters)
if ([string]::IsNullOrWhiteSpace($JwtSecret)) {
    if (-not [string]::IsNullOrWhiteSpace($env:JWT_SECRET_KEY)) {
        $JwtSecret = $env:JWT_SECRET_KEY
    }
    else {
        # Generate a cryptographically secure random 32-character key for local setup
        $bytes = New-Object byte[] 24
        [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $JwtSecret = [Convert]::ToBase64String($bytes).Replace('+', '-').Replace('/', '_').TrimEnd('=')
        Write-Host "[+] Generated cryptographically secure random JWT secret for local session." -ForegroundColor Green
    }
}

if ($JwtSecret.Length -lt 32) {
    Write-Error "Error: JWT secret must be at least 32 characters long."
    exit 1
}

$dbUrl = "postgresql://${DbUser}:${DbPassword}@${DbHost}:${DbPort}/${DbName}"
$env:DATABASE_URL = $dbUrl
$env:JWT_SECRET_KEY = $JwtSecret
$env:PYTHONPATH = "."

Write-Host "[1/3] Target Database: postgresql://${DbUser}:****@${DbHost}:${DbPort}/${DbName}" -ForegroundColor Yellow

# 3. Execute the python bootstrap script
Write-Host "[2/3] Initializing database, applying migrations, and seeding demo records..." -ForegroundColor Yellow
python backend/app/db/bootstrap.py --db-url $dbUrl

if ($LASTEXITCODE -eq 0) {
    Write-Host "[3/3] Setup and verification completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run the application with:" -ForegroundColor Cyan
    Write-Host "  Terminal 1 (Backend):" -ForegroundColor White
    Write-Host "    `$env:DATABASE_URL='$dbUrl'"
    Write-Host "    `$env:JWT_SECRET_KEY='$JwtSecret'"
    Write-Host "    python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"
    Write-Host ""
    Write-Host "  Terminal 2 (Frontend):" -ForegroundColor White
    Write-Host "    cd frontend; npm run dev"
}
else {
    Write-Host "[!] Setup failed. Please check your PostgreSQL credentials and server status." -ForegroundColor Red
    exit 1
}
