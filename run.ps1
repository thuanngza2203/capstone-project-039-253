# Bat ca he thong bang Docker (run.cmd goi file nay).
# Moi lan thue GPU Vast, IP va cong SSH deu doi: dan lenh SSH cua Vast (nut Connect) khi duoc hoi,
# script tu tach IP/cong ghi vao .env roi bat lai tunnel. Enter = giu IP/cong dang luu.

Set-Location $PSScriptRoot
$envFile = Join-Path $PSScriptRoot ".env"

if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $PSScriptRoot ".env.example") $envFile
    Write-Host "Da tao .env tu .env.example: kiem tra VAST_SSH_KEY_FILE va HF_CACHE_DIR trong file nay." -ForegroundColor Yellow
}

function Read-EnvValue([string]$Name) {
    foreach ($line in [System.IO.File]::ReadAllLines($envFile)) {
        if ($line -match "^\s*$Name\s*=\s*(.*)$") { return $Matches[1].Trim() }
    }
    return ""
}

function Write-EnvValues([hashtable]$Values) {
    $lines = [System.Collections.Generic.List[string]]::new([string[]][System.IO.File]::ReadAllLines($envFile))
    foreach ($name in $Values.Keys) {
        $entry = "$name=$($Values[$name])"
        $index = -1
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match "^\s*$name\s*=") { $index = $i; break }
        }
        if ($index -ge 0) { $lines[$index] = $entry } else { $lines.Add($entry) }
    }
    # UTF-8 khong BOM, xuong dong LF: docker compose doc sai dong dau tien neu file co BOM.
    [System.IO.File]::WriteAllText($envFile, ($lines -join "`n") + "`n", [System.Text.UTF8Encoding]::new($false))
}

function ConvertFrom-VastSsh([string]$Text) {
    # Lenh Vast dang: ssh -p 59921 root@92.180.27.84 -L 8080:localhost:8080
    # Hoac viet gon: 92.180.27.84:59921 / 92.180.27.84 59921
    $text = $Text.Trim().Trim('"').Trim()
    $sshUser = "root"; $sshHost = ""; $sshPort = ""
    if ($text -match '(?:^|\s)-p\s*(\d+)') { $sshPort = $Matches[1] }
    if ($text -match '([A-Za-z0-9_.\-]+)@([A-Za-z0-9.\-]+)') { $sshUser = $Matches[1]; $sshHost = $Matches[2] }
    if (-not $sshHost -and $text -match '^([A-Za-z0-9.\-]+)[\s:]+(\d+)$') { $sshHost = $Matches[1]; $sshPort = $Matches[2] }
    if (-not $sshHost -or -not $sshPort -or [int]$sshPort -lt 1 -or [int]$sshPort -gt 65535) { return $null }
    return @{ VAST_SSH_USER = $sshUser; VAST_SSH_HOST = $sshHost; VAST_SSH_PORT = $sshPort }
}

# ---------------------------------------------------------------- IP/cong Vast
if (Read-EnvValue "VAST_SSH_HOST") {
    Write-Host ("Vast dang luu: {0}@{1}, cong SSH {2}" -f (Read-EnvValue "VAST_SSH_USER"), (Read-EnvValue "VAST_SSH_HOST"), (Read-EnvValue "VAST_SSH_PORT"))
} else {
    Write-Host "Vast dang luu: (chua co)"
}
# run.cmd dua tham so dong lenh vao bien nay (vi du: run.cmd ssh -p 59921 root@92.180.27.84).
$text = "$env:VAST_SSH_INPUT".Trim()
if (-not $text) {
    $text = "$(Read-Host 'Dan lenh SSH cua Vast (nut Connect), Enter de giu nguyen')".Trim()
}
if ($text) {
    $parsed = ConvertFrom-VastSsh $text
    if (-not $parsed) {
        Write-Host "Khong doc duoc IP/cong tu: $text" -ForegroundColor Red
        Write-Host "Vi du hop le: ssh -p 59921 root@92.180.27.84 -L 8080:localhost:8080"
        exit 1
    }
    Write-EnvValues $parsed
    Write-Host ("Da luu vao .env: {0}@{1}, cong SSH {2}" -f $parsed.VAST_SSH_USER, $parsed.VAST_SSH_HOST, $parsed.VAST_SSH_PORT) -ForegroundColor Green
}
if (-not (Read-EnvValue "VAST_SSH_HOST") -or -not (Read-EnvValue "VAST_SSH_PORT")) {
    Write-Host "Chua co IP/cong SSH cua Vast: khong mo tunnel, chat chi dung duoc Gemini hoac Ollama." -ForegroundColor Yellow
}

# ---------------------------------------------------------------- Docker
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Dang mo Docker Desktop, doi mot chut..."
    Start-Process "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    $deadline = (Get-Date).AddMinutes(3)
    do {
        Start-Sleep -Seconds 3
        docker info *> $null
    } while ($LASTEXITCODE -ne 0 -and (Get-Date) -lt $deadline)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker Desktop chua san sang sau 3 phut." -ForegroundColor Red
        exit 1
    }
}

# --build: chi build lai phan da sua. Doi IP Vast thi chi container vast-tunnel duoc tao lai.
docker compose up -d --build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Khoi dong that bai, xem loi o tren." -ForegroundColor Red
    exit 1
}

$webPort = Read-EnvValue "WEB_PORT"
if (-not $webPort) { $webPort = "5173" }
Write-Host ""
Write-Host "Da bat. Web: http://localhost:$webPort" -ForegroundColor Green
Write-Host "RAG can khoang 1 phut de nap model. Xem log: docker compose logs -f rag"
Start-Process "http://localhost:$webPort"
