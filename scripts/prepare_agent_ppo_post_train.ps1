param(
    [string]$ZipPath = "",

    [int]$ModelId = 0,

    [string]$RepoRoot = "",

    [switch]$ForceRefresh
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Join-Path $PSScriptRoot ".."
}

$repo = (Resolve-Path -LiteralPath $RepoRoot).Path
$agentDir = Join-Path $repo "code\agent_ppo"
$ckptDir = Join-Path $agentDir "ckpt"
$configurePath = Join-Path $repo "code\conf\configure_app.toml"

if (-not (Test-Path -LiteralPath $agentDir)) {
    throw "agent_ppo dir not found: $agentDir"
}

if (-not (Test-Path -LiteralPath $configurePath)) {
    throw "configure_app.toml not found: $configurePath"
}

New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null

$resolvedCkptDir = (Resolve-Path -LiteralPath $ckptDir).Path
$resolvedAgentDir = (Resolve-Path -LiteralPath $agentDir).Path
if (-not $resolvedCkptDir.StartsWith($resolvedAgentDir, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to write checkpoint outside agent_ppo: $resolvedCkptDir"
}

function Get-ModelIdFromName {
    param([Parameter(Mandatory = $true)][string]$Name)
    if ($Name -match 'model\.ckpt-(\d+)\.pkl$') {
        return [int]$Matches[1]
    }
    return $null
}

function Get-UniqueLocalModelId {
    param([Parameter(Mandatory = $true)][string]$Dir)
    $files = @(Get-ChildItem -LiteralPath $Dir -Filter "model.ckpt-*.pkl" -File)
    if ($files.Count -eq 1) {
        return Get-ModelIdFromName -Name $files[0].Name
    }
    if ($files.Count -gt 1) {
        throw "Expected one local checkpoint in $Dir, found $($files.Count). Pass -ModelId explicitly."
    }
    return $null
}

function Get-UniqueArchiveModelId {
    param([Parameter(Mandatory = $true)][string]$ArchivePath)
    if (-not (Test-Path -LiteralPath $ArchivePath)) {
        throw "ZipPath not found: $ArchivePath"
    }
    $matches = @(tar -tf $ArchivePath | Where-Object { $_ -match '^ckpt/model\.ckpt-\d+\.pkl$' })
    if ($matches.Count -ne 1) {
        throw "Expected one ckpt/model.ckpt-*.pkl in archive, found $($matches.Count). Pass -ModelId explicitly."
    }
    return Get-ModelIdFromName -Name $matches[0]
}

if ($ModelId -eq 0) {
    $localModelId = Get-UniqueLocalModelId -Dir $ckptDir
    if ($null -ne $localModelId) {
        $ModelId = $localModelId
    }
    elseif (-not [string]::IsNullOrWhiteSpace($ZipPath)) {
        $ModelId = Get-UniqueArchiveModelId -ArchivePath $ZipPath
    }
    else {
        throw "No local checkpoint found. Provide -ZipPath or pass -ModelId with an existing checkpoint."
    }
}

$dst = Join-Path $ckptDir ("model.ckpt-$ModelId.pkl")

if ((Test-Path -LiteralPath $dst) -and $ForceRefresh) {
    if ([string]::IsNullOrWhiteSpace($ZipPath)) {
        throw "ForceRefresh requires -ZipPath so the checkpoint can be restored."
    }
    Remove-Item -LiteralPath $dst -Force
}

if (Test-Path -LiteralPath $dst) {
    Write-Host "Checkpoint already exists, keeping: $dst"
}
else {
    if ([string]::IsNullOrWhiteSpace($ZipPath)) {
        throw "Checkpoint not found: $dst. Provide -ZipPath to extract it."
    }
    if (-not (Test-Path -LiteralPath $ZipPath)) {
        throw "ZipPath not found: $ZipPath"
    }
    $archiveModelPath = "ckpt/model.ckpt-$ModelId.pkl"
    tar -xf $ZipPath -C $agentDir $archiveModelPath
}

if (-not (Test-Path -LiteralPath $dst)) {
    throw "Model file not found after extraction: $dst"
}

$content = Get-Content -LiteralPath $configurePath -Raw -Encoding UTF8
$content = $content -replace '(?m)^preload_model\s*=\s*(true|false)\s*$', 'preload_model = true'
$content = $content -replace '(?m)^preload_model_dir\s*=\s*".*"\s*$', 'preload_model_dir = "agent_ppo/ckpt"'
$content = $content -replace '(?m)^preload_model_id\s*=\s*\d+\s*$', "preload_model_id = 0"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($configurePath, $content, $utf8NoBom)

Get-Item -LiteralPath $dst | Select-Object FullName, Length, LastWriteTime
Select-String -LiteralPath $configurePath -Pattern 'preload_model|preload_model_dir|preload_model_id'
