# GorgeChase Release Packaging Script
# Builds a public-safe release archive or directory by default.

param(
    [string]$OutputDir = (Join-Path (Split-Path $PSScriptRoot -Parent) "release"),
    [string]$ArchiveName = "gorgechase-release",
    [switch]$UseZip = $false,
    [switch]$IncludeCheckpoints = $false
)

$RepoRoot = Split-Path $PSScriptRoot -Parent
$CodeDir = Join-Path $RepoRoot "code"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ReleaseName = "$ArchiveName-$Timestamp"
$ReleasePath = Join-Path $OutputDir $ReleaseName

$ItemsToCopy = @(
    "conf",
    "agent_ppo",
    ".vscode",
    "kaiwu.json",
    "train_test.py"
)

function Test-ReleaseExcluded {
    param([System.IO.FileSystemInfo]$Item)

    $name = $Item.Name
    $ext = $Item.Extension.ToLowerInvariant()

    if ($Item.PSIsContainer -and @("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv") -contains $name) {
        return $true
    }

    if (-not $IncludeCheckpoints -and ($name -eq "ckpt" -or $ext -eq ".pkl")) {
        return $true
    }

    if (@(".pyc", ".pyo", ".pyd", ".log", ".pt", ".pth", ".onnx", ".npy", ".npz") -contains $ext) {
        return $true
    }

    return $false
}

function Copy-ReleaseItem {
    param(
        [string]$SourcePath,
        [string]$DestPath
    )

    $source = Get-Item -LiteralPath $SourcePath
    if (-not $source.PSIsContainer) {
        if (-not (Test-ReleaseExcluded $source)) {
            Copy-Item -LiteralPath $SourcePath -Destination $DestPath -Force
        } else {
            Write-Host "Skipped local artifact: $SourcePath"
        }
        return
    }

    New-Item -ItemType Directory -Path $DestPath -Force | Out-Null
    Get-ChildItem -LiteralPath $SourcePath -Force | ForEach-Object {
        if (Test-ReleaseExcluded $_) {
            Write-Host "Skipped local artifact: $($_.FullName)"
            return
        }

        $childDest = Join-Path $DestPath $_.Name
        Copy-ReleaseItem -SourcePath $_.FullName -DestPath $childDest
    }
}

function Copy-ReleaseTree {
    param([string]$ReleaseCodeDir)

    New-Item -ItemType Directory -Path $ReleaseCodeDir -Force | Out-Null

    foreach ($item in $ItemsToCopy) {
        $sourcePath = Join-Path $CodeDir $item
        $destPath = Join-Path $ReleaseCodeDir $item

        if (-not (Test-Path $sourcePath)) {
            Write-Host "Warning: item not found: $item"
            continue
        }

        Copy-ReleaseItem -SourcePath $sourcePath -DestPath $destPath
        Write-Host "Copied: $item"
    }
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Host "Created output directory: $OutputDir"
}

if ($UseZip) {
    $StagingDir = Join-Path $OutputDir "staging"
    if (Test-Path $StagingDir) {
        Remove-Item -Path $StagingDir -Recurse -Force
    }

    Copy-ReleaseTree -ReleaseCodeDir (Join-Path $StagingDir "code")

    $ArchivePath = "$ReleasePath.zip"
    try {
        Compress-Archive -Path (Join-Path $StagingDir "*") -DestinationPath $ArchivePath -Force
    } finally {
        if (Test-Path $StagingDir) {
            Remove-Item -Path $StagingDir -Recurse -Force
        }
    }

    $LatestPath = Join-Path $OutputDir "latest.zip"
    if (Test-Path $LatestPath) {
        Remove-Item $LatestPath -Force
    }
    Copy-Item -Path $ArchivePath -Destination $LatestPath

    Write-Host ""
    Write-Host "Release package created successfully"
    Write-Host "  Archive: $ArchivePath"
    Write-Host "  Latest:  $LatestPath"
    Write-Host "  Size:    $([math]::Round((Get-Item $ArchivePath).Length / 1MB, 2)) MB"
} else {
    $ReleaseCodeDir = Join-Path $ReleasePath "code"
    Copy-ReleaseTree -ReleaseCodeDir $ReleaseCodeDir

    $TotalSize = (Get-ChildItem -Path $ReleasePath -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB

    $LatestPath = Join-Path $OutputDir "latest"
    if (Test-Path $LatestPath) {
        Remove-Item -Path $LatestPath -Recurse -Force
    }
    Copy-Item -Path $ReleasePath -Destination $LatestPath -Recurse -Force

    Write-Host ""
    Write-Host "Release directory created successfully"
    Write-Host "  Directory: $ReleasePath"
    Write-Host "  Latest:    $LatestPath"
    Write-Host "  Size:      $([math]::Round($TotalSize, 2)) MB"
}
