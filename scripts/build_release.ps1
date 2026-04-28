# GorgeChase Release Packaging Script
# This script packages the necessary files and directories into a release archive

param(
    [string]$OutputDir = "P:\Repos\GorgeChase\release",
    [string]$ArchiveName = "gorgechase-release",
    [switch]$UseZip = $false
)

# Define source paths
$CodeDir = "P:\Repos\GorgeChase\code"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ReleaseName = "$ArchiveName-$Timestamp"
$ReleasePath = Join-Path $OutputDir $ReleaseName

# Create output directory if it doesn't exist
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Host "✓ Created output directory: $OutputDir"
}

# List of items to copy
$ItemsToCopy = @(
    "conf",
    "agent_ppo",
    ".vscode",
    "kaiwu.json",
    "train_test.py"
)

if ($UseZip) {
    # ZIP MODE: Create temporary staging directory
    $StagingDir = Join-Path $OutputDir "staging"
    if (Test-Path $StagingDir) {
        Remove-Item -Path $StagingDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $StagingDir | Out-Null

    # Create code directory structure in staging
    $StagingCodeDir = Join-Path $StagingDir "code"
    New-Item -ItemType Directory -Path $StagingCodeDir -Force | Out-Null

    # Copy items
    foreach ($item in $ItemsToCopy) {
        $sourcePath = Join-Path $CodeDir $item
        $destPath = Join-Path $StagingCodeDir $item
        
        if (Test-Path $sourcePath) {
            if ((Get-Item $sourcePath) -is [System.IO.DirectoryInfo]) {
                Copy-Item -Path $sourcePath -Destination $destPath -Recurse -Force
                Write-Host "✓ Copied directory: $item"
            } else {
                Copy-Item -Path $sourcePath -Destination $destPath -Force
                Write-Host "✓ Copied file: $item"
            }
        } else {
            Write-Host "⚠ Warning: Item not found - $item"
        }
    }

    # Create archive
    $ArchivePath = "$ReleasePath.zip"

    try {
        Compress-Archive -Path $StagingDir -DestinationPath $ArchivePath -Force
        Write-Host ""
        Write-Host "✅ Release package created successfully!"
        Write-Host "   Archive: $ArchivePath"
        Write-Host "   Size: $((Get-Item $ArchivePath).Length / 1MB -as [int]) MB"
    } catch {
        Write-Host "❌ Error creating archive: $_"
        exit 1
    }

    # Clean up staging directory
    Remove-Item -Path $StagingDir -Recurse -Force
    Write-Host "✓ Cleaned up temporary files"

    # Create latest.zip reference
    $LatestPath = Join-Path $OutputDir "latest.zip"
    if (Test-Path $LatestPath) {
        Remove-Item $LatestPath -Force
    }
    Copy-Item -Path $ArchivePath -Destination $LatestPath
    Write-Host "✓ Created latest.zip reference"

    Write-Host ""
    Write-Host "📦 Release Info:"
    Write-Host "   Name: $ReleaseName"
    Write-Host "   Location: $ArchivePath"
    Write-Host "   Latest: $LatestPath"

} else {
    # DIRECTORY MODE: Create release directory directly
    $ReleaseCodeDir = Join-Path $ReleasePath "code"
    New-Item -ItemType Directory -Path $ReleaseCodeDir -Force | Out-Null

    # Copy items
    foreach ($item in $ItemsToCopy) {
        $sourcePath = Join-Path $CodeDir $item
        $destPath = Join-Path $ReleaseCodeDir $item
        
        if (Test-Path $sourcePath) {
            if ((Get-Item $sourcePath) -is [System.IO.DirectoryInfo]) {
                Copy-Item -Path $sourcePath -Destination $destPath -Recurse -Force
                Write-Host "✓ Copied directory: $item"
            } else {
                Copy-Item -Path $sourcePath -Destination $destPath -Force
                Write-Host "✓ Copied file: $item"
            }
        } else {
            Write-Host "⚠ Warning: Item not found - $item"
        }
    }

    # Calculate directory size
    $TotalSize = (Get-ChildItem -Path $ReleasePath -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB

    Write-Host ""
    Write-Host "✅ Release directory created successfully!"
    Write-Host "   Directory: $ReleasePath"
    Write-Host "   Size: $([math]::Round($TotalSize, 2)) MB"

    # Create latest reference (copy instead of symlink for Windows compatibility)
    $LatestPath = Join-Path $OutputDir "latest"
    if (Test-Path $LatestPath) {
        Remove-Item -Path $LatestPath -Recurse -Force
    }
    Copy-Item -Path $ReleasePath -Destination $LatestPath -Recurse -Force
    Write-Host "✓ Created latest directory reference"

    Write-Host ""
    Write-Host "📦 Release Info:"
    Write-Host "   Name: $ReleaseName"
    Write-Host "   Location: $ReleasePath"
    Write-Host "   Latest: $LatestPath"
}

Write-Host ""
