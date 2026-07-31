# Clone the upstream Kronos repo at a pinned commit into phase-a/Kronos.
#
# The clone is READ-ONLY (working rule 5): never edit it in place. Our changes live
# in this repo as overlays (e.g. phase-a/scripts/smoke_test.py). Because it is a
# separate upstream project, it is gitignored rather than vendored -- this script
# reproduces it exactly.

$ErrorActionPreference = "Stop"

$UpstreamUrl = "https://github.com/shiyu-coder/Kronos.git"
$UpstreamSha = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
$Target = Join-Path $PSScriptRoot "..\Kronos"

if (Test-Path $Target) {
    $current = (git -C $Target rev-parse HEAD).Trim()
    if ($current -eq $UpstreamSha) {
        Write-Output "Upstream Kronos already at pinned commit $UpstreamSha"
        exit 0
    }
    Write-Output "Upstream Kronos at $current, expected $UpstreamSha -- fetching."
} else {
    git clone $UpstreamUrl $Target
}

git -C $Target fetch --depth 50 origin $UpstreamSha
git -C $Target checkout --detach $UpstreamSha
Write-Output "Upstream Kronos pinned at $UpstreamSha"
