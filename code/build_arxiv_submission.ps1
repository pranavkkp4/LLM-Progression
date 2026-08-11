[CmdletBinding()]
param()

$projectRoot = Split-Path -Parent $PSScriptRoot
$paperDirectory = Join-Path $projectRoot "paper"
$outputDirectory = Join-Path $projectRoot "arxiv submit"
$figureOutput = Join-Path $outputDirectory "figures"
$zipPath = Join-Path $outputDirectory "paper_arxiv.zip"

if (Test-Path -LiteralPath $outputDirectory) {
    throw "Refusing to overwrite existing submission folder: $outputDirectory"
}

$requiredFigures = @(
    "architecture.pdf",
    "results.png",
    "calibration.png",
    "robustness.pdf"
)
$requiredPaperFiles = @("main.tex", "references.bib", "main.bbl")

foreach ($name in $requiredPaperFiles) {
    $path = Join-Path $paperDirectory $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing required paper file: $path"
    }
}
foreach ($name in $requiredFigures) {
    $path = Join-Path (Join-Path $paperDirectory "figures") $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing required figure: $path"
    }
}

function Expand-LatexInputs {
    param([Parameter(Mandatory)][string]$Text)

    $inputPattern = [regex]'\\input\{([^}]+)\}'
    while ($true) {
        $match = $inputPattern.Match($Text)
        if (-not $match.Success) {
            return $Text
        }

        $relativePath = $match.Groups[1].Value
        if (-not [System.IO.Path]::HasExtension($relativePath)) {
            $relativePath = "$relativePath.tex"
        }
        $inputPath = [System.IO.Path]::GetFullPath(
            (Join-Path $paperDirectory $relativePath)
        )
        if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
            throw "Unable to expand LaTeX input: $inputPath"
        }

        $replacement = Get-Content -Raw -LiteralPath $inputPath
        $replacement = Expand-LatexInputs -Text $replacement
        $Text = $Text.Substring(0, $match.Index) +
            $replacement +
            $Text.Substring($match.Index + $match.Length)
    }
}

New-Item -ItemType Directory -Path $figureOutput | Out-Null

$mainSource = Get-Content -Raw -LiteralPath (Join-Path $paperDirectory "main.tex")
$flattenedSource = Expand-LatexInputs -Text $mainSource
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText(
    (Join-Path $outputDirectory "main.tex"),
    $flattenedSource,
    $utf8NoBom
)

Copy-Item -LiteralPath (Join-Path $paperDirectory "references.bib") -Destination $outputDirectory
Copy-Item -LiteralPath (Join-Path $paperDirectory "main.bbl") -Destination $outputDirectory
foreach ($name in $requiredFigures) {
    Copy-Item -LiteralPath (Join-Path (Join-Path $paperDirectory "figures") $name) -Destination $figureOutput
}

$archiveInputs = @(
    (Join-Path $outputDirectory "main.tex"),
    (Join-Path $outputDirectory "references.bib"),
    (Join-Path $outputDirectory "main.bbl"),
    $figureOutput
)
Compress-Archive -LiteralPath $archiveInputs -DestinationPath $zipPath

Write-Output $zipPath
