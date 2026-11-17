param([string]$Python = '', [int]$Port = 8767)
$ErrorActionPreference = 'Stop'
$BundleRoot = $PSScriptRoot
$SchoolApp = Join-Path $BundleRoot 'school-placement'
$SchoolData = Join-Path $BundleRoot 'python-package-boilerplate-updated\examples\data_tyumen\refined'
if (-not (Test-Path -LiteralPath (Join-Path $SchoolApp 'server.py'))) { throw 'Рядом с этим файлом должна быть папка school-placement.' }
if (-not (Test-Path -LiteralPath (Join-Path $SchoolData 'accessibility_matrix.parquet'))) { throw 'Нет матрицы. Распакуйте полный комплект данных, сохраняя структуру папок.' }
if (-not $Python) {
    $Candidates = @(
        (Join-Path $BundleRoot '.venv-blocksnet\Scripts\python.exe'),
        (Join-Path $BundleRoot 'python-package-boilerplate\.venv-blocksnet\Scripts\python.exe'),
        'D:\Рабочий стол\ITMO AI\python-package-boilerplate\.venv-blocksnet\Scripts\python.exe'
    )
    foreach ($Candidate in $Candidates) { if (Test-Path -LiteralPath $Candidate) { $Python = $Candidate; break } }
}
if (-not $Python -or -not (Test-Path -LiteralPath $Python)) { throw 'Укажите Python 3.12 из окружения BlocksNet: .\OPEN_PROJECT.ps1 -Python "C:\путь\Scripts\python.exe"' }
$env:SCHOOL_DATA = $SchoolData
$env:SCHOOL_STATE = Join-Path $SchoolApp 'state'
Set-Location -LiteralPath $SchoolApp
Write-Host "Приложение: http://127.0.0.1:$Port/"
Write-Host "Прошлая практика: http://127.0.0.1:$Port/practice2"
Write-Host 'Полный поиск НЕ запускается. Сохранённые результаты читаются из базы.'
& $Python -X utf8 server.py --port $Port
