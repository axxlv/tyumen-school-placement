param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$SchoolPython = 'D:\Рабочий стол\ITMO AI\python-package-boilerplate\.venv-blocksnet\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $SchoolPython)) { throw 'Не найден Python окружения BlocksNet. Укажите путь в start.ps1.' }
Set-Location -LiteralPath $PSScriptRoot
& $SchoolPython -X utf8 server.py --port $Port
