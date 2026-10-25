# Runs the rag-local-eval-loop suite against this project. Extra args go to eval.runner, e.g.
#   .\eval.ps1 --num-answerable 3 --num-unanswerable 3 --workers 1
$root = $PSScriptRoot
$env:RAG_PROJECT_ROOT = $root
$env:PYTHONPATH = $root
Get-Content "$root\.env" | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_]+)\s*=\s*(.+?)\s*$' -and $_ -notmatch '^\s*#') { Set-Item "env:$($Matches[1])" $Matches[2] }
}
Set-Location "$root\eval-loop"
python -m eval.runner --rag-root $root @args
