# Starts NEXUS for local development: API on :8100, web app on :3000. Ctrl+C stops both.
$root = $PSScriptRoot
$api = Start-Process "$root\.venv\Scripts\python.exe" -ArgumentList "-m","uvicorn","nexus.main:app","--port","8100" -WorkingDirectory $root -NoNewWindow -PassThru
try {
    Set-Location "$root\frontend"
    npm run dev
} finally {
    Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
    Set-Location $root
}
