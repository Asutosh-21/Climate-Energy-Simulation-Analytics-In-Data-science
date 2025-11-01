<#
start-app.ps1

Simple helper to start the Streamlit app on Windows (PowerShell).
If a `.venv` exists in the repo root, the script will use its Python interpreter.
Otherwise it falls back to the system `python` and runs `python -m streamlit run app.py`.

Usage:
  .\start-app.ps1

#>

Write-Host "Starting Streamlit app..."

if (Test-Path ".\.venv\Scripts\python.exe") {
    Write-Host "Found .venv — using .\.venv\Scripts\python.exe"
    & .\.venv\Scripts\python.exe -m streamlit run .\app.py
} else {
    Write-Host "No .venv found — using system python (ensure streamlit is installed)"
    python -m streamlit run .\app.py
}
