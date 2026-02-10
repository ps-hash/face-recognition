$ErrorActionPreference = "Stop"
try {
    $WshShell = New-Object -comObject WScript.Shell
    $DesktopPath = [Environment]::GetFolderPath("Desktop")
    $ShortcutPath = "$DesktopPath\FaceAttendance.lnk"
    $Shortcut = $WshShell.CreateShortcut($ShortcutPath)
    
    # Try to find pythonw, else python
    $PythonPath = ""
    try {
        $PythonPath = (Get-Command pythonw).Source
    } catch {
        $PythonPath = (Get-Command python).Source
    }

    if (-not $PythonPath) {
        Write-Error "Python executable not found."
    }

    $Shortcut.TargetPath = $PythonPath
    $Shortcut.Arguments = '"c:\Users\prash\OneDrive\Desktop\face attendence\face recognition\main.py"'
    $Shortcut.WorkingDirectory = "c:\Users\prash\OneDrive\Desktop\face attendence\face recognition"
    $Shortcut.Save()
    Write-Host "Shortcut created at $ShortcutPath"
} catch {
    Write-Error "Failed to create shortcut: $_"
    exit 1
}
