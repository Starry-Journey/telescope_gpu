# Headless ZOS-API probe (Windows PowerShell / .NET Framework).
#
# Usage (headless server attempt):
#   powershell -ExecutionPolicy Bypass -File scripts\zos_probe.ps1 `
#     -InstallDir "D:\Program Files\Ansys Zemax OpticStudio 2023 R1.00" `
#     -Zmx "C:\temp\model.zmx" -Log "C:\temp\zos_probe.log"
#
# Observed on this machine (2026-11):
#   - CreateNewApplication() starts an application (0 systems), but
#     LoadNewSystem(ZMX) hangs in the non-interactive session.
#   - CreateZemaxServer("OpticStudio"/install dir) returns
#     IsValidLicenseForAPI=False, InitializationErrors=Win32Exception,
#     "The system cannot find the file specified".
# Do not fake S. The script exits non-zero when the API is unavailable.
param(
  [Parameter(Mandatory=$true)][string]$InstallDir,
  [Parameter(Mandatory=$true)][string]$Zmx,
  [string]$Log = "$env:TEMP\zos_probe.log",
  [string]$ExeName = "OpticStudio"
)

$ErrorActionPreference = "Stop"
function Write-Log($m) { $m | Tee-Object -FilePath $Log -Append }

try {
  Write-Log "probe start $(Get-Date -Format o)"
  Write-Log "install=$InstallDir"
  Write-Log "zmx=$Zmx"
  [System.Reflection.Assembly]::LoadFrom((Join-Path $InstallDir "ZOSAPI_Interfaces.dll")) | Out-Null
  [System.Reflection.Assembly]::LoadFrom((Join-Path $InstallDir "ZOSAPI.dll")) | Out-Null
  $conn = New-Object ZOSAPI.ZOSAPI_Connection
  Write-Log "CreateZemaxServer($ExeName)"
  $app = $conn.CreateZemaxServer($ExeName)
  if (-not $app) { Write-Log "CreateZemaxServer returned null"; exit 2 }
  Write-Log "mode=$($app.Mode) license=$($app.LicenseStatus) apiValid=$($app.IsValidLicenseForAPI)"
  Write-Log "initErrors=$($app.InitializationErrors)"
  if (-not $app.IsValidLicenseForAPI) { Write-Log "API license unavailable"; exit 2 }
  $sys = $app.LoadNewSystem($Zmx)
  if (-not $sys) { Write-Log "LoadNewSystem returned null"; exit 3 }
  $sd = $sys.SystemData
  Write-Log "EPD=$($sd.Aperture.EPD)"
  $ld = $sd.LensData
  Write-Log "surfaces=$($ld.NumberOfSurfaces)"
  for ($i = 0; $i -lt $ld.NumberOfSurfaces; $i++) {
    $s = $ld.GetSurfaceAt($i)
    Write-Log ("surface {0}: type={1} radius={2} conic={3} thickness={4}" -f $i,$s.Type,$s.Radius,$s.Conic,$s.Thickness)
  }
  Write-Log "probe ok"
  $app.CloseApplication()
  exit 0
} catch {
  Write-Log ("ERROR: " + $_.Exception.ToString())
  exit 1
}
