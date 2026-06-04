# run_all.ps1 - Pipeline completo (seq + OpenMP + CUDA) en WINDOWS nativo.
#
# Equivalente de scripts/run_all.sh para maquinas Windows con GPU NVIDIA,
# Visual Studio (compilador cl.exe) y CUDA Toolkit (nvcc). Compila las tres
# versiones, genera datasets, corre el benchmark y produce el mismo CSV
# (results/benchmark.csv) que el flujo de Linux/Mac.
#
# USO (recomendado):
#   1. Abrir "x64 Native Tools Command Prompt for VS 2022" (menu inicio), o
#      cualquier PowerShell: el script intenta cargar Visual Studio solo.
#   2. cd a la carpeta del proyecto (la que tiene este script en scripts\).
#   3. Ejecutar:
#         powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
#
# Variables opcionales (antes de llamar, p.ej. $env:REPS=1):
#   REPS=5  MAX_ITER=100  K_LIST="3 5 10 200"  THREADS_LIST=auto

$ErrorActionPreference = "Stop"

# Ir a la raiz del proyecto (un nivel arriba de scripts\)
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# --- Parametros (con defaults, sobreescribibles por variables de entorno) ---
$REPS     = if ($env:REPS)     { [int]$env:REPS }     else { 5 }
$MAX_ITER = if ($env:MAX_ITER) { [int]$env:MAX_ITER } else { 100 }
$K_LIST   = if ($env:K_LIST)   { $env:K_LIST -split '\s+' } else { @(3,5,10,200) }
$CSV      = if ($env:CSV)      { $env:CSV } else { "results\benchmark.csv" }

Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host " K-means: pipeline completo (seq + OpenMP + CUDA) en Windows"        -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

# --- 0. Asegurar el entorno de Visual Studio (cl.exe en PATH) ----------------
function Ensure-MSVC {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        Write-Host "    cl.exe ya disponible en PATH."
        return
    }
    Write-Host "    cl.exe no esta en PATH; buscando Visual Studio (vswhere)..."
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "No se encontro vswhere. Instala Visual Studio con 'Desktop development with C++', " +
              "o abre 'x64 Native Tools Command Prompt for VS' y reintenta."
    }
    $vsPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vsPath) { throw "Visual Studio con herramientas C++ no encontrado." }
    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) { throw "No se encontro vcvars64.bat en $vsPath." }

    Write-Host "    Cargando entorno x64 desde: $vcvars"
    # Ejecuta vcvars64.bat y vuelca el entorno resultante a esta sesion.
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            Set-Item -Path "env:$($matches[1])" -Value $matches[2]
        }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "Tras cargar vcvars, cl.exe sigue sin estar disponible."
    }
}

function Resolve-Nvcc {
    # Devuelve @{Nvcc; Inc; Lib; Static; DllDir} o $null. Intenta, en orden:
    #  1) nvcc del CUDA Toolkit, si estuviera en PATH (instalacion normal);
    #  2) un nvcc ya instalado por pip --user (sin admin);
    #  3) instalarlo por pip --user (sin admin) y volver a buscar.
    $inPath = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($inPath) {
        # Toolkit completo: nvcc localiza sus headers/libs solo.
        return @{ Nvcc = $inPath.Source; Inc = $null; Lib = $null; Static = $true; DllDir = $null }
    }

    $py = $null
    foreach ($p in @("python","py","python3")) {
        if (Get-Command $p -ErrorAction SilentlyContinue) { $py = $p; break }
    }
    if (-not $py) { return $null }
    $site = (& $py -c "import site; print(site.getusersitepackages())" 2>$null)
    if ($site) { $site = $site.Trim() }

    function Find-In-Site($s) {
        if (-not $s -or -not (Test-Path $s)) { return $null }
        $n = Get-ChildItem $s -Recurse -Filter nvcc.exe        -ErrorAction SilentlyContinue | Select-Object -First 1
        $h = Get-ChildItem $s -Recurse -Filter cuda_runtime.h  -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not ($n -and $h)) { return $null }
        $st = Get-ChildItem $s -Recurse -Filter cudart_static.lib -ErrorAction SilentlyContinue | Select-Object -First 1
        $dy = Get-ChildItem $s -Recurse -Filter cudart.lib        -ErrorAction SilentlyContinue | Select-Object -First 1
        $dll = Get-ChildItem $s -Recurse -Filter cudart64_*.dll   -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($st) {
            return @{ Nvcc=$n.FullName; Inc=$h.Directory.FullName; Lib=$st.Directory.FullName; Static=$true; DllDir=$null }
        } elseif ($dy) {
            $dd = if ($dll) { $dll.Directory.FullName } else { $null }
            return @{ Nvcc=$n.FullName; Inc=$h.Directory.FullName; Lib=$dy.Directory.FullName; Static=$false; DllDir=$dd }
        }
        return $null
    }

    $found = Find-In-Site $site
    if ($found) { return $found }

    Write-Host "    nvcc no instalado: descargandolo por pip (--user, sin admin)..."
    Write-Host "    (puede tardar 1-2 min; descarga ~150 MB en tu carpeta de usuario)"
    # pip escribe avisos por stderr; con ErrorActionPreference=Stop eso
    # abortaria el script. Volcamos TODO a un log y revisamos exit code.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $py -m pip install --user --no-warn-script-location `
        nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 *> pip_nvcc.log
    $rc = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP
    if ($rc -ne 0) {
        Write-Host "    pip fallo (ver pip_nvcc.log). Se continuara sin CUDA." -ForegroundColor Yellow
        return $null
    }
    return (Find-In-Site $site)
}

Write-Host "`n>>> [1/4] Preparando compiladores..."
Ensure-MSVC
$cuda = Resolve-Nvcc
$haveNvcc = [bool]$cuda
if ($haveNvcc) {
    Write-Host "    nvcc: $($cuda.Nvcc)"
} else {
    Write-Host "    AVISO: nvcc no disponible; se compilara solo seq + OMP." -ForegroundColor Yellow
}

# --- 1. Compilar -------------------------------------------------------------
# /D_CRT_SECURE_NO_WARNINGS silencia los avisos de fopen/fscanf en MSVC.
# Compila con un helper que captura stdout+stderr y revisa SOLO el codigo de
# salida: asi un aviso por stderr no aborta el script (ErrorActionPreference).
$defs = "/D_CRT_SECURE_NO_WARNINGS"
Write-Host "`n>>> [2/4] Compilando..."

function Invoke-Compile([string]$name, [string]$exe, [string[]]$cmdArgs) {
    Write-Host "    $name..."
    # Bajar EAP durante la llamada externa: cl/nvcc escriben avisos por
    # stderr y con EAP=Stop eso abortaria el script aunque compile bien.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $log = & $cmdArgs[0] $cmdArgs[1..($cmdArgs.Count-1)] 2>&1
    $rc = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP
    if ($rc -ne 0) {
        $log | Write-Host
        throw "Fallo compilando $exe (codigo $rc)"
    }
}

Invoke-Compile "seq" "kmeans_seq" @("cl","/nologo","/O2",$defs,"/Fe:kmeans_seq.exe","src\kmeans_seq.c")
Invoke-Compile "omp" "kmeans_omp" @("cl","/nologo","/O2","/openmp",$defs,"/Fe:kmeans_omp.exe","src\kmeans_omp.c")
if ($haveNvcc) {
    # Si nvcc viene del Toolkit (Inc/Lib nulos) localiza todo solo; si viene
    # del paquete pip, hay que pasarle -I (headers) y -L (libs) explicitos.
    $nv = @($cuda.Nvcc, "-O2", "-o", "kmeans_cuda.exe", "src\kmeans_cuda.cu")
    if ($cuda.Inc) { $nv += @("-I", $cuda.Inc) }
    if ($cuda.Lib) { $nv += @("-L", $cuda.Lib) }
    $nv += @("-Xcompiler", $defs)
    Invoke-Compile "cuda" "kmeans_cuda" $nv
    # Enlace dinamico: la DLL del runtime debe estar junto al .exe en runtime.
    if (-not $cuda.Static -and $cuda.DllDir) {
        Get-ChildItem $cuda.DllDir -Filter "cudart64_*.dll" -ErrorAction SilentlyContinue |
            ForEach-Object { Copy-Item $_.FullName . -Force }
    }
}
# Limpia .obj que deja cl
Remove-Item *.obj -ErrorAction SilentlyContinue

# --- 2. Datasets -------------------------------------------------------------
Write-Host "`n>>> [3/4] Datasets..."
New-Item -ItemType Directory -Force -Path data, results | Out-Null

$python = $null
foreach ($p in @("python", "py", "python3")) {
    if (Get-Command $p -ErrorAction SilentlyContinue) { $python = $p; break }
}
if ($python) {
    $synth = @(
        @(1000,   2,  3,  "data/small_2d.txt"),
        @(10000,  10, 5,  "data/medium_10d.txt"),
        @(100000, 50, 10, "data/large_50d.txt")
    )
    foreach ($s in $synth) {
        if (Test-Path $s[3]) { Write-Host "    ya existe $($s[3])" }
        else {
            try { & $python scripts\gen_dataset.py $s[0] $s[1] $s[2] $s[3] }
            catch { Write-Host "    no se pudo generar $($s[3]) (¿numpy?)" -ForegroundColor Yellow }
        }
    }
} else {
    Write-Host "    Python no encontrado: se omiten los sinteticos." -ForegroundColor Yellow
    Write-Host "    (El benchmark usara solo los datasets presentes en data\.)"
}
if (-not (Test-Path "data/punto0_kmeans.txt")) {
    Write-Host "    AVISO: falta data/punto0_kmeans.txt (dataset real)." -ForegroundColor Yellow
}

# --- 3. Numero de nucleos y lista de hebras ----------------------------------
$ncpu = [int]$env:NUMBER_OF_PROCESSORS
if ($env:THREADS_LIST) {
    $threads = $env:THREADS_LIST -split '\s+' | ForEach-Object { [int]$_ }
} else {
    $threads = @(); $t = 1
    while ($t -le $ncpu) { $threads += $t; $t *= 2 }
    if ($threads -notcontains $ncpu) { $threads += $ncpu }
    $threads += [Math]::Min($ncpu * 2, $ncpu + 8)
    $threads = $threads | Sort-Object -Unique
}
Write-Host "    nucleos=$ncpu  THREADS_LIST=[$($threads -join ' ')]"

# --- 4. Benchmark ------------------------------------------------------------
Write-Host "`n>>> [4/4] Benchmark (REPS=$REPS MAX_ITER=$MAX_ITER K=[$($K_LIST -join ' ')])..."

$datasets = @(
    "data/small_2d.txt",
    "data/medium_10d.txt",
    "data/large_50d.txt",
    "data/punto0_kmeans.txt"
) | Where-Object { Test-Path $_ }

"version,dataset,N,D,K,threads,rep,iters,tiempo_s" | Out-File -Encoding ascii $CSV

function Parse-Run([string]$out) {
    $h = @{}
    if ($out -match 'N=(\d+)')       { $h.N = $matches[1] }
    if ($out -match 'D=(\d+)')       { $h.D = $matches[1] }
    if ($out -match 'iters=(\d+)')   { $h.iters = $matches[1] }
    if ($out -match 'tiempo=([\d.]+)') { $h.t = $matches[1] }
    return $h
}

# Durante el benchmark, no abortar por avisos de stderr de los binarios
# (p.ej. el runtime de CUDA). Solo nos importa el texto que imprimen.
$ErrorActionPreference = "Continue"

foreach ($ds in $datasets) {
    Write-Host "=== Dataset: $ds ===" -ForegroundColor Green
    foreach ($k in $K_LIST) {
        Write-Host "-- K=$k --"
        # Secuencial (linea base, threads=1)
        for ($r = 1; $r -le $REPS; $r++) {
            $o = & ".\kmeans_seq.exe" $ds $k $MAX_ITER | Out-String
            $p = Parse-Run $o
            "seq,$ds,$($p.N),$($p.D),$k,1,$r,$($p.iters),$($p.t)" | Add-Content $CSV
        }
        # OpenMP variando hebras
        foreach ($th in $threads) {
            $env:OMP_NUM_THREADS = $th
            for ($r = 1; $r -le $REPS; $r++) {
                $o = & ".\kmeans_omp.exe" $ds $k $MAX_ITER | Out-String
                $p = Parse-Run $o
                "omp,$ds,$($p.N),$($p.D),$k,$th,$r,$($p.iters),$($p.t)" | Add-Content $CSV
            }
        }
        # CUDA
        if ($haveNvcc -and (Test-Path ".\kmeans_cuda.exe")) {
            for ($r = 1; $r -le $REPS; $r++) {
                $o = & ".\kmeans_cuda.exe" $ds $k $MAX_ITER | Out-String
                $p = Parse-Run $o
                "cuda,$ds,$($p.N),$($p.D),$k,0,$r,$($p.iters),$($p.t)" | Add-Content $CSV
            }
        }
    }
}
Write-Host "`nResultados crudos en $CSV"

# --- Analisis: tablas (CSV + Markdown) y figuras -----------------------------
if ($python) {
    Write-Host "`n>>> Analisis (tablas + figuras)..."
    try { & $python scripts\analyze.py $CSV }
    catch { Write-Host "    analyze.py no corrio (¿falta matplotlib?). El CSV quedo en $CSV." -ForegroundColor Yellow }
}

Write-Host "`n===================================================================" -ForegroundColor Cyan
Write-Host " Listo." -ForegroundColor Cyan
Write-Host "   Tiempos: $CSV" -ForegroundColor Cyan
Write-Host "   Tablas:  results\tables\summary.csv y summary.md" -ForegroundColor Cyan
Write-Host "   Figuras: results\figures\*.png" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
