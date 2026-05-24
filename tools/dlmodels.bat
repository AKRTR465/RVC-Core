@echo off
chcp 65001 >nul
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT=%%~fI"
set "ARIA2=aria2c"

where %ARIA2% >nul 2>nul
if errorlevel 1 (
    if exist "%SCRIPT_DIR%aria2\aria2c.exe" (
        set "ARIA2=%SCRIPT_DIR%aria2\aria2c.exe"
    ) else (
        echo aria2c not found. Install aria2 or put aria2c.exe under tools\aria2.
        exit /b 1
    )
)

echo working dir is %ROOT%
echo dir check start.

if not exist "%ROOT%\assets\pretrained" mkdir "%ROOT%\assets\pretrained"
if not exist "%ROOT%\assets\pretrained_v2" mkdir "%ROOT%\assets\pretrained_v2"
if not exist "%ROOT%\assets\hubert" mkdir "%ROOT%\assets\hubert"
if not exist "%ROOT%\assets\rmvpe" mkdir "%ROOT%\assets\rmvpe"

echo dir check finished.
echo required files check start.

call :download "pretrained" "D32k.pth" "pretrained/D32k.pth" || exit /b 1
call :download "pretrained" "D40k.pth" "pretrained/D40k.pth" || exit /b 1
call :download "pretrained" "D48k.pth" "pretrained/D48k.pth" || exit /b 1
call :download "pretrained" "G32k.pth" "pretrained/G32k.pth" || exit /b 1
call :download "pretrained" "G40k.pth" "pretrained/G40k.pth" || exit /b 1
call :download "pretrained" "G48k.pth" "pretrained/G48k.pth" || exit /b 1
call :download "pretrained_v2" "f0D40k.pth" "pretrained_v2/f0D40k.pth" || exit /b 1
call :download "pretrained_v2" "f0G40k.pth" "pretrained_v2/f0G40k.pth" || exit /b 1
call :download "pretrained_v2" "D40k.pth" "pretrained_v2/D40k.pth" || exit /b 1
call :download "pretrained_v2" "G40k.pth" "pretrained_v2/G40k.pth" || exit /b 1
call :download "hubert" "hubert_base.pt" "hubert_base.pt" || exit /b 1
call :download "rmvpe" "rmvpe.pt" "rmvpe.pt" || exit /b 1

echo required files check finished.
exit /b 0

:download
set "SUBDIR=%~1"
set "NAME=%~2"
set "REMOTE=%~3"
set "TARGET=%ROOT%\assets\%SUBDIR%\%NAME%"
echo checking %NAME%
if exist "%TARGET%" (
    echo %NAME% in assets\%SUBDIR% checked.
    exit /b 0
)

echo downloading %NAME%
%ARIA2% --console-log-level=error -c -x 16 -s 16 -k 1M "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/%REMOTE%" -d "%ROOT%\assets\%SUBDIR%" -o "%NAME%"
if exist "%TARGET%" (
    echo download successful.
    exit /b 0
)

echo please try again.
exit /b 1
