@echo off
setlocal

REM Prompt user for folder name suffix (e.g., mydata)
set /p INPUT_NAME=Enter folder name suffix (e.g., mydata): 

REM Prevent empty input
if "%INPUT_NAME%"=="" (
    echo No input provided. Exiting...
    pause
    exit /b
)

REM Create data_<INPUT_NAME> folder and its subdirectories in the current script directory
set "DATA_DIR=%~dp0data_%INPUT_NAME%"

mkdir "%DATA_DIR%"
mkdir "%DATA_DIR%\audio_mp3"
mkdir "%DATA_DIR%\audio_wav"
mkdir "%DATA_DIR%\result"
mkdir "%DATA_DIR%\transcribe"
mkdir "%DATA_DIR%\video"

echo Folder "%DATA_DIR%" and subfolders created successfully.

set "WAV_WORKERS_ENV="
if not "%ANITTS_WAV_CONVERT_WORKERS%"=="" (
    set "WAV_WORKERS_ENV=-e ANITTS_WAV_CONVERT_WORKERS=%ANITTS_WAV_CONVERT_WORKERS%"
    echo Passing ANITTS_WAV_CONVERT_WORKERS=%ANITTS_WAV_CONVERT_WORKERS%
)

REM Run docker container with bind mounts to the newly created data folder and the current folder's module/model
REM The container name now includes the user-provided suffix appended with a dash
docker run -it --rm -p 7860:7860 %WAV_WORKERS_ENV% --gpus all --name anitts-container-%INPUT_NAME% -v "%~dp0data_%INPUT_NAME%:/workspace/AniTTS-Builder-v3/data" -v "%~dp0module/model:/workspace/AniTTS-Builder-v3/module/model" anitts-builder-v3

pause
