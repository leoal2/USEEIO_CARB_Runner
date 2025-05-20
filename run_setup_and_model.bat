@echo off
echo === Activating Conda environment ===
call conda activate buildings

if errorlevel 1 (
    echo Failed to activate 'buildings' environment. Make sure Conda is installed and environment is created.
    pause
    exit /b 1
)

echo.
echo === Installing CARB-modified FLOWSA package ===
pip install git+https://github.com/leoal2/flowsa_CARB_version.git

echo.
echo === (Optional) Setting R environment variables ===
setx R_HOME "%ProgramFiles%\R\R-4.4.2"
setx R_USER "%UserProfile%\Documents"
setx R_LIBS_USER "%LocalAppData%\Programs\R\R-4.4.2\library"

echo.
echo === Reminder: Install R packages manually ===
echo Please open R and run:
echo   install.packages("devtools", type = "win.binary")
echo   devtools::install_github("USEPA/useeior")
echo   devtools::install_github("USEPA/stateior")
echo.
pause

echo.
echo === Running the model ===
python run_model.py

echo.
echo === Done! ===
pause
