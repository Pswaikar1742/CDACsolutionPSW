@echo off
setlocal enabledelayedexpansion

:: Colors for output
set RED=[91m
set GREEN=[92m
set YELLOW=[93m
set NC=[0m

echo %GREEN%Setting up EEG Attention Classification Solution...%NC%

:: Check Python version
python -c "import sys; assert sys.version_info >= (3,8), 'Python 3.8+ required'" 2>nul
if errorlevel 1 (
    echo %RED%Error: Python version must be >= 3.8%NC%
    python -V
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo %YELLOW%Creating virtual environment...%NC%
    python -m venv venv
)

:: Activate virtual environment
echo %YELLOW%Activating virtual environment...%NC%
call venv\Scripts\activate.bat

:: Install requirements
echo %YELLOW%Installing requirements...%NC%
pip install -r requirements.txt

:: Install package in development mode
echo %YELLOW%Installing package in development mode...%NC%
pip install -e .

:: Run tests
echo %YELLOW%Running tests...%NC%
python -m pytest test_solution.py -v

:: Check if output directory exists
if not exist "output" (
    echo %YELLOW%Creating output directory...%NC%
    mkdir output
)

:: Run the solution
echo %YELLOW%Running the solution...%NC%
python main.py --validate --save_model --output_dir output

echo %GREEN%Setup and execution completed!%NC%
echo Results can be found in the 'output' directory
echo.
echo To deactivate the virtual environment, run: deactivate

endlocal
