@echo off
REM Generate evaluation visualizations from training results

echo ========================================
echo Generating Evaluation Visualizations
echo ========================================
echo.

REM Default: Use Initial_Test results
if "%1"=="" (
    set PROJECT=Initial_Test
    set GROUP=simplenet_test
    set RUN=bottle_initial
    echo Using default: Initial_Test results
) else (
    set PROJECT=%1
    set GROUP=%2
    set RUN=%3
    echo Using: %PROJECT%\%GROUP%\%RUN%
)

echo.
python generate_evaluations.py --results_path results --project %PROJECT% --group %GROUP% --run %RUN%

echo.
echo ========================================
echo Evaluation complete!
echo Check the evaluations folder in your results directory
echo ========================================
pause

