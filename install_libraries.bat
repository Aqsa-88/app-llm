@echo off
echo Activating virtual environment...
call llm_env\Scripts\activate

echo Installing required packages...
pip install -r requirements.txt

echo Done! Press any key to exit.
pause
