@echo off
REM Reset an IPA platform user password (admin use)
REM Usage: reset_user_password.bat EMPLOYEE_ID NEW_PASSWORD
if "%~2"=="" (
  echo Usage: reset_user_password.bat EMPLOYEE_ID NEW_PASSWORD
  echo Example: reset_user_password.bat 80100173 MyNewPass123
  exit /b 1
)
cd /d "%~dp0..\backend"
py scripts\reset_password.py %1 %2
pause
