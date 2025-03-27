@echo off
python -m venv new
call new\Scripts\activate
pip install tensorflow==2.12.0 numpy flask pillow
deactivate
exit
