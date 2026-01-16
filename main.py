import os

print("Sangai weather")
#COMMAND TO RUN THE JOBLIB FILES 
#uv run weatherfinalfinal.py
os.system("uv run uvicorn SERVER.main:app --host 0.0.0.0 --port 8000")
