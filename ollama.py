import pandas as pd
import requests
from datetime import datetime
import json
import time
import os

def get_summary_from_ollama(prompt_text):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": prompt_text},
            stream=True
        )

        output = ""
        for line in response.iter_lines():
            if line:
                try:
                    result = json.loads(line.decode("utf-8"))
                    output += result.get("response", "")
                except Exception as e:
                    output += f"\n[Error parsing line: {e}]"
        return output.strip()

    except Exception as e:
        return f"Error communicating with Ollama: {e}"

while True:
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen for cleaner output
    print(f"🔁 Running EDA at {datetime.now().strftime('%H:%M:%S')}...\n")

    try:
        df = pd.read_csv("vehicle_detections.csv")
    except Exception as e:
        print("❌ Failed to read CSV:", e)
        time.sleep(10)
        continue

    if df.empty:
        print("⚠️ CSV is empty. Waiting for new data...")
        time.sleep(10)
        continue

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])

    if df.empty:
        print("⚠️ No valid timestamps found.")
        time.sleep(10)
        continue

    df['Hour'] = df['Timestamp'].dt.hour

    total = len(df)
    vehicle_counts = df['Class'].value_counts().to_dict()

    if 'Speed (px/sec)' in df.columns:
        valid_speeds = df['Speed (px/sec)'].dropna()
        avg_speed = round(valid_speeds.mean(), 2) if not valid_speeds.empty else 0
    else:
        avg_speed = 0

    if 'NumberPlate' in df.columns:
        detected_plates = df[df['NumberPlate'].str.len() > 3]['NumberPlate'].nunique()
    else:
        detected_plates = 0

    peak_hour = df['Hour'].value_counts().idxmax() if not df['Hour'].empty else "N/A"

    prompt = f"""
You are a traffic data analyst. Analyze the following vehicle detection data summary and give a clear, professional explanation.

Data Summary:
- Total vehicles: {total}
- Vehicle types and counts: {vehicle_counts}
- Average speed: {avg_speed} px/sec
- Unique number plates detected: {detected_plates}
- Peak traffic hour: {peak_hour}

Your job:
1. Identify the most common vehicle type and percentage.
2. Evaluate the average speed (low/medium/high).
3. Comment on number plate recognition success.
4. Highlight peak hour traffic significance.
5. Write a concise plain English report summarizing all insights.
"""

    ollama_response = get_summary_from_ollama(prompt)
    print("\n📊 Ollama's EDA Report:\n")
    print(ollama_response)

    with open("ollama_traffic_report.txt", "w", encoding="utf-8") as file:
        file.write("📊 Ollama's EDA Report\n")
        file.write("=======================\n\n")
        file.write(ollama_response)
        print("\n✅ Report saved to: ollama_traffic_report.txt")

    print("\n⏳ Waiting 10 seconds before next update...\n")
    time.sleep(10)
