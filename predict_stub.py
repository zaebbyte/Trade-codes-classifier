import os
import sys
from datetime import datetime

# Add root project directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from ml.predict import predict_top_k

LOG_FILE = "logs/predictions_log.txt"

def get_prediction_response(input_text):
    """
    Simulates an API call that returns HS6 predictions with description.
    
    Args:
        input_text (str): Product description

    Returns:
        dict: {
            "query": ...,
            "predictions": [
                {"hs6": ..., "confidence": ..., "description": ...},
                ...
            ]
        }
    """
    top_k = predict_top_k(input_text)

    response = {
        "query": input_text,
        "predictions": [
            {
                "hs6": p["hs6"],
                "confidence": p["confidence"],
                "description": p["description"]
            }
            for p in top_k
        ]
    }

    log_entry = f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] QUERY: {input_text}\n"
    for pred in response["predictions"]:
        log_entry += f"→ {pred['hs6']} ({pred['confidence']*100:.2f}%) - {pred['description']}\n"

    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)
    return response

# Example usage
if __name__ == "__main__":
    sample = "toys for kids"
    result = get_prediction_response(sample)
    print(result)
