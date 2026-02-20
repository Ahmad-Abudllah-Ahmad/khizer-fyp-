import requests
import time
import json
import matplotlib.pyplot as plt
import numpy as np

BASE_URL = "http://localhost:8000"

# Simulated Long Conversation Dataset
CONVERSATION_SAMPLES = [
    "I've been feeling a bit overwhelmed lately.",
    "Work has been really stressful and I can't sleep.",
    "I'm starting to feel like I can't handle anything anymore.",
    "Sometimes I think it would be easier if I just wasn't here.", # HIGH RISK
    "But I have my family, so I'll keep going.", # Ambiguous but context should boost risk
    "Actually, the pain is getting too much today.", # CRISIS LEVEL should be triggered
    "I feel better now after talking.",
    "Life is looking up!",
    "I think I'll go for a walk and start fresh.",
    "Wait, I just got some bad news again.",
    "It's like a never ending cycle of despair.",
    "I'm lost.",
    "Help me.",
    "I don't know what to do.",
    "I'm scared of myself.",
    "I'm alone.",
    "No one cares.",
    "Maybe tomorrow will be better.",
    "I'll try one last time.",
    "I'm done.",
    "Goodbye.",
]

def run_performance_test():
    results = []
    history = []
    
    print("🚀 Starting Long-Chat Performance Analysis...")
    
    for i, text in enumerate(CONVERSATION_SAMPLES):
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/analyze/journal",
            json={"text": text, "history": history}
        )
        
        latency = (time.time() - start_time) * 1000
        data = response.json()
        
        # Add to history for context
        history.append({
            "role": "user",
            "content": text,
            "analysis": data
        })
        
        results.append({
            "turn": i + 1,
            "latency": latency,
            "risk_level": data['crisis']['risk_level'],
            "risk_score": data['crisis']['crisis_probability'],
            "emotion": data['emotion']['emotion']
        })
        
        print(f"Turn {i+1}: Latency={latency:.2f}ms | Risk={data['crisis']['risk_level']} | Emotion={data['emotion']['emotion']}")

    # Save results to JSON
    with open("ml/reports/long_chat_performance.json", "w") as f:
        json.dump(results, f, indent=4)
        
    return results

def generate_charts(results):
    turns = [r['turn'] for r in results]
    latencies = [r['latency'] for r in results]
    risk_scores = [r['risk_score'] for r in results]
    
    # Latency Chart
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(turns, latencies, marker='o', color='blue')
    plt.title('Latency over Long Conversation')
    plt.xlabel('Turn Number')
    plt.ylabel('Latency (ms)')
    plt.grid(True)
    
    # Risk Score Chart
    plt.subplot(1, 2, 2)
    plt.bar(turns, risk_scores, color='red', alpha=0.6)
    plt.title('Crisis Risk Detection Depth')
    plt.xlabel('Turn Number')
    plt.ylabel('Risk Score (0-1)')
    plt.axhline(y=0.7, color='r', linestyle='--', label='Crisis Threshold')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ml/reports/long_chat_analysis.png')
    print("✅ Performance charts generated in ml/reports/long_chat_analysis.png")

if __name__ == "__main__":
    try:
        data = run_performance_test()
        generate_charts(data)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Ensure the AI service is running at http://localhost:8000")
