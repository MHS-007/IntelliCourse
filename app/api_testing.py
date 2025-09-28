import requests

URL = "http://127.0.0.1:8000/chat"

# Multiple queries for testing
queries = [
    "What are the prerequisites for Software Engineering?",
    "What is the job market like for data scientists?",
    "What are the prerequisites for Introduction To Programming?",
    "Who is the instructor for the course Abnormal Psychology?"
]

for q in queries:
    try:
        payload = {"query": q}
        response = requests.post(URL, json=payload, timeout=30)

        # If API returns error code
        if response.status_code != 200:
            print(f"\n❌ Error for query: {q}")
            print(f"Status: {response.status_code}")
            print("Response:", response.text)
        else:
            data = response.json()
            print(f"\n✅ Query: {q}")
            print("Answer:", data.get("answer", "N/A"))
            print("Source Tool:", data.get("source_tool", "N/A"))

    except Exception as e:
        print(f"\n⚠️ Exception for query '{q}': {e}")
