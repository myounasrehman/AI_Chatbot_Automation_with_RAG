import requests
import time
import logging
import json
from sentence_transformers import SentenceTransformer, util

# API endpoint for the chatbot and configure thresholds for similarity and retrieval
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
SIMILARITY_THRESHOLD = 0.3  # Adjust for response validation
RETRIEVAL_THRESHOLD = 0.4  # Adjust for knowledge base retrieval
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialization
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def load_knowledge_base(file_path):
    """
    Load and preprocess the knowledge base from a text file.
    """
    knowledge_base = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                knowledge_base.append(line.strip())
    return knowledge_base

def retrieve_context(query, knowledge_base):
    """
    Retrieve the most relevant context from the knowledge base using semantic similarity.
    """
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    kb_embeddings = embedding_model.encode(knowledge_base, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, kb_embeddings).squeeze(0)

    # Find the most relevant context
    max_score, best_index = scores.max().item(), scores.argmax().item()
    if max_score >= RETRIEVAL_THRESHOLD:
        return knowledge_base[best_index]
    return ""

def calculate_similarity(expected, actual):
    """
    Calculate semantic similarity between expected and actual responses.
    """
    expected_embedding = embedding_model.encode(expected, convert_to_tensor=True)
    actual_embedding = embedding_model.encode(actual, convert_to_tensor=True)
    similarity_score = util.cos_sim(expected_embedding, actual_embedding).item()
    return similarity_score

def send_query(query, context=""):
    """
    Sends a query to Ollama API.
    """
    start_time = time.time()
    try:
        payload = {
            "model": "llama2",
            "messages": [
                {"role": "system", "content": "Use the following context to answer the query: " + context},
                {"role": "user", "content": query}
            ]
        }
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        response_time = time.time() - start_time

        if response.status_code == 200:
            collected_message = ""
            for chunk in response.iter_lines():
                if chunk:
                    data = chunk.decode('utf-8')
                    try:
                        parsed_data = json.loads(data)
                        if "message" in parsed_data and "content" in parsed_data["message"]:
                            collected_message += parsed_data["message"]["content"]
                        if parsed_data.get("done", False):
                            break
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse chunk: {data}, Error: {e}")
            return collected_message.strip(), response_time
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        raise Exception(f"API call failed: {e}")

def log_result(test_case, passed, response_time, actual, similarity_score, expected):
    """
    Logs the test case results.
    """
    logging.info(f"Test: {test_case}")
    logging.info(f"Result: {'PASS' if passed else 'FAIL'}")
    logging.info(f"Response Time: {response_time:.2f} seconds")
    logging.info(f"Similarity Score: {similarity_score * 100:.2f}%")
    logging.info(f"Expected: {expected}")
    logging.info(f"Actual: {actual}\n")

# Define test cases
test_cases = [
    {"name": "Simple Greeting", "query": "Hello", "expected": "Hello there! How are you today?"},
    {"name": "Return Policy", "query": "What is the return policy?", "expected": "Items can be returned within 30 days."},
    {"name": "Track Order", "query": "Track order #12345.", "expected": "Please provide additional details to track your order."},
    {"name": "Complex Query", "query": "Summarize the history of the Roman Empire in 200 words.", "expected": "The Roman Empire was founded..."},
    {"name": "Edge Case: Empty Input", "query": "", "expected": "I didn't catch that. Could you please rephrase?"}
]

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(filename="results.log", level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Starting chatbot testing with RAG...")

    # Load knowledge base
    knowledge_base = load_knowledge_base("knowledge_base.txt")
    logging.info("Knowledge base loaded.")

    for test in test_cases:
        try:
            # Retrieve relevant context
            context = retrieve_context(test["query"], knowledge_base)

            # Send query to the chatbot
            response, response_time = send_query(test["query"], context)

            # Calculate semantic similarity
            similarity_score = calculate_similarity(test["expected"], response)
            passed = similarity_score >= SIMILARITY_THRESHOLD

            # Log the results
            log_result(test["name"], passed, response_time, response, similarity_score, test["expected"])
        except Exception as e:
            logging.error(f"Test Case: {test['name']} | Error: {e}")

    logging.info("Chatbot testing completed.")
