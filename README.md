# AI Chatbot Automation with Retrieval-Augmented Generation (RAG)

This repository provides a Python-based solution to automate the testing of an AI chatbot that leverages Retrieval-Augmented Generation (RAG) for context-aware responses. The testing includes functionality validation, response quality assessment, and performance measurement.

## Features

- **Automated Testing**: Test the chatbot's functionality, response quality, and performance.
- **RAG Integration**: Enhance the chatbot's answers with relevant knowledge from a custom knowledge base.
- **Semantic Similarity Evaluation**: Validate responses using sentence embeddings and cosine similarity.
- **Customizable Thresholds**: Configure similarity and retrieval thresholds to suit your needs.
- **Extensive Logging**: Detailed logs capture test results and performance metrics.

---

## Prerequisites

1. **Python 3.8+**
2. **Libraries**:
   - `requests`
   - `sentence-transformers`
   - `torch`
   - `logging`

3. **Ollama Installation**: Install and configure [Ollama](https://ollama.ai) for Llama2 or your chosen model.
   - Ollama must be running locally at `http://127.0.0.1:11434/api/chat`.

4. **Knowledge Base**: A text file named `knowledge_base.txt` containing domain-specific data.

---

## Setup Instructions

1. **Clone Repository**
   ```bash
   git clone https://github.com/myounasrehman/AI_Chatbot_Automation_with_RAG.git
   cd AI_Chatbot_Automation_with_RAG

## 2. Install Dependencies

pip install -r requirements.txt

## 3. Run Ollama Server Ensure Ollama is installed and running on your machine. Set it up to use a model like llama2 or any compatible language model.

## 4.Prepare Knowledge Base Create a file named knowledge_base.txt in the repository root with your domain-specific information. Example content:

      Return policy: Items can be returned within 30 days.
      Track order: Visit our website or contact support with your order number.
      History of the Roman Empire: The Roman Empire was founded in 27 BC...

      ## 5. Run the Script
            python test_llm_chatbot.py
            
## Configuration

  ##  Thresholds:
        SIMILARITY_THRESHOLD: Minimum similarity for response validation (default: 0.3).
        RETRIEVAL_THRESHOLD: Minimum relevance for knowledge base retrieval (default: 0.4).

 ##   Knowledge Base File:
        Update or replace knowledge_base.txt with relevant domain informati
        
 ##  File Structure

test_llm_chatbot.py: Main Python script for testing.
knowledge_base.txt: Customizable knowledge base file.
requirements.txt: Python dependencies.
results.log: Generated log file capturing test outcomes.

 ##  EXAMPLE LOGS
2025-01-05 13:53:32,907 - Starting chatbot testing with RAG...
2025-01-05 13:53:32,907 - Knowledge base loaded.
2025-01-05 13:53:46,843 - Test: Simple Greeting
2025-01-05 13:53:46,843 - Result: PASS
2025-01-05 13:53:46,843 - Response Time: 11.86 seconds
2025-01-05 13:53:46,843 - Similarity Score: 96.94%
2025-01-05 13:53:46,843 - Expected: Hello there! How are you today?
2025-01-05 13:53:46,843 - Actual: Hi there! How are you today?
2025-01-05 13:36:45,559 - Actual: Hi there! How are you today?

## Customization

    Test Cases: Modify the test_cases list in test_llm_chatbot.py to suit your testing needs.
    Knowledge Base: Add more content to knowledge_base.txt for improved contextual responses.

