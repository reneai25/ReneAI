import google.generativeai as genai

def answer_question(query, context, gemini_api_key):
    """Answers a question based on the provided context using Gemini."""
    prompt = f"Answer the following question based on the context provided:\n\nQuestion: {query}\n\nContext: {context}"
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = model.start_chat()
    response = chat.send_message(prompt)
    return response.text.strip()