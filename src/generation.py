import os
from dotenv import load_dotenv
from groq import Groq
from config import LLM_MODEL_NAME

def build_prompt(question, results, chat_history=[]):
    # Format context with Source and Page numbers
    context_parts = []
    for doc in results:
        source = doc.metadata.get('source', 'unknown')
        filename = source.split('\\')[-1].split('/')[-1] # cross-platform basename
        page = doc.metadata.get('page')
        page_str = f" (Page {int(page) + 1})" if page is not None else ""
        citation = f"{filename}{page_str}"
        context_parts.append(f"Source: [{citation}]\n{doc.page_content}")
        
    context = "\n\n---\n\n".join(context_parts)
    
    # Format chat history (keep last 4 messages to save tokens)
    history_str = ""
    if chat_history:
        for msg in chat_history[-4:]:
            role = "User" if msg.role == "user" else "Assistant"
            history_str += f"{role}: {msg.content}\n"
            
    prompt = f"""You are a helpful AI assistant answering questions about uploaded documents.
Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't know based on the provided documents."
When you provide a fact from the context, YOU MUST cite the source at the end of the sentence using the provided source names like this: [filename.pdf (Page X)].

Context:
{context}

---
Chat History:
{history_str}
---

Current Question: {question}
Answer:"""
    return prompt

def ask_groq(prompt):
     load_dotenv()
     client = Groq(api_key=os.getenv("GROQ_API_KEY"))
     response = client.chat.completions.create(
         model=LLM_MODEL_NAME,
         messages=[{"role":"user","content":prompt}]
         ) 
     return response.choices[0].message.content
