from indexing_retrieval import embedder, collection, tokenizer, model

def retrieve_relevant_chunks(query: str, k: int = 3):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results["documents"][0]

def generate_response(query: str):
    context_chunks = retrieve_relevant_chunks(query)
    context = "\n\n".join(context_chunks)

    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": "You are a helpful assistant using the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ], tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()

query = """
Were Alexa and Brittany different ?
"""
print(generate_response(query))