import json
from pathlib import Path

from src.backend.core.embedder     import Embedder
from src.backend.core.vector_store import VectorStore
from src.backend.core.rag_qa       import RAGQA

def main():
    # 1. Load your chunks (output of chunk.py)
    chunks_path = Path("output") / "chunks" / "chunks.json"
    
    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 2. Embed the chunks
    # The new Embedder automatically routes text to mxbai and multimodal to CLIP.
    print("\nInitializing Embedder and embedding chunks...")
    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    # 3. Store in Qdrant
    # VectorStore automatically handles the dual collections 
    # ("text_chunks" 1024-dim, and "image_chunks" 512-dim)
    print("\nInitializing VectorStore and upserting chunks...")
    store = VectorStore()
    store.upsert(embedded_chunks)
    
    # Show collection info
    print("\nCollection Stats:")
    store.info()

    # 4. Peek at stored points (Testing Dual-Collection Similarity Search)
    print("\nTesting similarity search for 'newspaper'...")
    # embed_query now returns a dict: {"text": [...], "multimodal": [...]}
    query_vectors = embedder.embed_query("newspaper")
    results = store.similarity_search(query_vectors, top_k=3)
    
    for r in results:
        source = r.get("source_collection", "unknown")
        print(f"[{source}] (Score: {r['score']:.4f}) {r['text'][:80]}...")

#     # 5. Ask questions using the full RAG pipeline
#     print("\nInitializing RAG QA and asking question...")
#     qa = RAGQA(store=store, embedder=embedder)
    
#     question = "What are the students doing in the photo from the 'Tomorrow's teachers' article, and who are they?"
#     # verbose=True will print out the chunks it retrieves before answering
#     result = qa.ask(question, top_k=5, verbose=True)

#     # Print the outputs
#     print("\n=== CONTEXT SENT TO LLM ===")
#     print(result["context"])
#     print("\n=== ANSWER ===")
#     print(result["answer"])

# if __name__ == "__main__":
#     main()
    ########## OR RUN INTERACTIVE MODE ########## 
    # 5. Ask questions using the full RAG pipeline
    print("\nInitializing RAG QA...")
    qa = RAGQA(store=store, embedder=embedder)
    
    print("\n" + "="*50)
    print("RAG System Ready! Type 'exit' or 'quit' to stop.")
    print("="*50)

    # Interactive loop
    while True:
        # Prompt the user for a question in the terminal
        question = input("\nYour Question: ").strip()
        
        # Check if the user wants to exit
        if question.lower() in ['exit', 'quit']:
            print("Exiting RAG system. Goodbye!")
            break
            
        # Skip empty inputs
        if not question:
            continue

        print("\nSearching and Generating Answer...")
        
        # verbose=True will print out the chunks it retrieves before answering
        result = qa.ask(question, top_k=5, verbose=True)

        # Print the outputs
        print("\n=== CONTEXT SENT TO LLM ===")
        print(result["context"])
        print("\n=== ANSWER ===")
        print(result["answer"])
        print("-" * 50) # Just a separator for readability between questions
if __name__ == "__main__":
    main()
"""
 # question = "According to the document, who are the top grade 12 pupils associated with the Capitec Foundation?" y
    # question = "Who are the Capitec Foundation students holding certificates in the photo?" y
    # question = "Who is the CEO of GMMDC?"y
    # question = "What did the newspapers say about GMMDC in 2019?" y
Based on the document you provided and the multimodal architecture we just built, here are some excellent sample questions you can ask your RAG system to test its capabilities. 

I have categorized them to test specific parts of your pipeline (Text, Image Logic, and Fusion).

### Category 1: Testing the "Equal Focus" (Text + Visuals)
These questions require the LLM to combine facts written in the main text with the visual details described in the captions or spatial bounding boxes.

* **"In the photo showing the top Grade 12 pupils in Port Elizabeth's Incubator School Programme, who is the student standing on the far right?"** y
    * [cite_start]*Why it tests the system:* The LLM must find the caption [cite: 60, 61][cite_start], realize the image shows students and adults, and determine that Stephanie Cronjè from Linkside High is the student on the far right[cite: 60].
* **"What are the students doing in the photo from the 'Tomorrow's teachers' article, and who are they?"** y
    * [cite_start]*Why it tests the system:* It must locate the image caption describing the Nelson Mandela University final-year students (Monique Paulse, Danielle du Plessis, and Anita Rossouw)[cite: 144, 145].

### Category 2: Testing the "Missing Captions" (Spatial Inference)
These questions test your `is_above`, `is_left_of` bounding box logic. The images might not have a formal "Caption," so the LLM must use the surrounding text you appended.

* **"There is a cartoon graphic or drawing of a student at a desk. What specific digital app or technology is being described in the text associated with that image?"** no
    * [cite_start]*Why it tests the system:* The first image is an illustration[cite: 1], not a photograph. [cite_start]The LLM must link this image to the surrounding text discussing the "Touch Tutor" app[cite: 13, 17].
* **"What is the name of the competition where students create artworks inspired by nature and manmade designs, as seen in the photos of the winning students?"** n - model hallucinated
    * [cite_start]*Why it tests the system:* It requires linking the photos of the art competition winners [cite: 211] [cite_start]to the text discussing the "Math-Art Competition"[cite: 128, 212].

### Category 3: Testing the Standard Text Retrieval (Deep Facts)
These questions test the `mxbai` text model's ability to pull highly specific facts and statistics from the dense paragraphs.

* **"How much did Zimbini Ntsula's maths and science marks improve, and what school does she attend?"** y
    * *Why it tests the system:* It tests specific entity extraction. [cite_start]The LLM should find that she attends Ndlovukazi Senior Secondary School in Komani and her marks improved by 22%[cite: 67].
* **"According to Prof Werner Olivier, what percentage of students from their programmes go on to university?"** y
    * *Why it tests the system:* It tests statistical retrieval. [cite_start]The answer is around 50%[cite: 42].
* **"What are the two categories that students could choose from in the Math-Art Competition?"** y
    * *Why it tests the system:* It tests the extraction of specific rules. [cite_start]The categories are "mathematics in manmade designs" and "nature"[cite: 130].

    """

"""
To verify that your updated chunk.py is correctly capturing, truncating, and passing the spatial text to your Embedder without losing critical entities, you need to ask questions about specific images where the answers are buried in the paragraphs immediately surrounding them.

Based on the parsed PDF text you shared, here are two targeted queries you can run in your terminal, along with the expected answers and what they test:

Test Query 1: The Port Elizabeth Pupils (Page 2)
The Query to run:

"Based on the photos in the article, who are the top Grade 12 pupils in Port Elizabeth's Incubator School Programme standing in the front row, and who is standing behind them?"

Expected LLM Answer:

The front row includes first-placed Asisipho Ndinisa, her twin sister Siphoselhu, and second-placed Stephanie Cronjé. Standing behind them are Neptal Khoza, Luzuko Jama, and Ncedile Konzana.

Why this tests the changes:
In your parsed data, this image is spatially tied to and.

Previously, the Embedder might have truncated the end of this combined string, dropping the names of the men in the back row (Neptal, Luzuko, Ncedile).

With the new chunk.py, will be truncated to 100 characters for the `image_caption`, and the remaining details (including) will flow into the image_description array, ensuring CLIP sees the most critical keywords early in its context window.

Test Query 2: The Final Year Students (Page 3)
The Query to run:

"In the photo showing final-year students looking forward to boosting learners' maths and science skills, who are the three women pictured from left to right?"

Expected LLM Answer:

The students are Monique Paulse, Danielle du Plessis, and Anita Rossouw.

Why this tests the changes:
This text spans across three short spatial extractions (, , ``).

This specifically tests your image_description = image_description[:3] logic. It ensures that the chunker successfully grabbed these three adjacent fragments, treated them as "points" (since extract_points defaults to the whole string if no numbers are found), and successfully fused them into the multimodal chunk payload.

"""