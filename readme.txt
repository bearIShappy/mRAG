MRAG/
│
├── documents/                # raw input files (PDF, DOCX)
│   ├── drylab.pdf
│   ├── index.pdf
│
├── output/                   # ALL generated data
│   ├── parsed/
│   │   ├── drylab_parsed.json
│   │   ├── GMNDC_parsed.json
│   │
│   ├── chunks/
│   │   ├── chunks.json
│   │
│   ├── images/
│       ├── figure-1-1.jpg
│       ├── figure-2-2.jpg
│
├── src/
│   ├── backend/
│   │   ├── core/
│   │   │   ├── chunk.py
│   │   │   ├── embedder.py
│   │   │   ├── vector_store.py
│   │   │   ├── rag_qa.py
│   │   │
│   │   ├── doc_parser.py
│   │   ├── utils/
│   │   ├── config/
│
├── frontend/
├── requirements.txt





doc_parser.py  →  chunk.py  →  embedder.py  →  vector_store.py  →  rag_qa.py
   (parse)        (chunk)       (embed)          (store)            (ask)  
 [unstructured]   [regex +        ↓             [Qdrant]         [LLM - gemma4]
                  spatial]        ↓ 
                                  ↓ 
                              split by chunk type:
                    
                              text-only  →  mxbai  →  collection: "text_chunks"   (1024-dim)
                              img+text   →  CLIP   →  collection: "image_chunks"  (512-dim)       

start your backend:
python -m src.backend.core.main


db:
D:\Jasleen space\mRAG>mkdir qdrant_storage
D:\Jasleen space\mRAG>docker run -p 6333:6333 -p 6334:6334 -v "%cd%\qdrant_storage:/qdrant/storage" qdrant/qdrant
#Host_Port : Container_Port (Port   Type    Purpose
                        6333:6333   HTTP  This is for the REST API. When you use your browser or a Python script to talk to Qdrant, you’ll point it to http://localhost:6333.
                        6334:6334   gRPC  This is for high-performance data transfer. Many Qdrant client libraries (like the Python SDK) use this port by default because it's faster than standard HTTP; 
                        Think of the Container as an office building. The Container Port is the internal extension number, and the Host Port is the external phone number you dial to get through the front desk.)
#qdrant/qdrant : image name 
#-v flag stands for Volume. By default, files inside a Docker container are deleted when the container stops. A volume "links" a folder  on your computer to a folder inside the container so your data is saved permanently.
#"Take requests from my local ports 6333 and 6334 and send them inside the container."
  "Take anything the container saves in container path(/qdrant/storage) and put it in my Windows folder host path (%cd%\qdrant_storage).

