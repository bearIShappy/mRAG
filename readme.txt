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






c  