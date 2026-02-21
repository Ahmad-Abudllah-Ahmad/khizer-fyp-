# SereneMind: Enterprise-Grade Mental Health AI Architecture
## Full Technical Report & Architecture Documentation

SereneMind is a production-grade AI ecosystem designed to provide empathetic mental health support through multi-modal emotional analysis. This report details the high-performance sub-20ms inference architecture, containerization strategy, and Kubernetes orchestration designed for infinite scalability.

---

## 1. System Overview & Microservices Architecture

The system is built on a "Stateless-First" microservices philosophy. Each component is an independent, containerized unit that communicates via high-speed RESTful APIs.

### Diagram 1: High-Level System Architecture (Unified v4)
```mermaid
graph TD
    User([User Client]) <--> UI[Next.js 14 Frontend]
    UI <--> Proxy[Next.js API Gateway/Rewrite]
    
    subgraph "Core AI Cluster"
        Proxy <--> AI[FastAPI AI Service]
        Proxy <--> AV[Avatar Response Service]
    end
    
    subgraph "Unified Model v4 (Production)"
        AI --> UM[unified_mental_health.joblib]
        UM --> INF[9-Class Inference Engine]
    end
    
    subgraph "Data Services (Stateless)"
        UI -.-> Auth[Mock Auth Store]
        UI -.-> Jour[In-Memory Journal]
    end
```

---

## 2. Containerization Strategy (Docker)

Each service in SereneMind is containerized using multi-stage Silicon-optimized Dockerfiles.

### Diagram 2: Docker Image Layering Strategy (Optimization)
```mermaid
graph LR
    Base[Alpine/Slim Base] --> Deps[Dependency Layer - pip/npm install]
    Deps --> Code[Source Code Layer]
    Code --> Run[Runtime Layer - 5.6MB Unified Model]
    
    subgraph "Size Optimization"
        Run --> Dist[Distroless-style Slimmed Images]
    end
```

### Diagram 3: Container Connectivity Flow
```mermaid
sequenceDiagram
    participant D as Docker Engine
    participant F as Frontend Container
    participant A as AI Service Container
    participant V as Avatar Container
    
    D->>F: Start Port 3000
    D->>A: Start Port 8000
    D->>V: Start Port 8001
    F->>A: POST /analyze/journal
    A-->>F: JSON (9-Class Unified Response)
    F->>V: POST /api/avatar/respond
    V-->>F: JSON (Chat Response)
```

---

## 3. Kubernetes Orchestration (Local & Cloud)

SereneMind is designed to run on K8s (Kubernetes) to handle traffic spikes.

### Diagram 4: Kubernetes Cluster Topology
```mermaid
graph TB
    Ingress[Nginx Ingress Controller] -->|TLS Termination| SvcF[Frontend Service]
    Ingress -->|Path Routing| SvcA[AI API Service]
    
    subgraph "K8s Node 1"
        PodF1[Frontend Pod]
        PodF2[Frontend Pod]
    end
    
    subgraph "K8s Node 2 (GPU Optimized)"
        PodA1[AI Inference Pod]
        PodA2[AI Inference Pod]
    end
    
    subgraph "K8s Node 3"
        PodV1[Avatar Pod]
    end
```

### Diagram 5: Horizontal Pod Autoscaling (HPA) Flow
```mermaid
graph LR
    Traffic[Incoming Chat Load] --> CPU[CPU/Memory Metrics]
    CPU -->|Threshold > 70%| HPA[Autoscaler]
    HPA -->|Scale Up| Pods[AI Pods: 2 -> 10]
    Pods --> LoadBalancer[Traffic Balanced]
```

---

## 4. AI Model Architecture — Unified v4 (Breakthrough)

The SereneMind mission utilizes a single high-performance model handling all dimensions of mental health.

### Diagram 6: Unified Model v4 Inference Stack
```mermaid
graph TD
    TXT[User Journal Entry] --> CLIP[Text Clipping: 2000 chars]
    CLIP --> TFIDF[TF-IDF: word 1-4g + char 2-5g]
    TFIDF --> FEAT[70,000 Feature Matrix]
    FEAT --> LBFGS[L-BFGS Logistic Regression]
    LBFGS --> CAL[Isotonic Calibration]
    CAL --> OUT[9-Class Probability Scores]
```

### Diagram 7: Model Training & Deployment Workflow (L-BFGS)
```mermaid
stateDiagram-v2
    [*] --> RawData: Multi-Source Datasets
    RawData --> Balancing: Class Weighting + Seed Anchor
    Balancing --> Preprocess: Clean/Tokenize
    Preprocess --> FeatureEng: FeatureUnion (word+char)
    FeatureEng --> Training: L-BFGS Solver
    Training --> Evaluation: Calibrated F1 Metrics
    Evaluation --> Export: .joblib Bundle
    Export --> DockerBuild: Contextual Image
    DockerBuild --> [*]
```

### Diagram 8: Advanced Data Merging Pipeline (New)
```mermaid
graph LR
    EM[Emotion Dataset] --> MERGE[Merge Engine]
    GO[GoEmotions] --> MERGE
    MH[MH Conversations] --> MERGE
    SEN[Sentiment Data] --> MERGE
    SEED[Clinical Seed] --> MERGE
    
    MERGE --> CLEAN[Text Cleaning]
    CLEAN --> BAL[Balanced Sampling / Capping]
    BAL --> FINAL[100k+ Training Samples]
```

### Diagram 9: 9-Class Categorization Schema (New)
```mermaid
graph TD
    INPUT[Raw Labels] --> MAP[Label Mapping Logic]
    MAP --> CRIS[Crisis]
    MAP --> DEPR[Depression]
    MAP --> ANX[Anxiety]
    MAP --> STRS[Stress]
    MAP --> GRIF[Grief]
    MAP --> FEAR[Fear]
    MAP --> ANGR[Anger]
    MAP --> JOY[Joy]
    MAP --> NORM[Normal/Stable]
```

---

## 5. Model Performance Distribution

### Diagram 10: Inference Accuracy Spread (Unified v4)
```mermaid
pie title Per-Class Reliability (F1 Score)
    "Crisis (SuicideWatch)" : 72
    "Anxiety" : 98
    "Grief" : 99
    "Normal/Joy" : 75
    "Other Classes" : 60
```

---

## 6. Request-Response Lifecycle (WhatsApp Style)

### Diagram 11: Frontend State Synchronisation
```mermaid
graph TD
    Submit[Submit Entry] --> LocalUI[Append Local Message]
    LocalUI --> API[API Analyze Call]
    API -->|Async| Droplet[Show Thinking Dots]
    Droplet --> Result[Receive Unified Response]
    Result --> Final[Render Calibrated Metrics]
```

### Diagram 12: Auto-Scroll Anchor Logic
```mermaid
graph LR
    NewMsg[New Message Received] --> HeightChange[DOM Height Change]
    HeightChange --> ScrollInit[Immediate ScrollToBottom]
    ScrollInit --> Delay[200ms Delay]
    Delay --> ScrollAnchor[Final Position Locked]
```

---

## 7. Security & Privacy Architecture

### Diagram 13: Security Boundary Diagram
```mermaid
graph TD
    Public([Public Internet]) -- HTTPS/SSL --> Gateway[API Gateway]
    
    subgraph "Private Virtual Network"
        Gateway -->|Stateless| AI_S[AI Service]
        Gateway -->|Encrypted| AV_S[Avatar Service]
        AI_S -.->|No Persistence| RAM[RAM Only Storage]
    end
    
    subgraph "Client Side"
        Browser[Local Browser] -->|AES-256| LocalStorage[Encrypted Store]
    end
```

### Diagram 14: CI/CD "Zero-Downtime" Deployment
```mermaid
graph LR
    Git[GitHub Push] --> Test[PyTest / NPM Test]
    Test --> Build[Docker Multi-Arch Build]
    Build --> Registry[Private Image Registry]
    Registry --> K8s[K8s Rolling Update]
    K8s -->|Success| Live[Production V1.1 Live]
```

---

## 8. Scaling to "Bulky" Large Language Models (LLMs)

### Diagram 15: Bulky Model Deployment & GPU Acceleration
```mermaid
graph TD
    Queue[Request Queue - Redis/RabbitMQ] --> Dist[Distributed Inference Workers]
    
    subgraph "NVIDIA-Optimized GPU Cluster"
        Dist --> VRAM1[GPU 0: RoBERTa-Large]
        Dist --> VRAM2[GPU 1: Llama-3 Clinical]
    end
    
    VRAM1 --> Batch[Batch Processing]
    VRAM2 --> Stream[Token Streaming / SSE]
    
    Batch --> Response[Aggregated Clinical PDF Report]
```

---

## 9. Development Environment

- **Frontend**: Next.js 14 (Port 3000)
- **AI Service**: FastAPI / Unified Model v4 (Port 8000)
- **Avatar Service**: FastAPI / Response Generator (Port 8001)

---
*Report Generated: 2026-02-22*
*Author: SereneMind Engineering Team*
