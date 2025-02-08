# Roadmap-LLMs-para-Pioneros-Researchers

# Prerrequisitos

- **Mathematical Foundations**
    - **Linear Algebra**: operaciones con matrices, valores propios/vectores propios, SVD.
    - **Calculus**: diferenciación, derivadas parciales, regla de la cadena para _backpropagation_.
    - **Probability & Statistics**: distribuciones, expectativa, varianza; enfoques bayesiano vs. frecuentista.
    - **Optimization**: gradient descent, métodos estocásticos.
- **Programming & ML Fundamentals**
    - Sólidos conocimientos de **Python**.
    - Familiaridad con algún framework de _deep learning_ (por ejemplo, **PyTorch**; ¡intenta evitar TensorFlow a toda costa! JAX no está mal, pero sigue siendo poco usado).
    - Comprensión de conceptos básicos de **ML**: división train/val/test, sobreajuste, ajuste de hiperparámetros.
- **Foundational Knowledge of Neural Networks**
    - Redes feed-forward clásicas, redes convolucionales, redes recurrentes.
    - _Backpropagation_ y diferenciación automática.
    - Familiaridad con workflows de ML y nociones básicas sobre GPU/aceleradores.
- **Basic HPC & Distributed Computing** (recomendado para trabajar con LLM a gran escala)
    - Conocimientos en entrenamiento multi-GPU y paralelismo de datos.
    - Buenas prácticas en containerización (Docker) y control de versiones (Git).

---

# Hoja de Ruta

## 1. Fundamentos y Contexto Histórico

### Temas Clave

El intento de modelar el lenguaje se ha abordado desde hace mucho tiempo. No es necesario reimplementar o memorizar todos estos modelos antiguos; basta con comprender su funcionamiento, ya que los modelos actuales de estado del arte han heredado mucho de ellos y son referenciados en investigaciones recientes.

- **Classical Language Modeling**: n-grams, _smoothing_ (Kneser–Ney, Good–Turing).
- **Early Neural Language Models**  
    _(Bengio et al., 2003)_: redes neuronales _feed-forward_ para la predicción de la siguiente palabra.  
    [Leer el paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- **Recurrent Architectures**: RNN, LSTM, GRU (todavía vigentes, bueno... casi).

Aquí es cuando se produce el primer cambio de paradigma:

- **Transformer Breakthrough**: “Attention Is All You Need” (Vaswani et al., 2017).
- **Milestone LLMs**: BERT, serie GPT.
- **Scaling Hypothesis**: relación entre el tamaño del modelo, el tamaño del dataset y el rendimiento.

### Recursos Recomendados

- **Papers**:
    - **A Neural Probabilistic Language Model (Bengio et al., 2003)**  
        Aquí se observa uno de los primeros intentos en modelar el lenguaje.  
        [https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
    - **Attention Is All You Need (Vaswani et al., 2017)**  
        El paper más importante de la IA contemporánea.  
        [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
    - **Language Models are Unsupervised Multitask Learners (GPT-2, OpenAI, 2019)**  
        Un modelo sumamente relevante que marca la pauta para los modelos de lenguaje a gran escala.  
        [https://openai.com/blog/better-language-models](https://openai.com/blog/better-language-models)
    - **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2019)**  
        Otro "gran modelo de lenguaje", pero no generativo autoregresivo, sino de tipo _encoder_.  
        [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
- **Blogs y Tutoriales**:
    - **The Illustrated Transformer (Jay Alammar)**  
        Un recurso muy citado y didáctico para entender el famoso Transformer.  
        [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- **Conferencias y Video Clases**:
    - **Stanford CS224N: Natural Language Processing with Deep Learning**  
        [Ver la playlist en YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4).  
        Si estás más acostumbrado a lecturas universitarias, este curso puede resultarte útil para ganar noción del NLP, aunque es probable que gran parte del conocimiento ya lo tengas.
    - **Yannic Kilcher’s YouTube Channel** (paper reviews y discusiones)  
        Una "mausquerramienta misteriosa" que usaremos ahora y más adelante. Cualquier modelo que te interese y si te da pereza leer el paper o no te queda claro, búscalo en este canal y te aclarará muchas cosas.  
        [https://www.youtube.com/c/yannickilcher](https://www.youtube.com/c/yannickilcher)

---

## 2. Profundizando en las Arquitecturas de Transformers

### Temas Clave

Tener un entendimiento a bajo nivel de todos los componentes de los LLM permitirá introducirte en técnicas modernas más fácilmente.

La parte de la anatomía del Transformer ya la habrás entendido en la sección anterior, pero es tan importante que deberías volver a darle un vistazo a otros recursos sobre el mismo tema para que quede sumamente claro:

- **Anatomía del Transformer**
    - _Self-attention_ multi-head (query, key, value), codificaciones posicionales.
    - Redes _feed-forward_, _layer normalization_, conexiones residuales.

Ahora empezamos a entrar en aquello que se construyó sobre el Transformer: manejar secuencias largas, variantes, etc.

- **Variantes y Extensiones**
    - Encoder-decoder vs. decoder-only vs. encoder-only.
    - Manejo de secuencias largas (_Sparse Attention_, Longformer, Big Bird, Reformer).
    - Enfoques de eficiencia en parámetros (Adapters, LoRA, Prompt Tuning).

Un aspecto clave que muchas personas omiten revisar, pero es esencial para entender ciertos comportamientos del modelo:

- **Tokenización y Vocabulario**
    - BPE (Byte-Pair Encoding), SentencePiece, WordPiece.
    - Compromisos en la granularidad de tokens y manejo de palabras fuera del vocabulario (OOV).

### Recursos Recomendados

- **Papers**:
    - **Attention Is All You Need (Vaswani et al., 2017)**  
        Efectivamente, léelo de nuevo.  
        [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
    - **Longformer: The Long-Document Transformer (Beltagy et al., 2020)**  
        Un paper que propone un método temprano para lidiar con documentos largos en Transformers.  
        [https://arxiv.org/abs/2004.05150](https://arxiv.org/abs/2004.05150)
- **Blogs y Tutoriales**:
    - **The Illustrated GPT-2 (and 1, and 3) (Jay Alammar)**  
        Otro artículo muy didáctico e ilustrativo para entender la generación autoregresiva.  
        [https://jalammar.github.io/illustrated-gpt2/](https://jalammar.github.io/illustrated-gpt2/)
    - **Hugging Face Course (Transformer usage)**  
        Hugging Face tiene un curso sobre Transformers. Aunque no suele cubrir a profundidad el tema, se sugiere pasar rápidamente a la parte de cómo usar su librería. A este punto probablemente estés ansioso por probar diferentes modelos; usar la librería Transformers de HF es la forma más fácil de comenzar.  
        [https://huggingface.co/learn/nlp-course/chapter2/1](https://huggingface.co/learn/nlp-course/chapter2/1)
- **Conferencias y Video Clases**:
    - **"Let's Build GPT" (Andrej Karpathy)**  
        Explicación en detalle del famoso GPT por Karpathy.  
        [https://youtu.be/kCc8FmEb1nY?si=sxSoN_OrWaVMFYeb](https://youtu.be/kCc8FmEb1nY?si=sxSoN_OrWaVMFYeb)
    - **"Let's Build the GPT Tokenizer" (Karpathy)**  
        Karpathy lo hace de nuevo, pero esta vez explicando los _tokenizers_.  
        [https://youtu.be/zduSFxRajkE?si=TTrBaVE3Q8qn780N](https://youtu.be/zduSFxRajkE?si=TTrBaVE3Q8qn780N)
    - **The Annotated Transformer (Harvard NLP – Video & Code Walkthrough)**  
        Otro gran recurso que disecciona el Transformer y lo implementa en código PyTorch, parte por parte.  
        [http://nlp.seas.harvard.edu/annotated-transformer/](http://nlp.seas.harvard.edu/annotated-transformer/)

---

## 3. Estrategias de Preentrenamiento y Objetivos

Ahora que ya tienes dominadas las partes de arquitectura y mecanismos, es un buen momento para entender la escala, los _datasets_, el preentrenamiento, los objetivos, las leyes de escala y técnicas más recientes.

### Temas Clave

- **Otros Paradigmas Auto-Supervisados**
    - Unificación del paradigma _decoder-encoder_ (T5, Bart) para multitarea.
    - Aprendizaje emergente en escenarios _few-shot_/_zero-shot_.
- **Curación de Datos y Pipeline**
    - Corpora a gran escala (Common Crawl, OpenWebText, C4).
    - Limpieza de datos, filtrado, deduplicación y consideraciones sobre representación y _bias_.
- **Scaling Laws y Entrenamiento Eficiente**
    - _Kaplan et al. (2020)_: tamaño del modelo, tamaño del dataset y compensaciones en _compute_.

### Recursos Recomendados

- **Papers**:
    - **Scaling Laws for Neural Language Models (Kaplan et al., 2020)**  
        El famoso paper de las leyes de escala, en el que se explica que al escalar el dataset, los parámetros y el _compute_, el modelo mejorará de forma predecible.  
        [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
    - **Language Models are Few-Shot Learners (GPT-3 paper)**  
        [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)  
        Aquí se presenta GPT-3, que representa un salto tremendo en el número de parámetros.
    - **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)**  
        En este paper se introduce el dataset C4. Luego se recomienda pasar a la nueva versión de T5: FLAN-T5.
    - **Scaling Instruction-Finetuned Language Models (FLAN-T5, Chung et al., 2022)**  
        Se muestran capacidades emergentes _zero-shot_ en un modelo no tan grande que incorpora _encoder_ y _decoder_.  
        [https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)
    - **Training Compute-Optimal Large Language Models**  
        Un paper sobre leyes de escala que define un "número óptimo de parámetros" para un determinado tamaño de dataset.
- **Blogs y Tutoriales**:
    - [Common Crawl](https://commoncrawl.org/) – Página oficial del proyecto para recolectar toda la data de Internet.
    - [RedPajama-Data-V2](https://www.together.ai/blog/redpajama-data-v2) – Un dataset más curado y relativamente reciente. Aún toca esperar para ver el estado del arte en _datasets_.
- **Conferencias y Video Clases**:
    - **Yannic Kilcher’s Explanations (GPT-3, T5, XLNet, etc.)**  
        [https://www.youtube.com/c/yannickilcher](https://www.youtube.com/c/yannickilcher)

---

## 4. Métodos de Fine-Tuning y Adaptación

Hasta ahora solo habíamos visto el preentrenamiento, pero ¿dónde están los chatbots? ¿Qué pasa luego de que entrenas de manera auto-supervisada en un corpus enorme? ¡Queremos conversar con estos modelos llenos de información a nuestros términos!

### Temas Clave

¿Cómo pasar de GPT-3 a ChatGPT?

- **Instruction Tuning y RLHF**
    - Unificar múltiples tareas bajo una interfaz basada en instrucciones.
    - _Reinforcement Learning from Human Feedback (RLHF)_ para la alineación.

Ahora, digamos que queremos tener un modelo bueno en una tarea específica nuestra:

- **Fine-Tuning Clásico**
    - Capas finales específicas para la tarea, con riesgo de sobreajuste y olvido catastrófico.
- **Fine-Tuning Eficiente en Parámetros**
    - Adapters (Houlsby et al.), LoRA (Hu et al.), _Prefix-Tuning_, _P-Tuning_.
    - Compensaciones en el uso de memoria, estabilidad en el entrenamiento y rendimiento.
- **Adaptación a Dominios y Aprendizaje Continuo**
    - Manejo de corpora específicos de dominio (médico, legal, científico).
    - Estrategias para mitigar el olvido catastrófico (EWC, métodos de _rehearsal_).

### Recursos Recomendados

- **Papers**:
    - **Training Language Models to Follow Instructions with Human Feedback**  
        [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)  
        El paper que explica cómo alinear el modelo (SFT + RLHF).
    - **LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)**  
        Un método ampliamente usado actualmente para _fine-tuning_ de modelos de lenguaje con parámetros reducidos.  
        [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
    - **Fine-Tuning Language Models from Human Preferences (Ziegler et al., 2019)**  
        [https://arxiv.org/abs/1909.08593](https://arxiv.org/abs/1909.08593)
- **Blogs y Tutoriales**:
    - [Blog de InstructGPT](https://openai.com/index/instruction-following/) – El mismo paper de InstructGPT, pero en formato de blog.
    - **Hugging Face Trainer Documentation**  
        [https://huggingface.co/docs/transformers/main/en/training](https://huggingface.co/docs/transformers/main/en/training)  
        Probablemente querrás entrenar modelos en tus datasets pequeños e intentar técnicas como el entrenamiento con pocos parámetros.

---

## 5. Evaluación y Benchmarking

Verás muchos números en los papers que usan los investigadores para presumir su _overfitting_, ejem, quize decir: el nivel de sus modelos, lol. Hay que echar un vistazo a lo que representan estos números.

### Temas Clave

Primero, algunas métricas clásicas que miden qué tanto el modelo pudo capturar el dataset (no simbolizan capacidad en alguna vertical en particular):

- **Métricas Clásicas**
    - _Perplexity_, BLEU, ROUGE, METEOR (con limitaciones para tareas generativas).

Ahora, si entramos en benchmarks que miden qué tan bueno es un LLM generando código, resolviendo problemas de razonamiento, preguntas de matemáticas, etc.:

- **Benchmarks Específicos para Tareas**
    - GLUE, SuperGLUE para clasificación/NLU.
    - MMLU, BIG-Bench para capacidades avanzadas de razonamiento.
    - Benchmarks comunes (GPQA, MATH, HumanEval, etc.).
    - Evaluaciones humanas: basadas en preferencias, corrección factual y coherencia.
- **Robustez y Pruebas de Estrés**
    - _Prompts_ adversariales, datos fuera de distribución.

### Recursos Recomendados

- **Papers**:
    - **Measuring Massive Multitask Language Understanding (Hendrycks et al.)**  
        [https://arxiv.org/abs/2009.03300](https://arxiv.org/abs/2009.03300)  
        Este es el paper que presenta el benchmark MMLU.
    - **Emergent Abilities of Large Language Models (Wei et al.)**  
        [https://arxiv.org/abs/2009.03300](https://arxiv.org/abs/2009.03300)  
        En este paper se exploran técnicas tempranas de _prompting_ y se brinda noción de los benchmarks actuales.
- **Repositorios para Benchmarking**:
    - **EleutherAI’s LM Evaluation Harness**  
        Un repositorio con código de evaluación para diferentes modelos en distintos benchmarks.  
        [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
    - **Hugging Face Evaluate**  
        Un repositorio con evaluaciones clásicas.  
        [https://huggingface.co/docs/evaluate/index](https://huggingface.co/docs/evaluate/index)
    - [Wikipedia – Lista de Benchmarks](https://en.wikipedia.org/wiki/List_of_language_model_benchmarks)  
        Explora la lista, lee las descripciones y revisa algunos ejemplos; la mayoría se encuentran en Hugging Face Datasets.

---

## 6. Infraestructura de Implementación y Entrenamiento

Si queremos entrenar modelos enormes, debemos conocer cómo se realizan estos cálculos en el hardware, cómo hacerlo más eficiente, escalarlo y paralelizarlo.

### Temas Clave

Conceptos básicos que debes conocer para implementar entrenamientos a gran escala:

- **Hardware y Paralelismo**
    - GPU vs. TPU vs. aceleradores especializados (Gaudi, Cerebras).
    - Entrenamiento con _data parallel_, _model parallel_, _pipeline parallel_.
    - Optimización de memoria (gradient checkpointing, _mixed precision_).
    - Entrenamiento con _batches_ grandes y acumulación de gradientes.

Si escribir tu propio framework de entrenamiento es imposible (a menos que seas un 10x engineer o tengas un equipo contigo), usar frameworks dedicados a esto es una gran alternativa.

- **Frameworks de Software**
    - PyTorch (DDP/FSDP), JAX (_pmap_).
    - **Accelerate** de Hugging Face para entrenamiento distribuido.
    - Torchtitan.

Para aprovechar al máximo las GPUs, es fundamental alimentarlas con un flujo rápido y eficiente de datos:

- **Gestión de Datasets**
    - _Streaming_ desde almacenamiento en la nube, datasets particionados, versionado de datasets (DVC).
    - Webdatasets.

### Recursos Recomendados

- **Papers/Guías**:
    - **DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale**  
        [https://arxiv.org/abs/2206.07682](https://arxiv.org/abs/2206.07682)
- **Herramientas y Documentación**:
    - **DeepSpeed Documentation**  
        [https://www.deepspeed.ai/](https://www.deepspeed.ai/)
    - **PyTorch Distributed Overview**  
        [https://pytorch.org/tutorials/intermediate/ddp_tutorial.html](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
    - **Hugging Face Accelerate**  
        [https://huggingface.co/docs/transformers/accelerate](https://huggingface.co/docs/transformers/accelerate)
    - [Torchtitan](https://github.com/pytorch/torchtitan)

---

## 7. Integrando Multimodalidad en LLMs

### Temas Clave

Es hora de revisar cómo lograr que los modelos de lenguaje puedan entender imágenes, audio u otra modalidad.

Los primeros intentos (incluyendo GPT-4) solían usar _embeddings_ de _encoders_ de visión.

- **Modelos Vision-Language**
    - Cómo se combina la información visual y textual: **Flamingo**, **# Qwen2-VL**.
    - Desafíos de alineación y procesamiento conjunto (encoders duales vs. _embeddings_ compartidos).

En enfoques más recientes se busca integrarlo en el mismo espacio de tokens.

- **Multimodalidad Nativa**
    - Integración desde la arquitectura base para procesar múltiples modalidades simultáneamente (**Chameleon**).
    - Uso de cabezales o capas especializadas para cada modalidad (texto, imagen, audio, etc.) (**Transfusion**).
    - Interacción en tiempo real con voz.

### Recursos Recomendados

- **Papers**:
    - **Flamingo: a Visual Language Model for Few-Shot Learning** (Alayrac et al., 2019)  
        El paper de Flamingo, uno de los primeros intentos de conseguir chats con un LLM para imágenes.  
        [https://arxiv.org/abs/2204.14198](https://arxiv.org/abs/2204.14198)
    - **Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution (Wang et al., 2024)**  
        Un paper de modelos actuales abiertos con mejor performance en tareas de visión.
    - **Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model (Zhou et al., 2024)**  
        El paper de Transfusion para integrar multimodalidad nativa.  
        [https://arxiv.org/abs/2408.11039](https://arxiv.org/abs/2408.11039)
    - **Chameleon: Mixed-Modal Early-Fusion Foundation Models (2024)**  
        Otra propuesta para integrar multimodalidad nativa, potencialmente la forma del GPT-4 o similar.  
        [https://arxiv.org/abs/2405.09818](https://arxiv.org/abs/2405.09818)
    - **Moshi: A Speech-Text Foundation Model for Real-Time Dialogue** (Defossez et al.)  
        De los pocos modelos _open source_ para audio a audio en tiempo real.
- **Herramientas y Documentación**:
    - **Demos en Hugging Face Spaces**

---

## 8. Modelos con Retrieval-Augmented y Agentes

### Temas Clave

Muchas áreas requieren que los modelos de lenguaje puedan acceder a información dinámica, incluso en tiempo real; se trata de brindarles esa información sin necesidad de reentrenarlos.

- **Retrieval-Augmented Generation (RAG)**
    - Arquitectura separada de _retriever_ + _generator_.
    - _Dense_ vs. _sparse retrieval_ (DPR, BM25).
    - Potencial para reducir alucinaciones al fundamentar las respuestas en textos externos.
- **Memoria y Edición del Conocimiento**
    - Bases de datos vectoriales (FAISS, ScaNN) o memorias _key-value_.
    - Actualización del conocimiento en tiempo real sin necesidad de reentrenamiento completo.
    - Conocimiento paramétrico vs. no paramétrico.
    - Memoria _neural_.

La capacidad de los modelos ha llegado a un punto en el que se los considera capaces de realizar trabajos por su cuenta, planear y ejecutar acciones; se ha abierto un ecosistema en torno a convertirlos en agentes.

- **Agentes basados en LLMs**
    - Frameworks para la orquestación de _chains_ (por ejemplo, **LangChain**).
    - Integración de herramientas externas (llamadas a API, bases de datos) para resolver tareas complejas.
    - _Action planning_: cómo el agente decide qué pasos dar y en qué orden.

### Recursos Recomendados

- **Papers**:
    - **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al.)**  
        [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
    - **Titans: Learning to Memorize at Test Time (Behrouz et al., 2024)**  
        Un enfoque muy reciente para contextos sumamente largos y uso de información dinámica.  
        [https://arxiv.org/abs/2501.00663](https://arxiv.org/abs/2501.00663)
- **Blogs y Tutoriales**:
    - Documentación de **RAG** en Hugging Face:  
        [https://huggingface.co/learn/cookbook/advanced_rag](https://huggingface.co/learn/cookbook/advanced_rag)
    - Documentación y ejemplos de **LangChain** (para la creación de agentes que combinan RAG con acciones externas; aunque es un framework popular, añade mucha complejidad innecesaria, así que no descartes alternativas).  
        [https://python.langchain.com/docs/tutorials/](https://python.langchain.com/docs/tutorials/)

---

## 9. Optimizaciones al Transformer

Te habrás dado cuenta de las grandes limitaciones que tiene la arquitectura; es increíble, ¿no? Sí, pero con ciertos trucos se puede mejorar.

### Temas Clave

Se pueden entrenar modelos para que no cuesten tantos FLOPs en inferencia:

- **Modificaciones a la Arquitectura**
    - _MoE_ (Mixture of Experts).

Podemos optimizar a bajo nivel la ejecución de las operaciones para que sean más rápidas y usen menos memoria:

- **Optimizaciones en los Kernels**
    - _Flash Attention 2_.

También podemos optar por modelos muy pequeños pero buenos en vertientes específicas:

- **Modelos Específicos**
    - LLMs especializados de dominio (por ejemplo, BioGPT para biología).

Recientemente se intenta revivir una alternativa algo olvidada:

- **Entrenamiento en Baja Precisión**
    - _BitNet 1.58_.

O incluso, hacer un mayor esfuerzo para alejarnos del Transformer:

- **Nuevas Arquitecturas**
    - Arquitecturas que no escalen cuadráticamente (por ejemplo, Mamba 2, RMKV, xLSTM).
    - Modelos con capas híbridas.

### Recursos Recomendados

- **Papers**:
    - **Mixtral of Experts (Jiang et al., 2024)**  
        La idea detrás es antigua, incluso del siglo pasado. Puedes leer este paper para comprender cómo se utiliza la arquitectura en un LLM.  
        [https://arxiv.org/abs/2401.04088](https://arxiv.org/abs/2401.04088)
    - **BitNet: Scaling 1-bit Transformers for Large Language Models (Wang et al., 2023)**  
        [https://arxiv.org/abs/2310.11453](https://arxiv.org/abs/2310.11453)
- **Blogs y Tutoriales**:
    - **Mamba: The Hard Way** – Blog de Sasha Rush. Es un recurso didáctico que probablemente requiera leer un par de veces, pero cuando lo entiendes, es muy satisfactorio.  
        [https://srush.github.io/annotated-mamba/hard.html](https://srush.github.io/annotated-mamba/hard.html)
    - **ELI5: FlashAttention** – Blog de Aleksa Gordic, que explica en detalle y de manera accesible esta optimización.  
        [https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
    - [RMKV Repository](https://github.com/BlinkDL/RWKV-LM) – Aquí están todas las actualizaciones de esta arquitectura.
    - **OpenBioLLM Blog** – Un modelo especializado en medicina basado en Llama3.  
        [https://www.saama.com/introducing-openbiollm-llama3-70b-8b-saamas-ai-research-lab-released-the-most-openly-available-medical-domain-llms-to-date/](https://www.saama.com/introducing-openbiollm-llama3-70b-8b-saamas-ai-research-lab-released-the-most-openly-available-medical-domain-llms-to-date/)

---

## 10. El Estado del Arte Actual

En términos de preentrenamiento (modelos base):

- **Deepseek V3**  
    [https://arxiv.org/abs/2412.19437v1](https://arxiv.org/abs/2412.19437v1)

En términos de _post-training_:

- **Tulu 3**  
    [https://arxiv.org/abs/2411.15124](https://arxiv.org/abs/2411.15124)

En construcción de _datasets_:

- **Fineweb / Fineweb Edu**  
    [https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
    
En modelos _encoders_:
- **ModernBert**
    [https://github.com/AnswerDotAI/ModernBERT](https://github.com/AnswerDotAI/ModernBERT)

---

## 11. Test Time Compute & RL (El Nuevo Paradigma)

Esta área de investigación recién ha empezado a gestarse hace un par de meses y ha mostrado resultados en modelos abiertos hace pocas semanas. Algunos papers relevantes podrían ser:

- **Deepseek R1**: Modelo de nivel O1 entrenado con un simple RL (el primer intento completamente exitoso).  
    [Ver PDF](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)
- **Kimi k1.5**: Otro modelo de calidad inferior a R1 que también quiere unirse a la fiesta.  
    [https://arxiv.org/abs/2501.12599v1](https://arxiv.org/abs/2501.12599v1)
- **O1 Replication Journey**: El laboratorio GAIR-NLP está posteando reportes constantes de su camino en la replicación de O1.  
    [https://github.com/GAIR-NLP/O1-Journey](https://github.com/GAIR-NLP/O1-Journey)
- **s1: Simple Test-Time Scaling**  
    [https://arxiv.org/abs/2501.12599v1](https://arxiv.org/abs/2501.12599v1)

---

## 12. Alineación y Seguridad

### Temas Clave

- **Riesgos de Desalineación e Impacto Social**
    - _Bias_, desinformación y generación de contenido dañino.
    - Preocupaciones de privacidad (memorization de datos personales por el modelo).
- **Técnicas de Alineación**
    - Bucles de retroalimentación humana: RLHF, métodos basados en debate.
    - _Constitutional AI_, RL orientado por políticas (investigación de Anthropic).
    - _Red-teaming_ y pruebas adversariales.
- **Gobernanza y Regulación del Modelo**
    - Marcos legales emergentes y gobernanza de datos.
    - Transparencia, rendición de cuentas y procesos de auditoría.
- **Investigación en Seguridad a Largo Plazo**
    - Discusiones sobre riesgos existenciales de la IA.
    - Colaboraciones entre el mundo académico, la industria y organismos políticos.

### Recursos Recomendados

- **Papers**:
    - **Constitutional AI: Harmlessness from AI Feedback (Bai et al., Anthropic)**  
        [https://arxiv.org/abs/2212.08073](https://arxiv.org/abs/2212.08073)
    - **On the Opportunities and Risks of Foundation Models (Stanford CRFM, Bommasani et al.)**  
        [https://arxiv.org/abs/2108.07258](https://arxiv.org/abs/2108.07258)
- **Blogs y Organizaciones**:
    - **Blog de Alignment de Anthropic**  
        [https://www.anthropic.com/research#alignment](https://www.anthropic.com/research#alignment)
- **Conferencias y Video Clases**:
    - **AI Safety Discussions** (por Stuart Russell, Andrew Ng, OpenAI, Anthropic, etc.)  
        (Busca “Stuart Russell AI Safety” en YouTube)  
        [YouTube Search](https://www.youtube.com/results?search_query=Stuart+Russell+AI+safety)
    - **Panels on Ethical AI at Conferences (NeurIPS, ICML, ICLR)**  
        [YouTube Search: “ethical AI panel”](https://www.youtube.com/results?search_query=ethical+AI+panel)

---

## 13. Despliegue y Producción

### Temas Clave

- **Escalado de la Inferencia**
    - Servir el modelo en GPU, CPU o hardware especializado.
    - Cacheo de _prompts_/respuestas frecuentes.
    - _Sharding_ del modelo para LLMs extremadamente grandes.
- **Latencia, Throughput y Coste**
    - _Quantization_ (int8, int4), _pruning_, _distillation_ para acelerar la inferencia.
    - _Dynamic gating_ (modelo pequeño vs. modelo grande).
    - Procesamiento por lotes y políticas de _autoscaling_.
- **Seguridad y Privacidad**
    - Ataques de _prompt injection_ y exfiltración de datos.
    - Ejecutar LLMs privados sobre datos propietarios.
- **Mantenimiento y Actualización del Modelo**
    - Monitoreo del _drift_ del modelo.
    - Estrategias de versionado, _rollback_ y _A/B testing_.

### Recursos Recomendados

- **Herramientas**:
    - ONNX Runtime para inferencia optimizada multiplataforma.
    - [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) para _serving_ acelerado por GPU.
    - Hugging Face Accelerate para inferencia distribuida.
    - Llama.cpp: repositorio de inferencia multiplataforma para LLMs.
    - GGUF format: el segundo formato más popular para compartir LLMs.
    - Ollama: montar un servidor de inferencia de LLMs súper rápido.
- **Conferencias y Video Clases**:
    - Sesiones en NVIDIA GTC sobre _serving_ y optimización de modelos.
    - Meetups de Hugging Face sobre mejores prácticas en producción.
    - ¡Servidor de Discord "CUDA mode"!

---

## 14. Experimentación y Metodología de Investigación

### Temas Clave

- **Investigación Reproducible**
    - Uso de entornos Docker/Conda y dependencias fijadas.
    - Control de versiones en _datasets_ y código (Git, DVC).
- **Experimentación Basada en Hipótesis**
    - Definición de _baselines_ claros y estudios de _ablation_.
    - Pruebas de significancia estadística y muestreo adecuado.
- **Colaboración y Comunidad**
    - Contribuciones _open-source_ (Hugging Face, EleutherAI).
    - Grupos de lectura de papers y canales en Slack/Discord.
- **Publicación Ética y Transparente**
    - Compartir código, modelos entrenados o _setups_ mínimos reproducibles.
    - Comunicación responsable de capacidades y limitaciones.

### Recursos Recomendados

- **Guías y Blogs**:
    - [Papers with Code](https://paperswithcode.com/) para rastreadores de código y reproducibilidad.
- **Herramientas**:
    - MLflow, Weights & Biases (wandb) para seguimiento de experimentos.
    - TensorBoard para registro y visualización.
- **Conferencias y Video Clases**:
    - **Tutorials on Reproducible Machine Learning (from ML conferences)**  
        [YouTube Search](https://www.youtube.com/results?search_query=reproducible+machine+learning+tutorial)
