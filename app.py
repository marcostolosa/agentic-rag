import os
import faiss
import html as std_html
import requests
import numpy as np
from datetime import datetime
import streamlit as st
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import torch
import pdfplumber
import logging
import json

# Configuração do Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgenticRAG")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Configurações Iniciais do Streamlit
st.set_page_config(
    page_title="Agentic-RAG",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Variáveis de Ambiente
os.environ["TOKENIZERS_PARALLELISM"] = "false"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SIMILARITY_TOP_K_DEFAULT = int(os.getenv("SIMILARITY_TOP_K_DEFAULT", 2))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Chave da API do OpenAI não encontrada. Configure OPENAI_API_KEY.")
    st.stop()

# Inicializar o cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Classe de Contexto para o Pentest
class PentestContext:
    def __init__(self):
        self.phase = "Reconhecimento"
        self.target = ""
        self.vulnerabilities = []
        self.report = []

context = PentestContext()

# Verificar se GPU está disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    logger.info("GPU detectada! Usando CUDA para acelerar o processamento.")
else:
    logger.info("Nenhuma GPU detectada. Usando CPU para processamento.")

# Funções de Negócio

def sanitize_input(text):
    """Sanitiza a entrada do usuário para evitar XSS."""
    return std_html.escape(text.strip())

def extract_text_from_pdf(file_path):
    """Extrai texto de arquivos PDF usando pdfplumber."""
    logger.info(f"Extraindo texto do PDF: {file_path}")
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    logger.info(f"Página {page_num} do PDF {file_path}: {len(page_text)} caracteres extraídos")
                else:
                    logger.warning(f"Página {page_num} do PDF {file_path}: Nenhum texto extraído")
            extracted_text = text.strip()
            if extracted_text:
                logger.info(f"Texto extraído do PDF {file_path}: {len(extracted_text)} caracteres")
            else:
                logger.warning(f"Nenhum texto extraído do PDF {file_path}")
            return extracted_text
    except Exception as e:
        logger.error(f"Erro ao extrair texto do PDF {file_path}: {str(e)}")
        st.warning(f"Erro ao extrair texto do PDF {file_path}: {str(e)}")
        return ""

def split_text(text, max_length=50000):
    """Divide texto longo em partes menores."""
    parts = []
    current_part = ""
    for paragraph in text.split("\n"):
        if len(current_part) + len(paragraph) + 1 > max_length:
            if current_part:
                parts.append(current_part.strip())
            current_part = paragraph
        else:
            if current_part:
                current_part += "\n" + paragraph
            else:
                current_part = paragraph
    if current_part:
        parts.append(current_part.strip())
    return parts

def load_documents():
    """Carrega documentos da pasta 'documents/' e os processa."""
    logger.info("Iniciando carregamento de documentos da pasta 'documents/'")
    if not os.path.exists('documents'):
        os.makedirs('documents')
        logger.warning("Pasta 'documents' criada. Nenhum arquivo encontrado.")
        st.warning("Pasta 'documents' criada. Adicione arquivos para começar!")
        return None

    files = os.listdir('documents')
    logger.info(f"Arquivos encontrados na pasta 'documents/': {files}")
    if not files:
        logger.warning("Nenhum arquivo encontrado na pasta 'documents/'")
        st.warning("Nenhum documento encontrado. Adicione arquivos para começar!")
        return None

    reader = SimpleDirectoryReader('documents')
    documents = reader.load_data()
    logger.info(f"Documentos carregados: {len(documents)}")
    if not documents:
        logger.warning("Nenhum documento carregado pela SimpleDirectoryReader")
        st.warning("Nenhum documento encontrado. Adicione arquivos para começar!")
        return None

    grouped_docs = {}
    for doc in documents:
        file_name = doc.metadata.get('file_name', 'desconhecido')
        if file_name not in grouped_docs:
            grouped_docs[file_name] = []
        grouped_docs[file_name].append(doc)

    valid_documents = []
    for file_name, docs in grouped_docs.items():
        logger.info(f"Processando arquivo: {file_name}")
        full_text = "\n".join(doc.text for doc in docs if doc.text)
        logger.info(f"Texto concatenado do arquivo {file_name}: {len(full_text)} caracteres")

        if not full_text or not full_text.strip():
            logger.warning(f"Arquivo {file_name} está vazio ou não contém texto legível. Pulando.")
            st.warning(f"Arquivo {file_name} está vazio ou não contém texto legível.")
            continue

        metadata = docs[0].metadata
        if len(full_text) > 50000:
            logger.info(f"Arquivo {file_name} tem {len(full_text)} caracteres. Dividindo em partes.")
            text_parts = split_text(full_text, max_length=50000)
            logger.info(f"Arquivo {file_name} dividido em {len(text_parts)} partes.")
            for part_num, part_text in enumerate(text_parts, 1):
                part_doc = Document(
                    text=part_text,
                    metadata={
                        **metadata,
                        "part_number": part_num,
                        "total_parts": len(text_parts),
                        "original_file_name": file_name
                    }
                )
                logger.info(f"Parte {part_num}/{len(text_parts)} do arquivo {file_name}: {len(part_text)} caracteres")
                valid_documents.append(part_doc)
        else:
            logger.info(f"Arquivo {file_name} válido: {len(full_text)} caracteres")
            valid_doc = Document(
                text=full_text,
                metadata=metadata
            )
            valid_documents.append(valid_doc)

    if not valid_documents:
        logger.error("Nenhum documento válido encontrado após verificação.")
        st.error("Nenhum documento válido encontrado. Certifique-se de que os arquivos contêm texto legível.")
        return None
    logger.info(f"Documentos válidos carregados: {len(valid_documents)}")
    return valid_documents

def create_or_load_index():
    """Cria ou carrega um índice FAISS para os documentos."""
    logger.info(f"Carregando modelo de embeddings: {EMBEDDING_MODEL}")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, device=device)
    logger.info("Modelo de embeddings carregado com sucesso")

    documents = load_documents()
    if not documents:
        logger.error("Falha ao carregar documentos. Indexação interrompida.")
        return None

    logger.info("Iniciando processo de indexação")
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        logger.info("Índice carregado com sucesso do armazenamento")
        st.success("Índice carregado com sucesso!")
        return index
    except Exception as e:
        logger.info(f"Nenhum índice encontrado ou erro ao carregar: {str(e)}. Criando um novo índice.")
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=Settings.embed_model
        )
        logger.info("Dividindo documentos em nós")
        nodes = splitter.get_nodes_from_documents(documents)
        logger.info(f"Nós gerados: {len(nodes)}")

        valid_nodes = [node for node in nodes if node.text and node.text.strip()]
        if not valid_nodes:
            logger.error("Nenhum nó válido encontrado para indexação.")
            st.error("Nenhum nó válido encontrado. Verifique se os documentos contêm texto legível.")
            return None
        logger.info(f"Nós válidos: {len(valid_nodes)}")

        logger.info("Gerando embeddings para os nós")
        for node in valid_nodes:
            file_name = node.metadata.get('file_name', 'desconhecido')
            try:
                node.embedding = Settings.embed_model.get_text_embedding(node.text)
                logger.info(f"Embedding gerado para nó do documento {file_name}: {len(node.text)} caracteres")
            except Exception as e:
                logger.warning(f"Erro ao gerar embedding para o nó do documento {file_name}: {str(e)}")
                st.warning(f"Erro ao gerar embedding para o nó do documento {file_name}: {str(e)}")
                continue
        embeddings = [node.embedding for node in valid_nodes if node.embedding is not None]
        if not embeddings:
            logger.error("Nenhum embedding gerado.")
            st.error("Nenhum embedding gerado. Verifique se os documentos contêm texto legível.")
            return None
        logger.info(f"Embeddings gerados: {len(embeddings)}")

        logger.info("Criando índice FAISS")
        dimension = 384
        faiss_index = faiss.IndexFlatL2(dimension)

        embeddings_array = np.array(embeddings, dtype=np.float32)
        logger.info(f"Embeddings convertidos para array NumPy: {embeddings_array.shape}")

        if embeddings_array.shape[0] == 0:
            logger.error("Nenhum embedding válido para o índice FAISS.")
            st.error("Nenhum embedding válido para o índice FAISS. Verifique os documentos.")
            return None

        if embeddings_array.shape[1] != dimension:
            logger.error(f"Dimensão dos embeddings ({embeddings_array.shape[1]}) não corresponde à dimensão esperada ({dimension}).")
            st.error(f"Dimensão dos embeddings ({embeddings_array.shape[1]}) não corresponde à dimensão esperada ({dimension}).")
            return None

        faiss_index.add(embeddings_array)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes=valid_nodes,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )
        index.storage_context.persist(persist_dir="./storage")
        logger.info("Índice FAISS criado e salvo com sucesso")
        st.success("Índice criado e salvo com sucesso!")
        return index

def save_uploaded_files(uploaded_files):
    """Salva arquivos enviados na pasta 'documents/'."""
    if not os.path.exists('documents'):
        os.makedirs('documents')
    for uploaded_file in uploaded_files:
        file_path = os.path.join('documents', uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Arquivo salvo: {file_path}")
        st.success(f"Arquivo {uploaded_file.name} salvo com sucesso!")

def document_search(query: str) -> dict:
    """Busca informações nos documentos indexados."""
    if "index" not in st.session_state or st.session_state.index is None:
        return {"response": "Erro: Índice não inicializado.", "sources": "Nenhuma fonte disponível"}
    try:
        query_engine = st.session_state.index.as_query_engine(
            similarity_top_k=st.session_state.similarity_slider if 'similarity_slider' in st.session_state else SIMILARITY_TOP_K_DEFAULT
        )
        response = query_engine.query(query)
        response_text = str(response)

        sources = {}
        total = sum(n.score for n in response.source_nodes) if response.source_nodes else 1
        for node in response.source_nodes:
            source = node.metadata.get('file_name', 'Desconhecido')
            part_num = node.metadata.get('part_number', '')
            if part_num:
                source += f" (Parte {part_num})"
            sources[source] = sources.get(source, 0) + node.score
        sources_str = " | ".join([f"{(v/total)*100:.0f}% de {k}" for k, v in sources.items()]) if sources else "Nenhuma fonte disponível"

        return {"response": response_text, "sources": sources_str}
    except Exception as e:
        logger.error(f"Erro na busca de documentos: {str(e)}")
        return {"response": f"Erro na busca de documentos: {str(e)}", "sources": "Nenhuma fonte disponível"}

def search_cve(cve_id: str) -> str:
    """Busca informações sobre uma CVE específica."""
    try:
        response = requests.get(f"https://api.nvd.nist.gov/rest/json/cve/1.0/{cve_id}", timeout=10)
        return json.dumps(response.json())
    except Exception as e:
        return f"Erro ao buscar CVE {cve_id}: {str(e)}"

def create_assistant():
    """Cria um assistente OpenAI para pentest."""
    try:
        assistant = client.beta.assistants.create(
            name="Pentest Assistant",
            instructions=f"""
            Você é um especialista em pentest seguindo o workflow:
            1. Reconhecimento
            2. Enumeração
            3. Exploração
            4. Relatório
            Acompanhe o usuário em todas as fases, fornecendo insights, dicas de especialistas e comandos úteis.
            Use as ferramentas disponíveis para buscar informações nos documentos locais e responder às perguntas.
            A fase atual do pentest é: {context.phase}.
            O alvo atual é: {context.target or 'Não definido'}.
            Sempre que responder usando informações de documentos, inclua as fontes no formato: (Fontes: [fontes aqui]).
            """,
            model="gpt-4",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "document_search",
                        "description": "Ferramenta para buscar informações nos documentos indexados.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "A consulta para buscar nos documentos"}
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_cve",
                        "description": "Busca informações sobre uma CVE específica.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "cve_id": {"type": "string", "description": "O ID da CVE (ex: CVE-2021-1234)"}
                            },
                            "required": ["cve_id"]
                        }
                    }
                }
            ]
        )
        logger.info("Assistente criado com sucesso")
        return assistant
    except Exception as e:
        logger.error(f"Erro ao criar o assistente: {str(e)}")
        st.error(f"Erro ao criar o assistente: {str(e)}")
        return None

def process_message(assistant, thread, question):
    """Processa uma mensagem do usuário usando o assistente OpenAI."""
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=question
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    sources = "Nenhuma fonte disponível"
    while run.status in ["queued", "in_progress", "requires_action"]:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        if run.status == "requires_action":
            tool_outputs = []
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                if function_name == "document_search":
                    result = document_search(arguments["query"])
                    output = result["response"]
                    sources = result["sources"]
                elif function_name == "search_cve":
                    output = search_cve(arguments["cve_id"])
                    sources = "NVD API"
                else:
                    output = f"Função {function_name} não reconhecida."
                    sources = "Nenhuma fonte disponível"
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": output
                })
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    for message in messages.data:
        if message.role == "assistant":
            response_text = message.content[0].text.value
            if "Fontes:" not in response_text:
                response_text += f"\n(Fontes: {sources})"
            return response_text
    return "Nenhuma resposta encontrada.\n(Fontes: Nenhuma fonte disponível)"

# Funções de Interface

def list_document_files():
    """Lista os arquivos na pasta 'documents/'."""
    if not os.path.exists('documents'):
        return []
    files = os.listdir('documents')
    file_list = []
    for file in files:
        file_path = os.path.join('documents', file)
        if os.path.isfile(file_path):
            stats = os.stat(file_path)
            file_list.append({
                "Nome": file,
                "Data": datetime.fromtimestamp(stats.st_mtime).strftime('%d/%m/%Y %H:%M'),
                "Tipo": os.path.splitext(file)[1][1:].upper() or "Desconhecido",
                "Tamanho": f"{stats.st_size / 1024:.2f} KB"
            })
    return file_list

def suggest_commands(context):
    """Sugere comandos com base na fase do pentest."""
    if context.phase == "Reconhecimento":
        return f"Use 'nmap -sV {context.target}' para escanear serviços no alvo."
    elif context.phase == "Enumeração":
        return f"Tente 'enum4linux -a {context.target}' para enumeração SMB."
    elif context.phase == "Exploração":
        return f"Use 'msfconsole' para explorar vulnerabilidades no {context.target}."
    return "Documente suas descobertas no relatório."

def export_report(history):
    """Exporta o histórico de conversas como um relatório em PDF."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    y = 750
    c.drawString(100, y, f"Relatório de Pentest - {datetime.now().strftime('%Y-%m-%d')}")
    y -= 30
    c.drawString(100, y, "Histórico de Consultas")
    y -= 20
    for entry in history:
        c.drawString(100, y, f"Pergunta: {entry['question']}")
        y -= 20
        c.drawString(100, y, "Resposta:")
        y -= 15
        response_lines = entry['response'].split('\n')
        for line in response_lines[:5]:
            c.drawString(120, y, line[:80])
            y -= 15
        c.drawString(100, y, f"Fontes: {entry['sources']}")
        y -= 30
        if y < 50:
            c.showPage()
            y = 750
    c.save()
    buffer.seek(0)
    st.download_button(
        "Download PDF",
        buffer,
        file_name="agentic_rag_report.pdf",
        mime="application/pdf"
    )

# Interface Principal
def main():
    # Estilo CSS
    st.markdown("""
    <style>
        .stApp {
            background-color: #0D1117;
            color: #C9D1D9;
            font-family: 'Consolas', monospace;
            margin: 0;
            padding: 0;
        }

        /* Barra de Input Fixa no Topo */
        .input-container {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background-color: #161B22;
            padding: 10px 15px;
            z-index: 1000;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .input-container .stTextInput > div > input {
            background-color: #21262D;
            border: 1px solid #30363D;
            border-radius: 5px;
            padding: 8px 12px;
            color: #C9D1D9;
            font-size: 14px;
            outline: none;
            width: 100%;
        }

        .input-container .stTextInput > div > input::placeholder {
            color: #8B949E;
        }

        .input-container .stButton > button {
            background-color: #00CC00;
            border: none;
            border-radius: 5px;
            padding: 8px 15px;
            color: #0D1117;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s ease;
        }

        .input-container .stButton > button:hover {
            background-color: #00B300;
        }

        .toggle-button {
            background: none;
            border: none;
            color: #00CC00;
            font-size: 20px;
            cursor: pointer;
        }

        /* Área de Conteúdo */
        .content-container {
            margin-top: 70px;
            padding: 20px;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        }

        /* Estilo dos Cards de Conversa */
        .chat-card {
            background-color: #161B22;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
        }

        .chat-header {
            background-color: #21262D;
            padding: 10px 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .chat-header h3 {
            margin: 0;
            font-size: 14px;
            font-weight: 500;
            color: #C9D1D9;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .chat-header h3 .icon {
            color: #00CC00;
            font-size: 16px;
        }

        .chat-header .stButton > button {
            background: none;
            border: none;
            color: #8B949E;
            font-size: 14px;
            padding: 5px;
        }

        .chat-content {
            padding: 15px;
            background-color: #161B22;
        }

        .chat-message {
            margin-bottom: 10px;
        }

        .chat-message strong {
            color: #00CC00;
        }

        .chat-message p {
            margin: 5px 0;
            line-height: 1.5;
            font-size: 13px;
        }

        /* Estilo da Sidebar */
        .stSidebar {
            background-color: #0D1117;
            padding: 15px;
            width: 280px !important;
        }

        .sidebar-content h2 {
            font-size: 18px;
            font-weight: 600;
            color: #C9D1D9;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .sidebar-content h2 .icon {
            color: #00CC00;
            font-size: 20px;
        }

        .sidebar-content .stSelectbox > div,
        .sidebar-content .stTextInput > div > input,
        .sidebar-content .stSlider > div {
            background-color: #21262D;
            border: 1px solid #30363D;
            border-radius: 5px;
            padding: 8px;
            color: #C9D1D9;
            font-size: 13px;
            margin-bottom: 15px;
        }

        .sidebar-content .stButton > button {
            background-color: #00CC00;
            border: none;
            border-radius: 5px;
            padding: 10px;
            color: #0D1117;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            margin-bottom: 10px;
            transition: background-color 0.2s ease;
        }

        .sidebar-content .stButton > button:hover {
            background-color: #00B300;
        }

        .sidebar-content .file-list {
            margin-top: 15px;
        }

        .sidebar-content .file-list h3 {
            font-size: 14px;
            font-weight: 500;
            color: #C9D1D9;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .sidebar-content .file-list h3 .icon {
            color: #00CC00;
            font-size: 16px;
        }

        .sidebar-content .file-list ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .sidebar-content .file-list li {
            background-color: #21262D;
            border-radius: 5px;
            padding: 8px;
            margin-bottom: 5px;
            font-size: 12px;
            color: #C9D1D9;
        }

        /* Sugestões */
        .suggestion {
            margin-top: 20px;
            padding: 15px;
            background-color: #21262D;
            border-radius: 5px;
            font-size: 13px;
            color: #C9D1D9;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
        }

        .suggestion strong {
            color: #00CC00;
        }
    </style>
    """, unsafe_allow_html=True)

    # Ícones
    icons = {
        "menu": "☰",
        "send": "➤",
        "question": "❓",
        "upload": "⬆",
        "clear": "🗑️",
        "download": "⬇",
        "files": "📂",
        "settings": "⚙️",
        "expand": "▼",
        "collapse": "▲"
    }

    # Estado da Sidebar (fechada por padrão)
    if "sidebar_open" not in st.session_state:
        st.session_state.sidebar_open = False

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown(f'<h2><span class="icon">{icons["settings"]}</span> Configurações</h2>', unsafe_allow_html=True)

        context.phase = st.selectbox(
            label="Fase Atual",
            options=["Reconhecimento", "Enumeração", "Exploração", "Relatório"],
            key="phase_select",
            label_visibility="collapsed"
        )

        context.target = st.text_input(
            label="Alvo",
            placeholder="Ex: 192.168.1.1",
            key="target_input",
            label_visibility="collapsed"
        )

        st.session_state.similarity_slider = st.slider(
            label="Resultados por Consulta",
            min_value=1,
            max_value=5,
            value=SIMILARITY_TOP_K_DEFAULT,
            help="Ajuste o número de resultados retornados pelo assistente.",
            label_visibility="collapsed"
        )

        uploaded_files = st.file_uploader(
            label="Upload de Arquivos",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files:
            with st.spinner("Salvando e indexando arquivos..."):
                save_uploaded_files(uploaded_files)
                st.session_state.index = create_or_load_index()

        if st.button(
            label=f"{icons['clear']} Limpar Histórico",
            help="Limpar o histórico de conversas",
            key="clear_history"
        ):
            st.session_state.conversation_history = []
            st.session_state.index = create_or_load_index()
            st.session_state.assistant = create_assistant()
            st.session_state.thread = client.beta.threads.create()
            st.success("Histórico limpo!")

        if st.button(
            label=f"{icons['download']} Exportar Relatório",
            help="Exportar o relatório em PDF",
            key="export_report"
        ):
            with st.spinner("Gerando relatório..."):
                export_report(st.session_state.conversation_history)

        # Lista de Arquivos
        files = list_document_files()
        if files:
            st.markdown('<div class="file-list">', unsafe_allow_html=True)
            st.markdown(f'<h3><span class="icon">{icons["files"]}</span> Arquivos Indexados</h3>', unsafe_allow_html=True)
            st.markdown('<ul>', unsafe_allow_html=True)
            for file in files:
                st.markdown(f'<li>{file["Nome"]} ({file["Tipo"]}, {file["Tamanho"]})</li>', unsafe_allow_html=True)
            st.markdown('</ul>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="file-list">', unsafe_allow_html=True)
            st.markdown(f'<h3><span class="icon">{icons["files"]}</span> Arquivos Indexados</h3>', unsafe_allow_html=True)
            st.markdown('<p style="color: #8B949E; font-size: 12px;">Nenhum arquivo encontrado.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Barra de Input Fixa no Topo
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        st.button(
            label=f"{icons['menu']}",
            key="sidebar_toggle",
            help="Abrir ou fechar a sidebar",
            on_click=lambda: setattr(st.session_state, "sidebar_open", not st.session_state.sidebar_open)
        )

        question = sanitize_input(
            st.text_input(
                label="Digite sua pergunta",
                placeholder="Digite sua pergunta (ex.: Quais vulnerabilidades afetam meu alvo?)",
                key="question_input",
                label_visibility="collapsed"
            )
        )
        
        if st.button(
            label=f"{icons['send']} Enviar",
            help="Enviar a pergunta",
            key="send_button"
        ):
            if not question:
                st.warning("Digite uma pergunta válida!")
            else:
                with st.spinner("Processando sua pergunta..."):
                    try:
                        if st.session_state.assistant is None or st.session_state.thread is None:
                            raise ValueError("O assistente não foi inicializado. Tente limpar o histórico ou reiniciar o aplicativo.")
                        client.beta.assistants.update(
                            assistant_id=st.session_state.assistant.id,
                            instructions=f"""
                            Você é um especialista em pentest seguindo o workflow:
                            1. Reconhecimento
                            2. Enumeração
                            3. Exploração
                            4. Relatório
                            Acompanhe o usuário em todas as fases, fornecendo insights, dicas de especialistas e comandos úteis.
                            Use as ferramentas disponíveis para buscar informações nos documentos locais e responder às perguntas.
                            A fase atual do pentest é: {context.phase}.
                            O alvo atual é: {context.target or 'Não definido'}.
                            Sempre que responder usando informações de documentos, inclua as fontes no formato: (Fontes: [fontes aqui]).
                            """
                        )
                        response_text = process_message(st.session_state.assistant, st.session_state.thread, question)
                        logger.info("Resposta gerada com sucesso")

                        context.report.append(response_text)

                        if "conversation_history" not in st.session_state:
                            st.session_state.conversation_history = []
                        st.session_state.conversation_history.append({
                            "question": question,
                            "response": response_text,
                            "sources": response_text.split("(Fontes: ")[-1].rstrip(")") if "(Fontes: " in response_text else "Nenhuma fonte disponível"
                        })

                    except Exception as e:
                        logger.error(f"Erro ao processar pergunta: {str(e)}")
                        st.error(f"Erro durante processamento: {str(e)}")
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Área de Conteúdo
    with st.container():
        st.markdown('<div class="content-container">', unsafe_allow_html=True)

        # Exibir Fase e Alvo
        st.markdown(f"""
        <div style="margin-bottom: 15px;">
            <p style="font-size: 12px; color: #8B949E;">
                <strong style="color: #00CC00;">Fase Atual:</strong> {context.phase} | 
                <strong style="color: #00CC00;">Alvo:</strong> {context.target or 'Não definido'}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Exibir Histórico de Conversas (mais recente no topo)
        if "conversation_history" in st.session_state and st.session_state.conversation_history:
            history = list(reversed(st.session_state.conversation_history))

            if 'expanded_states' not in st.session_state:
                st.session_state.expanded_states = {i: False for i in range(len(history))}
                if history:
                    st.session_state.expanded_states[0] = True

            for idx, entry in enumerate(history):
                with st.container():
                    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
                    col1, col2 = st.columns([8, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="chat-header">
                            <h3><span class="icon">{icons["question"]}</span>{entry['question'][:50]}{"..." if len(entry['question']) > 50 else ""}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        is_expanded = st.session_state.expanded_states.get(idx, False)
                        if st.button(
                            label=icons["collapse"] if is_expanded else icons["expand"],
                            key=f"toggle_{idx}",
                            help="Expandir ou contrair a resposta"
                        ):
                            st.session_state.expanded_states[idx] = not is_expanded

                    if st.session_state.expanded_states.get(idx, False):
                        st.markdown(f"""
                        <div class="chat-content">
                            <div class="chat-message">
                                <strong>Pergunta:</strong>
                                <p>{entry['question']}</p>
                            </div>
                            <div class="chat-message">
                                <strong>Resposta:</strong>
                                <p>{entry['response']}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="suggestion">
                <strong>Sugestão:</strong> {suggest_commands(context)}
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <p style="color: #8B949E; font-size: 14px; text-align: center; margin-top: 50px;">
                Nenhuma conversa iniciada. Faça uma pergunta para começar!
            </p>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# Inicialização
if "index" not in st.session_state:
    st.session_state.index = create_or_load_index()
if "assistant" not in st.session_state:
    st.session_state.assistant = create_assistant()
if "thread" not in st.session_state:
    st.session_state.thread = client.beta.threads.create()
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if __name__ == "__main__":
    main()