# Agentic RAG 🚀

[![Streamlit App](https://img.shields.io/badge/Streamlit-1.32.0-brightgreen)](https://streamlit.io/) [![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-Yes-green)](#como-usar) [![Made With](https://img.shields.io/badge/Made%20with-❤-e03997)](https://github.com/)

---

## 📖 Visão Geral

Agentic-RAG é uma aplicação web desenvolvida com Streamlit para auxiliar em atividades de pentest (teste de penetração), permitindo o upload e indexação de documentos, busca de informações e geração de relatórios em PDF. A aplicação utiliza embeddings de texto para buscar informações em documentos locais e integra um `assistente OpenAI` para responder perguntas relacionadas.

## Funcionalidades
- 📄 **Upload de Documentos:** Suporte para arquivos `.pdf`, `.txt` e `.md`.
- 🕵️ **Indexação de Documentos:** Usa embeddings gerados pelo modelo `sentence-transformers/all-MiniLM-L6-v2` e FAISS para busca eficiente.
- 💬 **Assistente de Pentest:** Integração com a API da OpenAI para responder perguntas com base nos documentos indexados e fornecer sugestões de comandos.
- 🔍 **Fases do Pentest:** Suporte para as fases de Reconhecimento, Enumeração, Exploração e Relatório.
- **Histórico de Conversas:** Exibe perguntas e respostas em cards expansíveis.
- **Exportação de Relatório:** Gera um relatório em PDF com o histórico de conversas incluso.

---

## ⚡ Funcionalidades Principais

| Funcionalidade          | Descrição                                                                 |
|-------------------------|---------------------------------------------------------------------------|
| **Chat Inteligente**     | Pergunte sobre vulnerabilidades, recomendações ou detalhes técnicos      |
| **Rastreamento de Fontes** | Identifica origem das informações com % de contribuição de cada documento |
| **Interface Moderna**    | Layout responsivo com temas escuros e animações suaves                    |
| **Histórico Persistente** | Mantém conversas mesmo após reiniciar o servidor                         |
| **Indexação Automática** | Detecta automaticamente novos arquivos na pasta `documents`              |

---

## 🛠 Como Usar

### 1. Instalação
```bash
# Clone o repositório
git clone https://github.com/marcostolosa/agentic-rog.git
cd agentic-rog/

# Crie ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instale dependências
pip install -r requirements.txt
```

### 2. Configuração
- Crie um arquivo `.env` com a `OpenAI API Key`
- Crie uma pasta chamada `documents` na raiz do projeto
- Adicione arquivos Markdown, PDF, TXT ou outros formatos suportados

### 3. Execução
```bash
streamlit run app.py
```

### 📂 Estrutura do Projeto

```
agentic-rog/
├── documents/          # ← Coloque seus arquivos aqui
├── app.py              # Código principal
├── requirements.txt    # Dependências
├── LICENSE             # MIT License
└── README.md           # Este arquivo
```

## 🤖 Como Funciona
1. **Indexação Automática**
Usa embeddings do HuggingFace (`all-MiniLM-L6-v2`) para criar representações vetoriais dos documentos
2. **Motor de Busca**
FAISS (Facebook AI Similarity Search) para busca rápida de similaridade
3. **Rastreamento de Fontes**
Analisa os metadados dos documentos para identificar origens das respostas
4. **Interface**
Streamlit com CSS customizado para experiência moderna


### 🤝 Contribuindo
1. Fork este repositório
2. Crie uma branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add feature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request