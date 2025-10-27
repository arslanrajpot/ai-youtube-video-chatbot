from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, embeddings_model="models/embedding-001", llm_model="gemini-1.5-flash", api_key=None):
        # Initialize embeddings with fallback
        self.embeddings = self._initialize_embeddings(embeddings_model, api_key)
        
        # Initialize LLM with fallback support
        self.google_api_key = api_key
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.llm = None
        self.llm_provider = None
        
        # Try to initialize Gemini first, then fallback to Groq
        self._initialize_llm()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.retriever_cache = {}  # Initialize the retriever cache
        self.user_info = {}  # Store user-provided information
        
        # Define general questions and responses
        self.general_responses = {
            r"(?i)\b(hi|hello|hey|greetings)\b": "Hello! I'm Video Talker, your assistant for exploring YouTube video content. Just submit a video URL, and I'll help you answer questions based on its transcript. How can I assist you today?",
            r"(?i)\bwho are you\b": "I'm Video Talker, a chatbot designed to help you dive into YouTube videos! Submit a video URL, and I can answer questions based on its transcript. What's on your mind?",
            r"(?i)\bhow are you\b": "I'm doing great as Video Talker, ready to help you with YouTube video insights! Submit a video, and let's explore its content together. What do you want to know?"
        }
        # Define patterns for storing user information
        self.info_patterns = {
            r"(?i)\bmy name is\s+([a-zA-Z]+)\b": "name"
        }
        # Define patterns for retrieving user information
        self.retrieval_patterns = {
            r"(?i)\bwhat(?:'s| is)\s+my\s+name\b": "name"
        }
        
        # Initialize the prompt template
        self.prompt = PromptTemplate(
            template="""
            You are a helpful assistant named Video Talker. Answer ONLY from the provided transcript context if the question is related to the video. 
            If the question is unrelated to the context, respond: "This video does not contain any information about that."
            If the question contains a term mentioned in the context but not explained, provide a general explanation of the term using your knowledge, and clarify that the explanation is not from the video.
            Context: {context}
            Question: {question}
            """,
            input_variables=["context", "question"]
        )
    
    def _initialize_embeddings(self, embeddings_model, api_key):
        """Initialize embeddings with fallback from Google to HuggingFace"""
        # Try Google embeddings first
        if api_key:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model, google_api_key=api_key)
                logger.info("✅ Google embeddings initialized successfully")
                return embeddings
            except Exception as e:
                logger.warning(f"⚠️ Google embeddings failed (quota exceeded?): {e}")
        
        # Fallback to HuggingFace embeddings (free, local)
        try:
            logger.info("🔄 Falling back to HuggingFace embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("✅ HuggingFace embeddings initialized successfully")
            return embeddings
        except Exception as e:
            logger.error(f"❌ HuggingFace embeddings also failed: {e}")
            raise Exception("Failed to initialize any embeddings. Please check your setup.")
    
    def _initialize_llm(self):
        """Initialize LLM with fallback from Gemini to Groq"""
        # Try Gemini first
        if self.google_api_key:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", 
                    google_api_key=self.google_api_key, 
                    temperature=0.2
                )
                self.llm_provider = "gemini"
                logger.info("✅ Initialized Gemini LLM successfully")
                return
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Gemini: {e}")
        
        # Fallback to Groq
        if self.groq_api_key:
            try:
                self.llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    groq_api_key=self.groq_api_key,
                    temperature=0.2
                )
                self.llm_provider = "groq"
                logger.info("✅ Initialized Groq LLM successfully")
                return
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Groq: {e}")
        
        # If both fail, raise an error
        raise Exception("❌ Failed to initialize any LLM. Please check your API keys for Google Gemini or Groq.")
    
    def _retry_with_fallback(self, question, retriever):
        """Retry with fallback LLM if the primary one fails"""
        try:
            return self._answer_with_current_llm(question, retriever)
        except Exception as e:
            logger.warning(f"⚠️ Primary LLM ({self.llm_provider}) failed: {e}")
            
            # Try to switch to the other provider
            if self.llm_provider == "gemini" and self.groq_api_key:
                logger.info("🔄 Switching to Groq as fallback...")
                try:
                    self.llm = ChatGroq(
                        model="llama-3.1-8b-instant",
                        groq_api_key=self.groq_api_key,
                        temperature=0.2
                    )
                    self.llm_provider = "groq"
                    return self._answer_with_current_llm(question, retriever)
                except Exception as groq_error:
                    logger.error(f"❌ Groq fallback also failed: {groq_error}")
            
            elif self.llm_provider == "groq" and self.google_api_key:
                logger.info("🔄 Switching to Gemini as fallback...")
                try:
                    self.llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash", 
                        google_api_key=self.google_api_key, 
                        temperature=0.2
                    )
                    self.llm_provider = "gemini"
                    return self._answer_with_current_llm(question, retriever)
                except Exception as gemini_error:
                    logger.error(f"❌ Gemini fallback also failed: {gemini_error}")
            
            # If all attempts fail
            raise Exception(f"❌ All LLM providers failed. Last error: {e}")
    
    def _answer_with_current_llm(self, question, retriever):
        """Answer question using the currently initialized LLM"""
        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        chain = (
            RunnableParallel({"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()})
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(question)

    def process_transcript(self, transcript_text, video_id):
        texts = self.text_splitter.create_documents([transcript_text])
        vector_store = FAISS.from_documents(texts, self.embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        self.retriever_cache[video_id] = retriever
        return retriever

    def get_retriever(self, video_id):
        return self.retriever_cache.get(video_id)

    def answer_question(self, retriever, question):
        # Store user information if provided
        for pattern, key in self.info_patterns.items():
            match = re.search(pattern, question)
            if match:
                self.user_info[key] = match.group(1)
                return f"Got it! I'll remember your {key} as {self.user_info[key]}. How can Video Talker assist you with a video?"

        # Handle questions about stored user information
        for pattern, key in self.retrieval_patterns.items():
            if re.search(pattern, question):
                if key in self.user_info:
                    return f"Your {key} is {self.user_info[key]}."
                return "I don't have that information yet. Please tell me, for example, 'My name is John.'"

        # Handle general questions
        for pattern, response in self.general_responses.items():
            if re.search(pattern, question):
                return response

        # Proceed with transcript-based answering if no general or user info question is matched
        try:
            answer = self._retry_with_fallback(question, retriever)
            # Check if the answer is empty or indicates no relevant context
            if not answer.strip() or "no information" in answer.lower():
                return "This video does not contain any information about that."
            return answer
        except Exception as e:
            logger.error(f"❌ Failed to get answer: {e}")
            return f"Sorry, I'm having trouble processing your question right now. Error: {str(e)}"