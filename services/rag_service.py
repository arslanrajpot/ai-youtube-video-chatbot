from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
import re

class RAGService:
    def __init__(self, embeddings_model="models/embedding-001", llm_model="gemini-1.5-flash", api_key=None):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model, google_api_key=api_key)
        self.llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=api_key, temperature=0.2)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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
        self.retriever_cache = {}
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
        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        chain = (
            RunnableParallel({"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()})
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        answer = chain.invoke(question)
        # Check if the answer is empty or indicates no relevant context
        if not answer.strip() or "no information" in answer.lower():
            return "This video does not contain any information about that."
        return answer