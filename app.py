import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from dotenv import load_dotenv
load_dotenv()  # Load only once


# Set environment variables
os.environ['USER_AGENT'] = 'myagent'

# Import necessary libraries
import bs4
from flask import Flask, render_template, jsonify, request, Response
import json
import time
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from tavily import TavilyClient


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,

    timeout=None,
    max_retries=2




)

# Initialize Tavily client for web search
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)

from langchain_community.document_loaders import WebBaseLoader

# ==============================================================================
# CENTRALIZED CONFIGURATION
# ==============================================================================

# Base URLs for the dental practice
BASE_URL = "https://www.edmondsbaydental.com"
PROCEDURES_URL = f"{BASE_URL}/procedures"

# Source URLs for knowledge base scraping
KNOWLEDGE_BASE_SOURCES = [
    "https://raw.githubusercontent.com/Dataworld123/Datas.txt/refs/heads/main/plain_text_crawled_data%20(1).txt"
]
WEBSITE_SCRAPE_URLS = [
    f"{BASE_URL}/",
    f"{BASE_URL}/dr-meenakshi-tomar-dds/",
    f"{BASE_URL}/our-practice/",
    f"{BASE_URL}/procedures/",
    f"{BASE_URL}/invisalign/",
    f"{BASE_URL}/faqs/"
]

# Mapping of keywords to specific page URLs
# The order matters: more specific keywords should come first.
LINK_MAPPING = {
    # Cleanings & Prevention
    "root_canal": (["root canal", "endodontic", "nerve", "pulp", "root canal therapy", "endodontic treatment"], f"{PROCEDURES_URL}/endodontics/root-canal-therapy/"),
    "Diagnodent": (["diagnodent", "cavity detection", "caries detection"], f"{PROCEDURES_URL}/cleanings-prevention/diagnodent/"),
    "Dental Exam & Cleanings": (["dental exam", "cleaning", "prophylaxis", "hygiene", "dental cleaning", "teeth cleaning"], f"{PROCEDURES_URL}/cleanings-prevention/dental-exams-cleanings/"),
    "Dental X-Rays": (["x-ray", "dental x-rays", "radiograph","Dental Xray"], f"{PROCEDURES_URL}/cleanings-prevention/dental-x-rays/"),
    "Digital X-Rays": (["digital x-ray", "digital radiograph"], f"{PROCEDURES_URL}/cleanings-prevention/dental-x-rays/"),
    "Fluoride Treatments": (["fluoride", "fluoride treatment", "fluorosis"], f"{PROCEDURES_URL}/cleanings-prevention/fluoride-treatments/"),
    "Home Care ": (["home care", "oral hygiene", "brushing", "flossing"], f"{PROCEDURES_URL}/cleanings-prevention/home-care/"),
    "Sealants": (["sealant", "pit", "fissure", "caries prevention"], f"{PROCEDURES_URL}/cleanings-prevention/sealants/"),
    # Cosmetic Dentistry
    "Composite Fillings": (["composite filling", "white filling", "esthetic filling"], f"{PROCEDURES_URL}/cosmetic-dentistry/composite-fillings/"),
    "Dental Implants": (["implant", "missing tooth", "tooth replacement", "dental implant", "implant surgery"], f"{PROCEDURES_URL}/cosmetic-dentistry/dental-implants/"),
    "Porcelain Crowns (Caps)": (["crown", "cap", "dental crown", "tooth cap"], f"{PROCEDURES_URL}/cosmetic-dentistry/porcelain-crowns-caps/"),
    "Porcelain Fixed Bridges ": (["bridge", "dental bridge", "gap created", "missing teeth"], f"{PROCEDURES_URL}/cosmetic-dentistry/porcelain-fixed-bridges/"),
    "Porcelain Inlays ": (["inlay", "onlay", "dental inlay", "dental onlay"], f"{PROCEDURES_URL}/cosmetic-dentistry/porcelain-inlays/"),
    "Porcelain Onlays ": (["onlay", "dental onlay"], f"{PROCEDURES_URL}/cosmetic-dentistry/porcelain-onlays/"),
    "Porcelain Veneers": (["veneer", "cosmetic", "smile makeover", "dental veneer", "porcelain veneer"], f"{PROCEDURES_URL}/cosmetic-dentistry/porcelain-veneers/"),
    "teeth_whitening": (["whitening", "bleaching", "white teeth", "teeth whitening", "tooth whitening", "brighten", "stain removal"], f"{PROCEDURES_URL}/cosmetic-dentistry/tooth-whitening/"),
    
    # Orthodontics
    "Orthodontists": (["orthodontist", "braces", "aligners", "straighten teeth", "crooked teeth", "malocclusion"], f"{PROCEDURES_URL}/orthodontics/what-is-an-orthodontist/"),
    " Malocclusion Correction"
    : (["malocclusion", "crooked teeth", "misaligned teeth", "bite correction"], f"{PROCEDURES_URL}/orthodontics/what-is-a-malocclusion-correction/"),
    "Who Can Benefit From Orthodontics": (["benefit from orthodontics", "who can benefit from orthodontics"], f"{PROCEDURES_URL}/orthodontics/who-can-benefit-from-orthodontics/"),
    "Orthodontic Treatment (Braces)": (["braces", "orthodontic braces", "orthodontic treatment"], f"{PROCEDURES_URL}/orthodontics/orthodontic-treatment/"),
    "Orthodontic Treatment Phases": (["orthodontic treatment phases", "phases of orthodontic treatment"], f"{PROCEDURES_URL}/orthodontics/orthodontic-treatment-phases/"),
    "Braces for Children": (["braces for children", "child braces", "early orthodontic treatment"], f"{PROCEDURES_URL}/orthodontics/braces-for-children/"),
    "Do Braces Hurt?": (["braces hurt", "pain from braces", "discomfort from braces"], f"{PROCEDURES_URL}/orthodontics/do-braces-hurt/"),
    "Care Following Orthodontics – Retainers": (["retainer", "care after braces", "retainers"], f"{PROCEDURES_URL}/orthodontics/care-following-orthodontics/"),
    "Orthodontic Dictionary": (["orthodontic dictionary", "orthodontic terms", "orthodontic glossary"], f"{PROCEDURES_URL}/orthodontics/orthodontic-dictionary/"),
    "Sleep Apnea ": (["sleep apnea", ], f"{PROCEDURES_URL}/orthodontics/sleep-apnea/"),
    "Sleep Apnea Appliances ": (["sleep apnea appliances", "sleep apnea treatment"], f"{PROCEDURES_URL}/orthodontics/sleep-apnea-appliances/"),
    "TMJ (Tempro-Mandibular Joint Dysfunction) ": (["tmj", "temporomandibular joint dysfunction"], f"{PROCEDURES_URL}/orthodontics/tmj-tempro-mandibular-joint-dysfunction/"),
    "Apicoectomy": (["apicoectomy", "root end surgery"], f"{PROCEDURES_URL}/endodontics/apicoectomy/"),
    "Bone Grafting": (["bone grafting", "ridge augmentation"], f"{PROCEDURES_URL}/endodontics/bone-grafting/"),
    "dental implants ": (["dental implants", "implant surgery"], f"{PROCEDURES_URL}/cosmetic-dentistry/dental-implants/"),

    # Periodontics
    "crowns": (["crown", "cap", "dental crown", "tooth cap"], f"{PROCEDURES_URL}/restorative-dentistry/crowns/"),
    "fillings": (["filling", "cavity", "decay", "dental filling", "tooth filling"], f"{PROCEDURES_URL}/restorative-dentistry/fillings/"),
    "dentures": (["denture"], f"{PROCEDURES_URL}/restorative-dentistry/dentures/"),
    "bridges": (["dental bridge", "gap created", "missing teeth", "fixed bridges"], f"{PROCEDURES_URL}/prosthodontics/fixed-bridges/"),
    # Preventive
    "cleanings": (["cleaning", "prophylaxis", "hygiene", "dental cleaning", "teeth cleaning"], f"{PROCEDURES_URL}/preventive-dentistry/cleanings/"),
    "fluoride": (["fluoride", "fluoride treatment", "fluorosis"], f"{PROCEDURES_URL}/preventive-dentistry/fluoride-treatment/"),
    # Pediatric
    "pediatric": (["baby bottle", "pediatric", "children", "kids", "child", "baby teeth"], f"{PROCEDURES_URL}/pediatric-dentistry/baby-bottle-tooth-decay/"),
    # Periodontics
    "gum_disease": (["gum disease", "periodontal", "gingivitis", "periodontitis", "gum infection"], f"{PROCEDURES_URL}/periodontics/gum-disease/"),
    # Oral Surgery
    "extractions": (["extraction", "wisdom tooth"], f"{PROCEDURES_URL}/oral-surgery/extractions/"),
    # Orthodontics
    "invisalign": (["invisalign", "clear aligners", "invisible braces", "clear braces"], f"{BASE_URL}/invisalign/"),
    "braces": (["braces", "orthodontic"], f"{PROCEDURES_URL}/orthodontics/braces/"),
    # Emergency
    "emergency": (["emergency", "tooth pain"], f"{PROCEDURES_URL}/emergency-dentistry/tooth-pain/"),
    # Practice Info
    "about": (["doctor", "dr", "dentist", "meenakshi", "about"], f"{BASE_URL}/dr-meenakshi-tomar-dds/"),
    "practice": (["practice", "office"], f"{BASE_URL}/our-practice/"),
    "contact": (["contact", "location", "address", "phone", "directions", "map", "appointment", "schedule", "book", "appoint"], f"{BASE_URL}/directions/"),
    "faqs": (["faq", "question", "help"], f"{BASE_URL}/faqs/"),
}

# Default link if no specific topic is found
DEFAULT_LINK = PROCEDURES_URL

# Keywords for topics the bot should gracefully decline to answer
OUT_OF_SCOPE_TOPICS = [
    "fees", "cost", "price", "insurance", "billing", "payment",
    "family identity", "personal information"
]

# Try multiple data sources with fallback
docs = []

# Option 1: Try Dr. Meenakshi Tomar's practice website
try:
    print("Attempting to load knowledge from primary source...")
    loader = WebBaseLoader(KNOWLEDGE_BASE_SOURCES)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} documents from practice website")
    if docs:
        print("Document content preview:", docs[0].page_content[:1000])
except Exception as e:
    print(f"❌ Failed to load from practice website: {str(e)}")

# Option 2: Try website scrape as fallback
if not docs:
    try:
        print("Attempting to load from website scrape...")
        loader = WebBaseLoader(WEBSITE_SCRAPE_URLS)
        docs = loader.load()
        print(f"✅ Loaded {len(docs)} documents from GitHub")
    except Exception as e:
        print(f"❌ Failed to load from GitHub: {str(e)}")

# Option 3: Create basic dental knowledge if no external sources work
if not docs:
    print("⚠️ No external sources available, creating basic dental knowledge base...")
    from langchain.schema import Document

    basic_dental_content = """
    Dr. Meenakshi Tomar, DDS - Dental Practice Information

    About Dr. Meenakshi Tomar:
    Dr. Meenakshi Tomar is a qualified dentist (DDS) specializing in comprehensive dental care.

    Services Offered:
    - General Dentistry
    - Preventive Care
    - Dental Cleanings
    - Fillings and Restorations
    - Cosmetic Dentistry
    - Teeth Whitening
    - Invisalign Treatment
    - Dental Examinations
    - Root Canal Therapy
    - Endodontic Treatment

    Common Dental Conditions:
    - Root Canal Therapy: A procedure to save a tooth when the nerve inside is infected or decayed. The pulp, nerves, bacteria, and decay are removed, and the space is filled with special dental materials. This endodontic treatment is performed by dentists or endodontists.
    - Baby Bottle Tooth Decay: Occurs when sugary liquids cling to baby's teeth for extended periods
    - Tooth Sensitivity: Caused by worn enamel or receding gums
    - Gum Disease: Infection of tissues surrounding teeth
    - Cavities: Tooth decay caused by bacteria and acid

    Prevention Tips:
    - Brush teeth twice daily with fluoride toothpaste
    - Floss daily to remove plaque between teeth
    - Regular dental checkups every 6 months
    - Limit sugary and acidic foods and drinks
    - Use mouthwash to kill bacteria

    Contact Information:
    - Phone: (425) 775-5162
    - Address: 51 W Dayton Street, Suite 301, Edmonds, WA 98020
    - Hours: Monday - Thursday, 8:00 AM - 5:00 PM. Closed on weekends.
    - For appointments, please call the office directly.
    """

    docs = [Document(page_content=basic_dental_content, metadata={"source": "basic_knowledge"})]
    print(f"✅ Created {len(docs)} basic knowledge documents")

def get_topic_and_link(text: str) -> tuple[str, str]:
    """
    Analyzes text to find the most relevant topic and its corresponding link.
    """
    if not text:
        return "general", DEFAULT_LINK

    text_lower = text.lower()
    for topic, (keywords, link) in LINK_MAPPING.items():
        if any(keyword in text_lower for keyword in keywords):
            return topic, link
    return "general", DEFAULT_LINK

# Improved text splitting with better parameters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Increased chunk size for better context
    chunk_overlap=100,  # Increased overlap for better continuity
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]  # Better separators
)
print(f"📄 Processing {len(docs)} documents for splitting...")
splits = text_splitter.split_documents(docs)
print(f"🔄 Generated {len(splits)} document splits")

# Show sample splits with better formatting
print("\n📝 Sample splits:")
for i, split in enumerate(splits[:10]):  # Show first 10 splits
    content_preview = split.page_content[:150].replace('\n', ' ')
    print(f"Split {i:3d}: {content_preview}...")

if len(splits) > 10:
    print(f"... and {len(splits) - 10} more splits")

print(f"\n📊 Split Statistics:")
print(f"   • Total splits: {len(splits)}")
print(f"   • Average length: {sum(len(s.page_content) for s in splits) // len(splits) if splits else 0} chars")
print(f"   • Shortest split: {min(len(s.page_content) for s in splits) if splits else 0} chars")
print(f"   • Longest split: {max(len(s.page_content) for s in splits) if splits else 0} chars")

# Enhanced metadata assignment with topic-specific links
print("\n🏷️ Assigning metadata to splits...")
for i, doc in enumerate(splits):
    # Preserve original source and add chunk-specific metadata
    original_source = doc.metadata.get("source", "unknown")
    
    # Use the centralized function to get topic and link
    detected_topic, topic_link = get_topic_and_link(doc.page_content)

    doc.metadata.update({
        "chunk_id": f"chunk_{i}",
        "chunk_index": i,
        "original_source": original_source,
        "chunk_length": len(doc.page_content),
        "doc_type": "dental_knowledge",
        "topic": detected_topic,
        "topic_link": topic_link
    })

    # Debug: Show topic detection for first few splits
    if i < 10:
        print(f"🔍 Split {i}: {doc.page_content[:80]}...")
        print(f"   📋 Detected topic: {detected_topic}")
        print(f"   🔗 Assigned link: {topic_link}")
        print("   ---")

print(f"✅ Assigned metadata to {len(splits)} splits with topic-specific links")


def setup_pinecone_vector_store(documents):
    try:
        print(f"\n🔧 Setting up Pinecone vector store with {len(documents)} documents...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "dental-knowledge-v2"  # New version to force fresh data

        # Check existing indexes
        existing_indexes = [index.name for index in pc.list_indexes()]
        print(f"📋 Existing indexes: {existing_indexes}")

        if index_name in existing_indexes:
            print(f"🗑️ Deleting existing index: {index_name}")
            pc.delete_index(index_name)
            import time
            time.sleep(15)  # Increased wait time

        # Create fresh index
        print(f"🆕 Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=768,  # Google embeddings dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

        import time
        time.sleep(15)  # Wait for index to be ready

        # Connect to index
        index = pc.Index(index_name)
        print(f"📊 Index stats: {index.describe_index_stats()}")

        # Create vector store
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )

        # Add documents in batches for better reliability
        batch_size = 50
        total_docs = len(documents)

        print(f"📤 Adding {total_docs} documents in batches of {batch_size}...")

        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size

            print(f"   Batch {batch_num}/{total_batches}: Adding {len(batch)} documents...")
            vector_store.add_documents(batch)

            # Small delay between batches
            if i + batch_size < total_docs:
                time.sleep(2)

        print("✅ All documents added successfully to Pinecone!")

        # Verify the upload
        final_stats = index.describe_index_stats()
        print(f"📈 Final index stats: {final_stats}")

        return vector_store

    except Exception as e:
        print(f"❌ Pinecone setup error: {str(e)}")
        print("🔄 This might be due to API limits or network issues")
        raise e


vectorstore = setup_pinecone_vector_store(splits)

# Test retrieval immediately
print("Testing retrieval...")
test_docs = vectorstore.similarity_search("flask", k=2)
print(f"Found {len(test_docs)} documents in test")
for doc in test_docs:
    print(f"Test doc: {doc.page_content[:50]}")

retriever = vectorstore.as_retriever()


def search_with_tavily(query, max_results=3):
    """
    Search the web using Tavily API for dental-related queries
    """
    if not tavily_client:
        print("Tavily client not initialized - missing API key")
        return None

    try:
        # Try multiple search strategies
        search_queries = [
            f"{query} dental health",  # Simple dental context
            f"baby bottle tooth decay dental" if "baby bottle" in query.lower() else f"{query} dentist",  # Specific terms
            query  # Original query as fallback
        ]

        for search_query in search_queries:
            print(f"Trying Tavily search with query: '{search_query}'")

            # First try with restricted domains
            try:
                search_results = tavily_client.search(
                    query=search_query,
                    search_depth="basic",
                    max_results=max_results,
                    include_domains=["https://www.edmondsbaydental.com/", "mayoclinic.org", "ada.org", "healthline.com", "mouthhealthy.org"]
                )

                if search_results and 'results' in search_results and len(search_results['results']) > 0:
                    print(f"Found {len(search_results['results'])} results with restricted domains")
                    break

            except Exception as domain_error:
                print(f"Domain-restricted search failed: {domain_error}")

            # If restricted search fails, try without domain restrictions
            try:
                search_results = tavily_client.search(
                    query=search_query,
                    search_depth="basic",
                    max_results=max_results
                )

                if search_results and 'results' in search_results and len(search_results['results']) > 0:
                    print(f"Found {len(search_results['results'])} results without domain restrictions")
                    break

            except Exception as open_error:
                print(f"Open search failed: {open_error}")
                continue

        if not search_results or 'results' not in search_results or len(search_results['results']) == 0:
            print("No search results found from Tavily")
            return None

        # Format the results for the LLM
        formatted_results = []
        for i, result in enumerate(search_results['results']):
            print(f"Result {i+1}: {result.get('title', 'No title')[:50]}...")
            formatted_result = f"Title: {result.get('title', 'N/A')}\n"
            formatted_result += f"Content: {result.get('content', 'N/A')}\n"
            formatted_result += f"URL: {result.get('url', 'N/A')}\n"
            formatted_results.append(formatted_result)

        final_results = "\n\n".join(formatted_results)
        print(f"Formatted search results length: {len(final_results)} characters")
        return final_results

    except Exception as e:
        print(f"Tavily search error: {str(e)}")
        return None

def generate_response_from_search(query, search_content):
    """
    Generate a response using search results. The link is added in post-processing.
    """
    # Get topic-specific link
    _, related_link = get_topic_and_link(query)

    search_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional dental assistant. "
         "Your task is to answer the user's question by summarizing the provided web search results from trusted sources. "
         "Follow this format:\n\n"
         "[Provide a brief, one-sentence summary formate based on the search results.]\n\n"
         "• [Create a concise bullet point from the search results.]\n"
         "• [Create another concise bullet point.]\n"
         "• [Create a final concise bullet point.]\n\n"
         "IMPORTANT: Do NOT attribute this information to Dr. Meenakshi Tomar, as it comes from external web sources. Do NOT include a 'Read more here' link. "
         "Base your entire answer ONLY on the text provided in the 'Search Results' section below.\n\n"
         "Search Results:\n{search_content}"
        ),
        ("human", "{query}")
    ])

    search_chain = search_prompt | llm

    try:
        response_llm = search_chain.invoke({
            "query": query,
            "search_content": search_content
        })
        
        raw_response_content = response_llm.content.strip()
        
        # Reliably add the link
        final_response = finalize_response_with_link(raw_response_content, related_link)
        print(f"Final formatted Tavily response: {final_response}")
        return final_response

    except Exception as e:
        print(f"Error generating response from search: {str(e)}")
        return ("• I apologize, but I'm having trouble accessing information right now\n"
                "• Please try again later or contact our office directly\n"
                "• Dr. Meenakshi Tomar's team is available to help with your dental questions")


def finalize_response_with_link(raw_text: str, link: str) -> str:
    """
    Takes raw text from the LLM and reliably appends a formatted link.
    """
    if not raw_text or not link:
        return raw_text or ""

    # Clean up any accidental link generation by the LLM
    cleaned_text = raw_text.strip()
    if "<a href" in cleaned_text:
        # A simple way to remove a potentially malformed link
        cleaned_text = cleaned_text.split("<a href")[0].strip()

    # Ensure there's a newline before the link
    if not cleaned_text.endswith("\n\n"):
        if cleaned_text.endswith("\n"):
            cleaned_text += "\n"
        else:
            cleaned_text += "\n\n"

    return f"{cleaned_text}<a href='{link}' target='_blank'>Read more here</a>"


contextualize_q_system_prompt = (
    "Given a chat history and a user question that might reference it, "
    "create a standalone question that can be understood without the chat history. "
    "Do NOT answer the question; just reformulate it if needed, otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


def is_medical_query(query):
    """
    Determine if query is medical/dental related or general information
    """
    query_lower = query.lower()

    # Medical/Dental keywords
    medical_keywords = [
        "pain", "tooth", "teeth", "dental", "cavity", "filling", "root canal",
        "whitening", "implant", "crown", "braces", "gum", "bleeding", "infection",
        "treatment", "procedure", "surgery", "extraction", "cleaning", "checkup",
        "orthodontic", "periodontal", "endodontic", "cosmetic", "veneer",
        "denture", "bridge", "fluoride", "plaque", "tartar", "gingivitis",
        "periodontitis", "abscess", "sensitivity", "decay", "enamel",
        "what is", "how to treat", "symptoms", "causes", "prevention",
        "baby bottle", "pediatric", "children teeth", "wisdom tooth"
    ]

    # General information keywords
    general_keywords = [
        "location", "address", "directions", "where", "office", "contact",
        "phone", "hours", "hour", "timing", "appointment", "schedule", "open",
        "closed", "working", "availability", "book", "reservation", "appoint",
        "insurance", "payment", "cost", "price", "about practice",
        "staff", "team", "facility"
    ]

    # Check for medical keywords
    medical_match = any(keyword in query_lower for keyword in medical_keywords)

    # Check for general keywords
    general_match = any(keyword in query_lower for keyword in general_keywords)

    # If both match, prioritize medical
    if medical_match:
        return True
    elif general_match:
        return False
    else:
        # Default to medical for ambiguous queries
        return True

def create_dynamic_system_prompt(query=""):
    """
    Create a clearer, more direct system prompt.
    The link is handled in post-processing to ensure it's always present.
    """
    is_medical = is_medical_query(query)

    if is_medical:
        # Medical response prompt (link is removed, will be added later)
        return (
            "You are a professional dental assistant for Dr. Meenakshi Tomar. "
            "Your task is to answer the user's question based on the provided context.\n\n"
            "CRITICAL RULE: If the user is asking for a recommendation, advice, suggestion, or opinion (e.g., using words like 'recommend', 'advise', 'suggest', 'what should I do', 'opinion on', 'thoughts on'), you MUST start your response with an attribution like 'Dr. Meenakshi Tomar recommends...' or 'For this, Dr. Meenakshi Tomar suggests...'.\n\n"
            "For general definitions (e.g., 'what is a cavity?'), do not use the doctor's name. Be natural.\n\n"
            "Follow this format:\n"
            "[Provide a helpful summary from the context. Follow the CRITICAL RULE about attribution.]\n\n"
            "• [Create a concise bullet point from the context.]\n"
            "• [Create another concise bullet point from the context.]\n"
            "• [Create a final concise bullet point from the context.]\n\n"
            "Base your entire answer ONLY on the text provided in the 'Context' section below.\n\n"
            "Context:\n"
            "{context}"
        )
    else:
        # General response prompt (link is also handled in post-processing)
        return (
            "You are a helpful dental assistant for Dr. Meenakshi Tomar's practice. "
            "Answer the user's question about the practice (e.g., services, general info) based on the provided context. "
            "Provide a direct, friendly answer in a few sentences, followed by bullet points if helpful.\n\n"
            "IMPORTANT: Do NOT use 'According to Dr. Meenakshi Tomar' for general questions. "
            "Do NOT include a 'Read more here' link in your response.\n\n"
            "Context:\n"
            "{context}"
        )

# Default system prompt for initialization
system_prompt = create_dynamic_system_prompt("")

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def get_link_from_retrieved_docs(retrieved_docs, query=""):
    """
    Determines the best link from retrieved documents and the original query.
    It prioritizes the query, then analyzes document metadata and content.
    """
    # 1. First, try to get a specific link from the user's query.
    query_topic, query_link = get_topic_and_link(query)
    if query_topic != "general":
        print(f"✅ Link found from query ('{query_topic}'): {query_link}")
        return query_link

    # 2. If query is general, check the metadata of the retrieved documents.
    if retrieved_docs:
        for doc in retrieved_docs:
            # The 'topic' and 'topic_link' are pre-assigned during document processing.
            doc_topic = doc.metadata.get("topic")
            if doc_topic and doc_topic != "general":
                doc_link = doc.metadata.get("topic_link", DEFAULT_LINK)
                print(f"✅ Link found from document metadata ('{doc_topic}'): {doc_link}")
                return doc_link

    # 3. As a fallback, analyze the combined content of the documents.
    if retrieved_docs:
        combined_content = " ".join([doc.page_content for doc in retrieved_docs])
        content_topic, content_link = get_topic_and_link(combined_content)
        if content_topic != "general":
            print(f"✅ Link found from combined document content ('{content_topic}'): {content_link}")
            return content_link

    # 4. If no specific link is found, use the default link.
    print(f"⚠️ No specific link found in query or docs, using default: {DEFAULT_LINK}")
    return DEFAULT_LINK

def get_static_response(query: str):
    """
    Handles static queries like contact, location, and hours to prevent hallucination.
    Returns a formatted string response if a keyword is matched, otherwise None.
    """
    query_lower = query.lower().strip()

    # Keywords for different categories
    greeting_keywords = ["hi", "hello", "hey"]
    contact_keywords = ["contact", "phone", "number", "call"]
    location_keywords = ["location", "address", "where", "directions", "map"]
    hours_keywords = ["hours", "hour", "timing", "open", "closed", "working", "work hour"]
    appointment_keywords = ["appointment", "schedule", "book", "appoint"]

    # Handle greetings with an exact match
    if query_lower in greeting_keywords:
        return "I am a Dental Expert. How can I help you with your dental health today?"

    # Check for matches and return hardcoded, accurate information
    if any(keyword in query_lower for keyword in contact_keywords):
        _, link = get_topic_and_link("contact")
        return (
            "You can contact Dr. Meenakshi Tomar's office by phone.\n\n"
            "• Phone: (425) 775-5162\n"
            "• Our team is ready to assist you with your questions.\n\n"
            f"<a href='{link}' target='_blank'>Contact Us</a>"
        )

    if any(keyword in query_lower for keyword in location_keywords):
        _, link = get_topic_and_link("location")
        return (
            "Dr. Meenakshi Tomar's office is located at:\n\n"
            "• Address: 51 W Dayton Street, Suite 301, Edmonds, WA 98020\n"
            "• We look forward to seeing you!\n\n"
            f"<a href='{link}' target='_blank'>Get Directions</a>"
        )

    if any(keyword in query_lower for keyword in hours_keywords):
        _, link = get_topic_and_link("practice")
        return (
            "Our office hours are as follows:\n\n"
            "• Monday & Tuesday: 7:00 AM - 6:00 PM\n"
            "• Thursday: 7:00 AM - 6:00 PM\n"
            "• Wednesday, Friday, Saturday & Sunday: Closed\n"
            "• Please call us at (425) 775-5162 to confirm availability.\n\n"
            f"<a href='{link}' target='_blank'>About Our Practice</a>"
        )

    if any(keyword in query_lower for keyword in appointment_keywords):
        # Appointments don't need a link, just the phone number.
        return (
            "To schedule an appointment, please call our office directly at **(425) 775-5162**.\n\n"
            "• Our team will help you find a suitable time.\n"
        )

    return None # No match found

def get_fallback_response(query: str) -> str:
    """
    Generates a context-aware fallback response when no information is found.
    """
    query_lower = query.lower()
    
    # Check if the query is about a topic we've defined as out of scope
    if any(term in query_lower for term in OUT_OF_SCOPE_TOPICS):
        return (
            "For questions about fees, insurance, or specific pricing, it's best to contact our office directly.\n\n"
            "• Our staff can provide you with the most accurate and up-to-date information.\n"
            "• You can reach us at (425) 775-5162 during business hours.\n"
            "• I can help with questions about dental procedures and general practice information."
        )
    # Generic fallback
    else:
        return (
            "I couldn't find information on that topic.\n\n"
            "• Please try asking about dental procedures, treatments, or our practice.\n"
            "• For detailed dental advice, I recommend consulting with a dental professional.\n"
            "• You can contact Dr. Meenakshi Tomar's office for personalized care."
        )

def get_error_fallback_response() -> str:
    """
    Generates a fallback response for system errors during RAG chain execution.
    """
    return (
        " I'm having trouble processing your question right now.\n\n"
        "• Please try asking again.\n"
        "• You can also contact our office directly for assistance.\n"
        "• Our system is working to resolve this issue.\n\n"
        "<a href='https://www.edmondsbaydental.com/procedures/' target='_blank'>Read more here</a>"
    )

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('modern_chat.html')

@app.route("/old")
def old_chat():
    return render_template('chat.html')

@app.route("/test")
def test_endpoint():
    return jsonify({
        "status": "working",
        "message": "Backend is running fine!",
        "response": "**Definition & Treatment Provider:**\nThis is a test response to check if the backend is working.\n\nTreatment & Care Details:**\n• Backend connection is successful\n• API endpoints are responding\n• System is ready to handle requests\n\n**Read More:**\nBackend test completed successfully."
    })

@app.route("/stream", methods=["POST"])
def stream_chat():
    """
    Streaming endpoint for real-time response generation
    """
    print("🌊 Stream endpoint called")

    # Extract request data OUTSIDE the generator
    try:
        msg = request.form["msg"]
        print(f"🔍 Streaming query: {msg}")
    except Exception as e:
        print(f"❌ Error getting form data: {str(e)}")
        return jsonify({"error": "Missing message data"}), 400

    def generate_stream(message):
        try:
            # Preprocess query for better matching
            print(f"🔍 Processing query: '{message}'")

            # Send typing indicator start
            yield f"data: {json.dumps({'type': 'typing', 'status': 'start'})}\n\n"
            time.sleep(0.5)  # Small delay to show typing

            # First, check for static info like contact, location, hours to prevent hallucination
            static_response = get_static_response(message)
            if static_response:
                print(f"✅ Found static response for: '{message}'")
                yield f"data: {json.dumps({'type': 'response', 'content': static_response})}\n\n"
                yield f"data: {json.dumps({'type': 'typing', 'status': 'end'})}\n\n"
                return

            # Test direct similarity search with better parameters
            try:
                print(f"🔍 Searching for: '{message}'")
                direct_results = vectorstore.similarity_search(message, k=5)  # Increased k for better results
                print(f"📊 Direct search found {len(direct_results)} documents")

                # Debug: Show what documents were found
                for i, doc in enumerate(direct_results[:2]):
                    print(f"   Doc {i}: {doc.page_content[:100]}...")

            except Exception as e:
                print(f"❌ Vector DB direct search failed: {str(e)}")
                direct_results = []

            # Test retriever
            try:
                retrieved_docs = retriever.invoke(message)
                print(f"🔍 Retriever found {len(retrieved_docs)} documents")

                # Debug: Show retrieved content
                for i, doc in enumerate(retrieved_docs[:2]):
                    print(f"   Retrieved {i}: {doc.page_content[:100]}...")

            except Exception as e:
                print(f"❌ Retriever failed: {str(e)}")
                retrieved_docs = []

            # Check if we have relevant documents
            if not retrieved_docs or len(retrieved_docs) == 0:
                print("No relevant documents found in vector DB, trying Tavily search...")

                # Fallback to Tavily search
                search_results = search_with_tavily(message)
                if search_results:
                    print("✅ Tavily search successful")
                    response_text = generate_response_from_search(message, search_results)
                else:
                    print("❌ Tavily search failed")
                    response_text = get_fallback_response(message)
                # Send complete response
                yield f"data: {json.dumps({'type': 'response', 'content': response_text})}\n\n"
                yield f"data: {json.dumps({'type': 'typing', 'status': 'end'})}\n\n"
                return

            # Use vector DB results with conversational RAG chain
            try:
                # Debug: Show retrieved documents info FIRST
                print(f"🔍 Retrieved {len(retrieved_docs)} documents for query: '{message}'")

                # Get link from retrieved documents BEFORE generating response
                retrieved_link = get_link_from_retrieved_docs(retrieved_docs, message)
                print(f"🔗 Extracted link from documents: {retrieved_link}")

                input_data = {"input": message}
                # Generate unique session ID for each request to avoid cached responses
                import uuid
                session_id = str(uuid.uuid4())[:8]

                print("🤖 Generating streaming response using RAG chain...")

                # Create dynamic prompt for this specific query
                dynamic_system_prompt = create_dynamic_system_prompt(message)

                # Use simple prompt without chat history for streaming
                simple_prompt = ChatPromptTemplate.from_messages([
                    ("system", dynamic_system_prompt),
                    ("human", "{input}"),
                ])

                # Create simple chain for streaming
                simple_chain = simple_prompt | llm

                # Generate response with dynamic prompt
                response_llm = simple_chain.invoke({
                    "input": message,
                    "context": "\n\n".join([doc.page_content for doc in retrieved_docs])
                })

                # Extract raw content
                raw_response_content = response_llm.content if hasattr(response_llm, 'content') else str(response_llm)

                # Reliably add the link
                final_response = finalize_response_with_link(raw_response_content, retrieved_link)

                # Stream the response word by word
                words = final_response.split()
                current_text = ""

                for i, word in enumerate(words):
                    current_text += word + " "
                    yield f"data: {json.dumps({'type': 'partial', 'content': current_text.strip()})}\n\n"
                    time.sleep(0.05)  # Small delay between words

                # Send final complete response
                yield f"data: {json.dumps({'type': 'response', 'content': final_response})}\n\n"
                yield f"data: {json.dumps({'type': 'typing', 'status': 'end'})}\n\n"

                print(f"✅ Streaming response completed with document-specific link: {retrieved_link}")

            except Exception as e:
                print(f"❌ RAG chain failed: {str(e)}")
                # As a fallback, check if it was a static query that was missed
                static_fallback = get_static_response(message)
                if static_fallback:
                    print(f"✅ RAG failed, but found a static response as fallback.")
                    fallback_response = static_fallback
                else:
                    fallback_response = get_error_fallback_response()
                
                yield f"data: {json.dumps({'type': 'response', 'content': fallback_response})}\n\n"
                yield f"data: {json.dumps({'type': 'typing', 'status': 'end'})}\n\n"

        except Exception as e:
            print(f"❌ Streaming error: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Connection error occurred'})}\n\n"
            yield f"data: {json.dumps({'type': 'typing', 'status': 'end'})}\n\n"

    return Response(generate_stream(msg), mimetype='text/plain', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*'
    })

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        print("📨 Received request to /get endpoint")

        # Check if request has form data
        if not request.form:
            print("❌ No form data in request")
            return jsonify({"error": "No form data"}), 400

        if "msg" not in request.form:
            print("❌ No 'msg' field in form data")
            return jsonify({"error": "Missing 'msg' field"}), 400

        msg = request.form["msg"]
        print(f"🔍 User query received: '{msg}'")

        # First, check for static info like contact, location, hours to prevent hallucination
        static_response = get_static_response(msg)
        if static_response:
            print(f"✅ Found static response for: '{msg}'")
            return jsonify({"response": static_response})

        # Test direct similarity search with error handling
        try:
            direct_results = vectorstore.similarity_search(msg, k=3)
            print(f"📊 Direct search found {len(direct_results)} documents")
        except Exception as e:
            print(f"❌ Vector DB direct search failed: {str(e)}")
            direct_results = []

        # Test retriever with error handling
        try:
            retrieved_docs = retriever.invoke(msg)
            print(f"🔍 Retriever found {len(retrieved_docs)} documents")
        except Exception as e:
            print(f"❌ Retriever failed: {str(e)}")
            retrieved_docs = []

    except Exception as e:
        print(f"❌ Error processing request: {str(e)}")
        return jsonify({"response": "Definition & Treatment Provider:\nI'm experiencing technical difficulties right now.\n\nTreatment & Care Details:\n• Please try refreshing the page and asking again\n• If the problem persists, contact Dr. Meenakshi Tomar's office directly\n• Our technical team is working to resolve this issue\n\nRead More:\nFor immediate assistance, visit: https://www.edmondsbaydental.com/procedures/"})

    # Check if we have relevant documents in vector DB
    if not retrieved_docs or len(retrieved_docs) == 0:
        print("No relevant documents found in vector DB, trying Tavily search...")

        # Fallback to Tavily search
        search_results = search_with_tavily(msg)

        if search_results:
            print("Found results from Tavily search")
            response_text = generate_response_from_search(msg, search_results)
            return jsonify({"response": response_text})
        else:
            print("No results from Tavily search either")
            fallback_response = get_fallback_response(msg)
            return jsonify({"response": fallback_response})

    # Check relevance score of retrieved documents
    # If the similarity score is too low, also try Tavily search
    low_relevance_threshold = 0.3  # Adjust this threshold as needed

    if hasattr(retrieved_docs[0], 'metadata') and 'score' in retrieved_docs[0].metadata:
        max_score = max([doc.metadata.get('score', 0) for doc in retrieved_docs])
        if max_score < low_relevance_threshold:
            print(f"Low relevance score ({max_score}), trying Tavily search...")
            search_results = search_with_tavily(msg)
            if search_results:
                response_text = generate_response_from_search(msg, search_results)
                return jsonify({"response": response_text})

    # Use vector DB results with conversational RAG chain
    try:
        # Debug: Show retrieved documents info FIRST
        print(f"🔍 Retrieved {len(retrieved_docs)} documents for query: '{msg}'")
        for i, doc in enumerate(retrieved_docs):
            print(f"📄 Doc {i}: {doc.page_content[:200]}...")
            if hasattr(doc, 'metadata'):
                print(f"📋 Metadata: {doc.metadata}")
                # Check if content matches the assigned link
                content_lower = doc.page_content.lower()
                assigned_link = doc.metadata.get('topic_link', 'No link')
                print(f"🔍 Content analysis:")
                print(f"   - Contains 'whitening': {'whitening' in content_lower}")
                print(f"   - Contains 'root canal': {'root canal' in content_lower}")
                print(f"   - Contains 'implant': {'implant' in content_lower}")
                print(f"   - Assigned link: {assigned_link}")
                print("---")

        # Get link from retrieved documents BEFORE generating response
        retrieved_link = get_link_from_retrieved_docs(retrieved_docs, msg)
        print(f"🔗 Extracted link from documents: {retrieved_link}")

        input_data = {"input": msg}
        # Generate unique session ID for each request to avoid cached responses
        import uuid
        session_id = str(uuid.uuid4())[:8]

        print("🤖 Generating response using RAG chain...")

        # Create dynamic prompt for this specific query
        dynamic_system_prompt = create_dynamic_system_prompt(msg)

        # Use simple prompt without chat history
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", dynamic_system_prompt),
            ("human", "{input}"),
        ])

        # Create simple chain
        simple_chain = simple_prompt | llm

        # Generate response with dynamic prompt
        response_llm = simple_chain.invoke({
            "input": msg,
            "context": "\n\n".join([doc.page_content for doc in retrieved_docs])
        })

        # Extract raw content
        raw_response_content = response_llm.content if hasattr(response_llm, 'content') else str(response_llm)

        # Reliably add the link
        final_response = finalize_response_with_link(raw_response_content, retrieved_link)

        print(f"✅ RAG response generated with document-specific link: {retrieved_link}")
        return jsonify({"response": final_response})

    except Exception as e:
        print(f"❌ RAG chain failed: {str(e)}")
        # As a fallback, check if it was a static query that was missed
        static_fallback = get_static_response(msg)
        if static_fallback:
            print(f"✅ RAG failed, but found a static response as fallback.")
            fallback_response = static_fallback
        else:
            fallback_response = get_error_fallback_response()
        return jsonify({"response": fallback_response})


if __name__ == '__main__':
    # Use debug=False to prevent double execution
    # Set use_reloader=False to prevent auto-restart on file changes
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)