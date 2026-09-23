from inspect import cleandoc
import os
import time
import base64
import fitz  # PyMuPDF
from typing import TypedDict
import pytesseract
from PIL import Image
import io
# LangChain / LangGraph Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from fpdf import FPDF

# Watchdog Imports
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv

load_dotenv() 
# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Ensure your GOOGLE_API_KEY is set in your environment variables

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", max_retries = 5)
SOURCE_DIR = "source_pdfs"
OUTPUT_DIR = "summary_notes"
# Create the folder if it doesn't exist
if not os.path.exists(SOURCE_DIR):
    os.makedirs(SOURCE_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class SummaryPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(100,116,139)
        self.cell(0, 8, "AUTOMATED DOCUMENT SUMMARY", border=0,new_x="LMARGIN", new_y="NEXT",align="R")
        self.set_draw_color(226,232,240)
        self.line(10,18,200,18)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148,163, 184)
        self.cell(0,10, f"page {self.page_no()}", align="C")


# ==========================================
# 2. DEFINE THE STATE
# ==========================================
class AgentState(TypedDict):
    file_path: str

    summary_text: str
    needs_revision: bool

# ==========================================
# 3. DEFINE THE NODES (Agent Actions)
# ==========================================

# If you are on Windows, point pytesseract to your installation:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def read_and_summarize(state: AgentState):
    print(f"Reading and Summarizing: {state['file_path']}")
    
    doc = fitz.open(state["file_path"])
    pdf_text = "\n".join([str(page.get_text()).strip() for page in doc]).strip()
    
    # 1. Set the BASE instruction (Do NOT include the pdf_text here yet)
    if state.get("needs_revision"):
        print("--> Agent was sent back for revision. Adjusting prompt...")
        base_instruction = "Your previous response was rejected because it missed the exact header. You MUST start your response with EXACTLY '### Executive Summary' with no extra bolding or characters."
    else:
        base_instruction = (
            "Provide a comprehensive and highly detailed summary of the following document. "
            "Break the information down into logical paragraphs and use bullet points for key details. "
            "Capture all main themes, arguments, and significant data. "
            "You MUST start your very first line with the exact header '### Executive Summary'."
        )
    # 2. DECISION LOGIC: Text vs Vision
    if len(pdf_text) > 50:
        print("--> Detected text-based PDF. Using fast text extraction.")
        #prompt = f"{base_instruction}\n\nDocument Text:\n{pdf_text}"
        #response = llm.invoke(prompt)
        final_text_content=pdf_text
        
    else:
        print("--> Detected scanned/image-based PDF. Booting up Vision mode...")
        full_instruction = f"{base_instruction} Read the attached scanned images to write the summary."
        ocr_pages = []
        #message_content: list = [{"type": "text", "text": full_instruction}]
        
        for page in doc[:3]:
            pix = page.get_pixmap(dpi=150)
            #img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            img=Image.open(io.BytesIO(pix.tobytes("png")))
            text=pytesseract.image_to_string(img)
            #message_content.append({
            #    "type": "image_url",
            #    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            #})
            ocr_pages.append(text)
        final_text_content = "\n".join(ocr_pages).strip()
            
        #message = HumanMessage(content=message_content)
        #response = llm.invoke([message])


        
    # 3. Safely extract text to keep Pyright happy
    prompt = f"{base_instruction}\n\nDocument Text:\n{final_text_content}"
    response_chunks = llm.stream(prompt)
    

    print(f"\n---GEMINI DRAFT ---")
    final_text = ""
    for chunk in response_chunks:
        raw_content = chunk.content
        chunk_text = ""

        if isinstance(raw_content,list):
            for item in raw_content:
                if isinstance(item, dict) and "text" in item:
                    chunk_text +=item["text"]
        else:
            chunk = str(raw_content)


        print(chunk_text, end="", flush=True)
        final_text += chunk_text
    print(f"\n----------------")

    '''if isinstance(raw_content, list) and len(raw_content) > 0:
        first_item = raw_content[0]
        if isinstance(first_item, dict):
            final_text = str(first_item.get("text", ""))
        else:
            final_text = str(first_item)
    else:
        final_text = str(raw_content)
        
    print(f"\n--- GEMINI DRAFT ---\n{final_text}\n-------------------\n")'''
    
    return {"summary_text": final_text}

def check_quality(state: AgentState):
    print("Checking quality...")
    text = state.get("summary_text", "")
    
    if "### Executive Summary" not in text:
        print("FAILED! Missing Executive Summary. Sending back.")
        return {"needs_revision": True}
    
    print("PASSED! Quality check successful.")
    return {"needs_revision": False}

def build_pdf(state: AgentState):
    print(f"Building final summary for {state['file_path']}...")
    
    # Save the output to a text file right next to the PDF
    filename = os.path.basename(state["file_path"]).replace(".pdf", "_summary.pdf")
    
    output_path = os.path.join(OUTPUT_DIR, filename)

    print(f"Genarating formatted PDF at: {output_path}...")

    pdf = SummaryPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    lines = state["summary_text"].split("\n")
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            pdf.ln(3)
            continue

        if clean_line.startswith("###"):
            pdf.set_font("Helvetica","B",14)
            pdf.set_text_color(15, 23, 42)
            header_title=clean_line.replace("###", "").strip()
            pdf.cell(0, 10, header_title, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif clean_line.startswith("*") or clean_line.startswith("-"):
            pdf.set_font("Helvetica", size=14)
            pdf.set_text_color(52, 65, 85)

            bullet_text=clean_line.lstrip("*- ").replace("**", "").strip()
            pdf.multi_cell(0,6,f"- {bullet_text}", new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", size=14)
            pdf.set_text_color(51, 65,85)
            plain_text = clean_line.replace("**","").strip()
            pdf.multi_cell(0,6,plain_text, new_x="LMARGIN", new_y="NEXT")

    pdf.output(output_path)
    print(f"DONE! PDF summary stored in '{OUTPUT_DIR}")
    return state

# ==========================================
# 4. BUILD THE GRAPH (The workflow)
# ==========================================
def router(state: AgentState):
    if state["needs_revision"]:
        return "read_and_summarize"
    return "build_pdf"

workflow = StateGraph(AgentState)

workflow.add_node("read_and_summarize", read_and_summarize)
workflow.add_node("check_quality", check_quality)
workflow.add_node("build_pdf", build_pdf)

workflow.set_entry_point("read_and_summarize")
workflow.add_edge("read_and_summarize", "check_quality")
workflow.add_conditional_edges("check_quality", router)
workflow.add_edge("build_pdf", END)

app = workflow.compile()

# ==========================================
# 5. THE FOLDER WATCHER (n8n Local File Trigger)
# ==========================================
class PDFWatcher(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not str(event.src_path).lower().endswith(".pdf"):
            return
            
        print(f"\n--- New PDF Detected: {event.src_path} ---")
        time.sleep(1) 
        
        initial_state :AgentState= {
            "file_path": str(event.src_path), 

            "summary_text": '', 
            "needs_revision": False
        }
        app.invoke(initial_state)
        print(f"--- Finished processing! Watching for next file... ---\n")

if __name__ == "__main__":
    watcher = Observer()
    watcher.schedule(PDFWatcher(), path=SOURCE_DIR, recursive=False)
    watcher.start()
    print(f"👀 Watching '{SOURCE_DIR}' for new PDFs. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
    watcher.join()