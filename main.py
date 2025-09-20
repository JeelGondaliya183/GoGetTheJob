import streamlit as st
import PyPDF2
import io
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from typing import Dict, Any, Optional

st.set_page_config(
    page_title="Job Application AI Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_llm(model_name: str = "llama3.2"):
    """Initialize Ollama LLM with specified model"""
    try:
        return OllamaLLM(model=model_name)
    except Exception as e:
        st.error(f"Error initializing Ollama model '{model_name}': {str(e)}")
        st.error("Make sure Ollama is running and the model is installed.")
        return None

def test_llm_connection(llm, model_name: str) -> bool:
    """Test if the LLM is working properly"""
    try:
        response = llm.invoke("Hello, respond with 'OK' if you can hear me.")
        return True
    except Exception as e:
        st.error(f"Failed to connect to model '{model_name}': {str(e)}")
        st.error("Please ensure Ollama is running and the model is available.")
        return False

class JobDescriptionAnalyzer:
    """Agent for analyzing job descriptions and extracting key requirements"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["job_description"],
            template="""You are an expert job analysis assistant. Analyze the following job description and extract key information in a clear, structured format.

Job Description: {job_description}

Please provide a comprehensive analysis including:

1. **Key Skills Required:**
   - Technical skills
   - Soft skills
   - Tools and technologies

2. **Experience Level Required:**
   - Years of experience
   - Level (entry, mid, senior)

3. **Education Requirements:**
   - Degree requirements
   - Certifications

4. **Key Responsibilities:**
   - Main duties
   - Core functions

5. **Company Culture/Values:**
   - Mentioned values
   - Work environment

6. **ATS Keywords:**
   - Important keywords for resume optimization
   - Industry-specific terms

Format your response clearly with headers and bullet points for easy reading."""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def analyze(self, job_description: str) -> str:
        return self.chain.run(job_description=job_description)

class ResumeOptimizer:
    """Agent for optimizing resumes based on job requirements"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["existing_resume", "job_analysis", "additional_prompt"],
            template="""You are an expert resume writer. Create an optimized resume that better matches the job requirements while maintaining truthfulness and professionalism.

**Job Analysis:**
{job_analysis}

**Current Resume:**
{existing_resume}

**Additional Instructions:**
{additional_prompt}

**Your Task:**
Create an improved resume that:
1. Highlights relevant skills mentioned in the job description
2. Uses important keywords for ATS optimization
3. Emphasizes experiences that align with job requirements
4. Maintains professional formatting and complete truthfulness
5. Quantifies achievements with specific numbers/metrics where possible
6. Uses strong action verbs
7. Tailors the professional summary to the role

**Important Guidelines:**
- Never fabricate experience or skills
- Only emphasize existing qualifications that match the job
- Maintain the original structure but optimize content
- Use bullet points for achievements
- Keep it concise and relevant

Please provide the complete optimized resume in a professional format."""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def optimize(self, existing_resume: str, job_analysis: str, additional_prompt: str = "") -> str:
        return self.chain.run(
            existing_resume=existing_resume,
            job_analysis=job_analysis,
            additional_prompt=additional_prompt
        )

class CoverLetterGenerator:
    """Agent for generating cover letters based on job requirements"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["job_analysis", "resume", "existing_cover_letter", "additional_prompt"],
            template="""You are an expert cover letter writer. Create a compelling, personalized cover letter based on the job requirements and candidate's background.

**Job Analysis:**
{job_analysis}

**Candidate's Resume:**
{resume}

**Existing Cover Letter (for reference):**
{existing_cover_letter}

**Additional Instructions:**
{additional_prompt}

**Your Task:**
Create a professional cover letter that:
1. Addresses specific requirements from the job description
2. Highlights relevant experiences from the resume with concrete examples
3. Shows genuine enthusiasm for the role and company
4. Uses a professional yet engaging and personable tone
5. Includes specific achievements with quantifiable results
6. Is concise (3-4 paragraphs maximum)
7. Has a strong opening that grabs attention
8. Ends with a clear call to action

**Structure:**
- Opening: Hook + position you're applying for
- Body 1-2 paragraphs: Relevant experience + achievements + how they match job needs
- Closing: Enthusiasm + next steps

Please provide a complete, professional cover letter ready to send."""
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def generate(self, job_analysis: str, resume: str, existing_cover_letter: str = "", additional_prompt: str = "") -> str:
        return self.chain.run(
            job_analysis=job_analysis,
            resume=resume,
            existing_cover_letter=existing_cover_letter,
            additional_prompt=additional_prompt
        )

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def main():
    st.title("Job Application AI Assistant")
    st.markdown("### Transform your resume and cover letter with AI-powered optimization using Llama")
    
    with st.sidebar:
        st.header("🔧 Configuration")
        
        st.markdown("### Ollama Setup")
        st.info("""
        **Prerequisites:**
        1. Install Ollama from https://ollama.ai
        2. Run: `ollama pull llama3.2` (or your preferred model)
        3. Start Ollama service
        """)
        
        model_options = [
            "llama3.2", "llama3.1", "llama3", "llama2", 
            "mistral", "codellama", "phi3", "gemma2"
        ]
        
        selected_model = st.selectbox(
            "Select Ollama Model:",
            model_options,
            index=0,
            help="Choose the language model to use. Make sure it's installed in Ollama."
        )
        
        if st.button("Test Connection"):
            with st.spinner("Testing connection..."):
                test_llm = get_llm(selected_model)
                if test_llm and test_llm_connection(test_llm, selected_model):
                    st.success(f"Connected to {selected_model} successfully!")
                else:
                    st.error("Connection failed. Check Ollama setup.")
        
        st.markdown("---")
        st.markdown("### Model Status")
        if st.button("📋 List Available Models"):
            try:
                import subprocess
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.code(result.stdout, language="bash")
                else:
                    st.error("Failed to list models. Is Ollama running?")
            except FileNotFoundError:
                st.error("Ollama not found. Please install Ollama first.")
    
    llm = get_llm(selected_model)
    if not llm:
        st.error("Failed to initialize language model. Please check your Ollama setup.")
        st.stop()
    
    if not test_llm_connection(llm, selected_model):
        st.error("Cannot connect to the language model. Please ensure Ollama is running.")
        st.stop()
    
    try:
        job_analyzer = JobDescriptionAnalyzer(llm)
        resume_optimizer = ResumeOptimizer(llm)
        cover_letter_generator = CoverLetterGenerator(llm)
        st.success(f"AI agents initialized with {selected_model}")
    except Exception as e:
        st.error(f"Error initializing AI agents: {str(e)}")
        st.stop()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input Documents")
        
        st.subheader("Job Description")
        job_description = st.text_area(
            "Paste the job description here:",
            height=200,
            placeholder="Paste the complete job description...",
            help="Include the full job posting with requirements, responsibilities, and company information."
        )
        
        st.subheader("Current Resume")
        resume_option = st.radio("Choose resume input method:", ["Upload PDF", "Paste Text"])
        
        if resume_option == "Upload PDF":
            resume_file = st.file_uploader("Upload your resume (PDF)", type=['pdf'])
            existing_resume = ""
            if resume_file:
                existing_resume = extract_text_from_pdf(resume_file)
                if existing_resume:
                    st.success("Resume uploaded successfully!")
                    with st.expander("Preview extracted text"):
                        preview_text = existing_resume[:1000] + "..." if len(existing_resume) > 1000 else existing_resume
                        st.text(preview_text)
        else:
            existing_resume = st.text_area(
                "Paste your resume text here:",
                height=200,
                placeholder="Paste your current resume content...",
                help="Include all sections: contact info, summary, experience, education, skills, etc."
            )
        
        st.subheader("Current Cover Letter (Optional)")
        cover_letter_option = st.radio("Choose cover letter input method:", ["Upload PDF", "Paste Text", "None"])
        
        existing_cover_letter = ""
        if cover_letter_option == "Upload PDF":
            cover_letter_file = st.file_uploader("Upload your cover letter (PDF)", type=['pdf'])
            if cover_letter_file:
                existing_cover_letter = extract_text_from_pdf(cover_letter_file)
                if existing_cover_letter:
                    st.success("Cover letter uploaded successfully!")
        elif cover_letter_option == "Paste Text":
            existing_cover_letter = st.text_area(
                "Paste your cover letter text here:",
                height=150,
                placeholder="Paste your current cover letter content..."
            )
        
        st.subheader("Additional Instructions (Optional)")
        additional_prompt = st.text_area(
            "Any specific requirements or preferences:",
            height=100,
            placeholder="e.g., emphasize leadership experience, highlight technical skills, use formal tone, focus on specific achievements..."
        )
    
    with col2:
        st.header("🤖 AI Processing & Results")
        
        if st.button("Generate Optimized Documents", type="primary", use_container_width=True):
            if not job_description.strip():
                st.error("Please provide a job description to continue!")
                return
            if not existing_resume.strip():
                st.error("Please provide your current resume to continue!")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Analyze Job Description
                status_text.text("Analyzing job description...")
                progress_bar.progress(25)
                
                with st.spinner("Analyzing job requirements..."):
                    job_analysis = job_analyzer.analyze(job_description)
                
                # Step 2: Optimize Resume
                status_text.text("Optimizing resume...")
                progress_bar.progress(50)
                
                with st.spinner("Optimizing your resume..."):
                    optimized_resume = resume_optimizer.optimize(
                        existing_resume, job_analysis, additional_prompt
                    )
                
                # Step 3: Generate Cover Letter
                status_text.text("Generating cover letter...")
                progress_bar.progress(75)
                
                with st.spinner("Creating your cover letter..."):
                    new_cover_letter = cover_letter_generator.generate(
                        job_analysis, optimized_resume, existing_cover_letter, additional_prompt
                    )
                
                progress_bar.progress(100)
                status_text.text("All documents generated successfully!")
                
                st.success("Documents generated successfully!")
                
                tab1, tab2, tab3 = st.tabs(["Job Analysis", "Optimized Resume", "Cover Letter"])
                
                with tab1:
                    st.markdown("### Job Requirements Analysis")
                    st.markdown(job_analysis)
                
                with tab2:
                    st.markdown("### Optimized Resume")
                    st.markdown(optimized_resume)
                    st.download_button(
                        "Download Resume",
                        optimized_resume,
                        file_name="optimized_resume.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with tab3:
                    st.markdown("### Generated Cover Letter")
                    st.markdown(new_cover_letter)
                    st.download_button(
                        "Download Cover Letter",
                        new_cover_letter,
                        file_name="cover_letter.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please check your Ollama setup and try again.")
                st.info("Try restarting Ollama or switching to a different model.")

if __name__ == "__main__":
    main()
