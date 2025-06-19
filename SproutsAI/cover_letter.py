import gradio as gr
from openai import OpenAI
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_key_info_from_job(job_description):
    """Extract key information from job description using regex and basic parsing"""

    # Common patterns for company names (usually at the beginning or after "at")
    company_patterns = [
        r"(?:Company|Organization):\s*([^\n]+)",
        r"(?:at|@)\s+([A-Z][a-zA-Z\s&,.]+?)(?:\s+is|\s+seeks|\s+looking|\n|$)",
        r"^([A-Z][a-zA-Z\s&,.]+?)(?:\s+is|\s+seeks|\s+looking)",
    ]

    company_name = "the company"  # default
    for pattern in company_patterns:
        match = re.search(pattern, job_description, re.MULTILINE | re.IGNORECASE)
        if match:
            company_name = match.group(1).strip()
            break

    # Extract job title patterns
    title_patterns = [
        r"(?:Position|Role|Job Title):\s*([^\n]+)",
        r"(?:Job Title|Title):\s*([^\n]+)",
        r"(?:We are looking for|Seeking|Hiring)\s+(?:a|an)\s+([^\n,.]+)",
    ]

    job_title = "this position"  # default
    for pattern in title_patterns:
        match = re.search(pattern, job_description, re.MULTILINE | re.IGNORECASE)
        if match:
            job_title = match.group(1).strip()
            break

    return company_name, job_title


def extract_key_info_from_resume(resume):
    """Extract key information from resume"""

    # Extract name (usually first line or after "Name:")
    name_patterns = [
        r"(?:Name|Full Name):\s*([^\n]+)",
        r"^([A-Z][a-z]+\s+[A-Z][a-z]+)",  # First line with two capitalized words
    ]

    name = "I"  # default
    lines = resume.split("\n")
    if lines:
        # Try first non-empty line as name
        for line in lines[:3]:  # Check first 3 lines
            if line.strip() and len(line.strip()) > 2:
                words = line.strip().split()
                if len(words) >= 2 and all(word[0].isupper() for word in words[:2]):
                    name = line.strip()
                    break

    # For pattern matching
    for pattern in name_patterns:
        match = re.search(pattern, resume, re.MULTILINE)
        if match:
            name = match.group(1).strip()
            break

    return name


def generate_cover_letter(job_description, resume):
    """Generate a personalized cover letter using OpenAI"""

    if not job_description.strip() or not resume.strip():
        return "Please provide both job description and resume."

    if not client.api_key:
        return (
            "OpenAI API key not found. Please set your OPENAI_API_KEY "
            "environment variable."
        )

    try:
        # Extract key information
        company_name, job_title = extract_key_info_from_job(job_description)
        applicant_name = extract_key_info_from_resume(resume)

        # Create a comprehensive prompt for the cover letter
        prompt = f"""
        You are a professional cover letter writer. Create a compelling, personalized cover letter based on the following information:

        JOB DESCRIPTION:
        {job_description}

        APPLICANT'S RESUME:
        {resume}

        EXTRACTED INFO:
        - Company: {company_name}
        - Position: {job_title}
        - Applicant: {applicant_name}

        Please write a professional cover letter that:
        1. Addresses the hiring manager professionally
        2. Shows enthusiasm for the specific role and company
        3. Highlights relevant experiences from the resume that match the job requirements
        4. Demonstrates knowledge of the company/role from the job description
        5. Includes a strong closing paragraph with a call to action
        6. Is formatted professionally with proper business letter structure
        7. Is approximately 250-400 words long

        Format the letter with:
        - Date
        - Recipient address placeholder
        - Professional salutation
        - 3-4 well-structured paragraphs
        - Professional closing
        - Signature line

        Make it personalized and compelling, not generic.
        """

        # Call OpenAI API using new interface
        system_msg = (
            "You are a professional cover letter writer with "
            "expertise in creating compelling, personalized "
            "cover letters that highlight the candidate's "
            "strengths and match them to job requirements."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.7,
        )

        cover_letter = response.choices[0].message.content.strip()
        return cover_letter

    except Exception as e:
        return f"Error generating cover letter: {str(e)}"


def create_cover_letter_interface():
    """Create the main Gradio interface"""

    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .description {
        text-align: center;
        font-size: 1.1em;
        color: #666;
        margin-bottom: 2em;
    }
    """

    with gr.Blocks(
        css=css, title="Cover Letter Generator", theme=gr.themes.Soft()
    ) as demo:

        # Header
        header_html = '<h1 class="main-header">🎯 AI Cover Letter Generator</h1>'
        desc_html = (
            '<p class="description">Create personalized cover '
            "letters from LinkedIn job descriptions and your resume</p>"
        )

        gr.HTML(header_html)
        gr.HTML(desc_html)

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📝 Input Information")

                job_placeholder = (
                    "Paste the job description from LinkedIn "
                    "here...\n\nInclude details like:\n- Company "
                    "name\n- Job title\n- Requirements\n- "
                    "Responsibilities\n- Company information"
                )

                job_input = gr.Textbox(
                    label="LinkedIn Job Description",
                    placeholder=job_placeholder,
                    lines=12,
                    max_lines=15,
                )

                resume_placeholder = (
                    "Paste your resume here...\n\nInclude:"
                    "\n- Your name\n- Work experience\n- "
                    "Skills\n- Education\n- Achievements"
                )

                resume_input = gr.Textbox(
                    label="Your Resume",
                    placeholder=resume_placeholder,
                    lines=12,
                    max_lines=15,
                )

                generate_btn = gr.Button(
                    "🚀 Generate Cover Letter", variant="primary", size="lg"
                )

            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 📄 Generated Cover Letter")

                output_placeholder = (
                    "Your generated cover letter will " "appear here..."
                )

                cover_letter_output = gr.Textbox(
                    label="Your Personalized Cover Letter",
                    placeholder=output_placeholder,
                    lines=25,
                    max_lines=30,
                    show_copy_button=True,
                )

        # Examples section
        gr.Markdown("---")
        gr.Markdown("### 💡 Tips for Best Results")

        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                **Job Description Tips:**
                - Include the complete job posting
                - Make sure company name is clearly mentioned
                - Include job title and requirements
                - Add any company culture information
                """
                )

            with gr.Column():
                gr.Markdown(
                    """
                **Resume Tips:**
                - Start with your full name
                - Include relevant work experience
                - Highlight skills matching the job
                - Add education and certifications
                """
                )

        # Example data
        gr.Markdown("### 📋 Example Data")

        example_job = """Software Engineer - Frontend
        
Company: TechCorp Solutions
Location: San Francisco, CA

About TechCorp Solutions:
TechCorp Solutions is a leading technology company specializing in innovative web applications and digital solutions for Fortune 500 companies.

Job Description:
We are seeking a talented Frontend Software Engineer to join our dynamic development team. The successful candidate will be responsible for creating responsive, user-friendly web applications using modern JavaScript frameworks.

Requirements:
- Bachelor's degree in Computer Science or related field
- 2+ years of experience with React.js and modern JavaScript
- Proficiency in HTML5, CSS3, and responsive design
- Experience with Git version control
- Strong problem-solving skills and attention to detail
- Excellent communication and teamwork abilities

Responsibilities:
- Develop and maintain frontend applications using React.js
- Collaborate with UX/UI designers to implement pixel-perfect designs
- Optimize applications for maximum speed and scalability
- Write clean, maintainable, and well-documented code
- Participate in code reviews and agile development processes"""

        example_resume = """John Smith
Frontend Developer
Email: john.smith@email.com | Phone: (555) 123-4567 | LinkedIn: linkedin.com/in/johnsmith

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley | 2020

EXPERIENCE
Frontend Developer
WebDev Inc. | 2021 - Present
- Developed responsive web applications using React.js and modern JavaScript (ES6+)
- Collaborated with design team to implement user-friendly interfaces
- Optimized application performance, reducing load times by 40%
- Worked with Git for version control and participated in agile development

Junior Web Developer
StartupXYZ | 2020 - 2021
- Built interactive websites using HTML5, CSS3, and JavaScript
- Implemented responsive designs for mobile and desktop platforms
- Worked closely with backend developers to integrate APIs

SKILLS
- Programming: JavaScript, React.js, HTML5, CSS3, Node.js
- Tools: Git, Webpack, npm, VS Code
- Design: Responsive Design, Bootstrap, Material-UI
- Other: Agile Development, RESTful APIs, Problem Solving

PROJECTS
- E-commerce Platform: Built a full-stack e-commerce site using React and Node.js
- Portfolio Website: Created responsive personal portfolio showcasing projects and skills"""

        with gr.Row():
            with gr.Column():
                example_job_btn = gr.Button("📄 Load Example Job Description")
                example_job_btn.click(fn=lambda: example_job, outputs=job_input)

            with gr.Column():
                example_resume_btn = gr.Button("👤 Load Example Resume")
                example_resume_btn.click(
                    fn=lambda: example_resume, outputs=resume_input
                )

        # Set up the main event handler
        generate_btn.click(
            fn=generate_cover_letter,
            inputs=[job_input, resume_input],
            outputs=cover_letter_output,
            show_progress=True,
        )

        # Footer
        footer_text = (
            "*💡 Tip: Make sure to review and customize the "
            "generated cover letter before sending it to potential "
            "employers.*"
        )

        gr.Markdown("---")
        gr.Markdown(footer_text)

    return demo


if __name__ == "__main__":
    demo = create_cover_letter_interface()
    demo.launch(server_name="0.0.0.0", share=False, debug=True)
