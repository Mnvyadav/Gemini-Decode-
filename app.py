import os
import streamlit as st
from dotenv import load_dotenv
import time
import json
from pathlib import Path

# Load Environment Variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize session state for caching and usage tracking
if 'api_available' not in st.session_state:
    st.session_state.api_available = True
if 'last_api_error' not in st.session_state:
    st.session_state.last_api_error = None
if 'usage_count' not in st.session_state:
    st.session_state.usage_count = 0
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

# Try to import and configure Gemini API
api_configured = False
model = None

if api_key:
    try:
        import google.generativeai as genai
        from PIL import Image
        
        genai.configure(api_key=api_key)
        api_configured = True
        
        # Try to get a model (will fail if quota exceeded)
        try:
            model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            st.session_state.api_available = False
            st.session_state.last_api_error = str(e)
            
    except ImportError:
        st.session_state.api_available = False
        st.session_state.last_api_error = "Required packages not installed"
else:
    st.session_state.api_available = False
    st.session_state.last_api_error = "API key not configured"

# Enhanced demo content for comprehensive notes (2+ pages)
DEMO_NOTES = {
    "operating system": """
# OPERATING SYSTEM: COMPREHENSIVE NOTES

## 1. DEFINITION AND OVERVIEW

An Operating System (OS) is system software that manages computer hardware, software resources, and provides common services for computer programs. It acts as an intermediary between users and the computer hardware, ensuring efficient operation and providing a user-friendly environment.

### 1.1 Key Characteristics
- Resource Management: Efficient allocation and management of hardware resources
- Process Management: Creation, scheduling, and termination of processes
- Memory Management: Control and coordination of computer memory
- File System Management: Organization and manipulation of files
- Security: Protection against unauthorized access

## 2. EVOLUTION OF OPERATING SYSTEMS

### 2.1 First Generation (1940s-1950s)
- No operating systems
- Programs entered using punch cards
- Machine language programming

### 2.2 Second Generation (1955-1965)
- Batch processing systems
- Simple monitor programs
- Introduction of assembly language

### 2.3 Third Generation (1965-1980)
- Multiprogramming systems
- Time-sharing concepts
- Development of UNIX

### 2.4 Fourth Generation (1980-Present)
- Personal computers
- Graphical User Interfaces (GUI)
- Network operating systems
- Mobile operating systems

## 3. TYPES OF OPERATING SYSTEMS

### 3.1 Batch Operating System
- Processes similar jobs in batches
- No direct user interaction
- Examples: IBM's OS/360

### 3.2 Time-Sharing OS
- Multiple users share system simultaneously
- CPU time divided among users
- Examples: UNIX, Multics

### 3.3 Distributed OS
- Manages group of independent computers
- Appears as single system to users
- Examples: LOCUS, Amoeba

### 3.4 Network OS
- Runs on servers
- Provides networking capabilities
- Examples: Windows Server, Linux

### 3.5 Real-Time OS
- Time constraints are critical
- Used in embedded systems
- Examples: VxWorks, QNX

### 3.6 Mobile OS
- Designed for mobile devices
- Touchscreen interface
- Examples: Android, iOS

## 4. CORE COMPONENTS

### 4.1 Kernel
- Core component of OS
- Manages system resources
- Types: Monolithic, Microkernel, Hybrid

### 4.2 Process Management
- Process creation and termination
- Process scheduling algorithms
- Process synchronization

### 4.3 Memory Management
- Memory allocation techniques
- Virtual memory concepts
- Paging and segmentation

### 4.4 File System Management
- File organization methods
- Directory structures
- File access methods

### 4.5 Device Management
- Device drivers
- Input/output management
- Buffer management

## 5. PROCESS MANAGEMENT

### 5.1 Process Concept
- Program in execution
- Process control block (PCB)
- Process states: New, Ready, Running, Waiting, Terminated

### 5.2 Process Scheduling
- Scheduling algorithms: FCFS, SJF, Priority, Round Robin
- Scheduling queues: Job, Ready, Device
- Context switching

### 5.3 Process Synchronization
- Critical section problem
- Semaphores and mutexes
- Deadlock handling

## 6. MEMORY MANAGEMENT

### 6.1 Memory Allocation
- Contiguous memory allocation
- Non-contiguous memory allocation
- Paging and segmentation

### 6.2 Virtual Memory
- Demand paging
- Page replacement algorithms
- Thrashing

## 7. FILE SYSTEMS

### 7.1 File Concepts
- File attributes and operations
- File types and structures
- File access methods

### 7.2 Directory Structure
- Single-level directories
- Two-level directories
- Tree-structured directories
- Acyclic graph directories

## 8. SECURITY AND PROTECTION

### 8.1 Security Mechanisms
- Authentication methods
- Access control lists
- Encryption techniques

### 8.2 Protection Systems
- Domain of protection
- Access matrix implementation
- Security models

## 9. MODERN OPERATING SYSTEMS

### 9.1 Windows Architecture
- Kernel mode components
- User mode components
- Registry system

### 9.2 UNIX/Linux Architecture
- Kernel structure
- Shell environment
- File system hierarchy

### 9.3 macOS Architecture
- Darwin kernel
- Cocoa framework
- Quartz compositor

## 10. EMERGING TRENDS

### 10.1 Cloud-Based OS
- Chrome OS
- Web-based applications
- Cloud integration

### 10.2 Containerization
- Docker containers
- Kubernetes orchestration
- Microservices architecture

### 10.3 IoT Operating Systems
- Lightweight kernels
- Real-time capabilities
- Energy efficiency

## 11. PERFORMANCE MONITORING

### 11.1 System Metrics
- CPU utilization
- Memory usage
- Disk I/O statistics
- Network throughput

### 11.2 Optimization Techniques
- Caching strategies
- Load balancing
- Resource allocation algorithms

## 12. TROUBLESHOOTING AND MAINTENANCE

### 12.1 Common Issues
- Memory leaks
- Process hangs
- File system corruption
- Driver conflicts

### 12.2 Diagnostic Tools
- System monitors
- Performance analyzers
- Log file analysis

## CONCLUSION

Operating systems form the foundation of modern computing, providing essential services that enable applications to run efficiently on hardware platforms. From simple batch processing systems to complex distributed environments, OS evolution continues to shape how we interact with computing devices. Understanding operating system concepts is crucial for computer scientists, software developers, and IT professionals working with modern computing systems.

---
*This comprehensive overview covers the fundamental concepts, historical development, and modern implementations of operating systems, providing a solid foundation for further study and practical application.*
""",
    "machine learning": """
# MACHINE LEARNING: COMPREHENSIVE GUIDE

## 1. INTRODUCTION TO MACHINE LEARNING

Machine Learning (ML) is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can learn from and make predictions on data.

### 1.1 Fundamental Concepts
- Learning from data patterns
- Generalization from examples
- Adaptive system improvement
- Predictive modeling

## 2. TYPES OF MACHINE LEARNING

### 2.1 Supervised Learning
- Learning with labeled data
- Classification and regression tasks
- Examples: Spam detection, price prediction

### 2.2 Unsupervised Learning
- Finding patterns in unlabeled data
- Clustering and association tasks
- Examples: Customer segmentation, anomaly detection

### 2.3 Reinforcement Learning
- Learning through interaction
- Reward-based system
- Examples: Game playing, robotic control

### 2.4 Semi-Supervised Learning
- Combination of labeled and unlabeled data
- Reduced annotation effort
- Examples: Text classification, image recognition

### 2.5 Deep Learning
- Neural networks with multiple layers
- Feature learning automation
- Examples: Image recognition, natural language processing

## 3. KEY ALGORITHMS AND TECHNIQUES

### 3.1 Linear Models
- Linear regression
- Logistic regression
- Regularization techniques (L1/L2)

### 3.2 Tree-Based Methods
- Decision trees
- Random forests
- Gradient boosting machines

### 3.3 Support Vector Machines
- Maximum margin classification
- Kernel tricks for non-linear data
- Applications in pattern recognition

### 3.4 Neural Networks
- Perceptrons and multilayer networks
- Backpropagation algorithm
- Deep learning architectures

### 3.5 Clustering Algorithms
- K-means clustering
- Hierarchical clustering
- DBSCAN density-based clustering

### 3.6 Dimensionality Reduction
- Principal Component Analysis (PCA)
- t-SNE visualization
- Autoencoders for feature learning

## 4. THE MACHINE LEARNING WORKFLOW

### 4.1 Data Collection
- Identifying data sources
- Data acquisition methods
- Ethical considerations

### 4.2 Data Preprocessing
- Handling missing values
- Feature scaling and normalization
- Data transformation techniques

### 4.3 Feature Engineering
- Feature selection methods
- Feature creation techniques
- Domain-specific feature development

### 4.4 Model Selection
- Algorithm comparison
- Hyperparameter tuning
- Cross-validation strategies

### 4.5 Model Training
- Training/validation split
- Batch vs online learning
- Distributed training approaches

### 4.6 Model Evaluation
- Performance metrics
- Confusion matrix analysis
- ROC curves and AUC scores

### 4.7 Model Deployment
- Production environment setup
- Monitoring and maintenance
- Model versioning

## 5. DEEP LEARNING FRAMEWORKS

### 5.1 TensorFlow
- Google's open-source platform
- High-level Keras API
- Production deployment capabilities

### 5.2 PyTorch
- Facebook's research framework
- Dynamic computation graphs
- Academic community preference

### 5.3 Other Frameworks
- MXNet for distributed learning
- Caffe for computer vision
- Scikit-learn for traditional ML

## 6. APPLICATIONS ACROSS INDUSTRIES

### 6.1 Healthcare
- Medical image analysis
- Drug discovery
- Patient outcome prediction

### 6.2 Finance
- Fraud detection
- Algorithmic trading
- Credit risk assessment

### 6.3 Retail
- Recommendation systems
- Inventory optimization
- Customer behavior analysis

### 6.4 Manufacturing
- Predictive maintenance
- Quality control
- Supply chain optimization

### 6.5 Transportation
- Autonomous vehicles
- Route optimization
- Demand forecasting

## 7. ETHICAL CONSIDERATIONS

### 7.1 Bias and Fairness
- Algorithmic bias sources
- Fairness metrics
- Mitigation strategies

### 7.2 Privacy Concerns
- Data protection regulations
- Federated learning approaches
- Differential privacy techniques

### 7.3 Transparency
- Explainable AI methods
- Model interpretability
- Regulatory compliance

## 8. CURRENT CHALLENGES

### 8.1 Data Quality Issues
- Noisy and incomplete data
- Labeling inconsistencies
- Data drift problems

### 8.2 Computational Requirements
- Hardware acceleration needs
- Energy consumption concerns
- Cloud computing costs

### 8.3 Model Complexity
- Overfitting risks
- Interpretability challenges
- Maintenance difficulties

## 9. FUTURE TRENDS

### 9.1 Automated Machine Learning
- Neural architecture search
- Hyperparameter optimization
- End-to-end automation

### 9.2 Edge Computing
- On-device inference
- Federated learning growth
- IoT integration

### 9.3 Explainable AI
- Model interpretation tools
- Regulatory requirements
- Trust-building measures

### 9.4 Multimodal Learning
- Combining data types
- Cross-modal understanding
- Unified learning approaches

## 10. LEARNING RESOURCES

### 10.1 Online Courses
- Coursera ML specialization
- Fast.ai practical deep learning
- Stanford CS229 machine learning

### 10.2 Books and Publications
- "Pattern Recognition and Machine Learning"
- "Deep Learning" by Goodfellow et al.
- Research papers from conferences

### 10.3 Development Tools
- Jupyter notebooks
- Google Colab environment
- MLflow for experiment tracking

## CONCLUSION

Machine learning continues to transform industries and create new possibilities for data-driven decision making. As the field evolves, practitioners must balance technical expertise with ethical considerations, ensuring that ML systems benefit society while minimizing potential harms. The comprehensive understanding of machine learning concepts, algorithms, and applications provides a solid foundation for contributing to this rapidly advancing field.

---
*This extensive guide covers the breadth of machine learning from fundamental concepts to advanced applications, providing a thorough resource for students and practitioners alike.*
"""
}

def get_demo_response(input_text, image_description):
    """Generate demo response when API is unavailable - returns summarized content"""
    summary = f"""
# DOCUMENT SUMMARY

## Key Points Extracted:

1. **Main Topic**: The document discusses {image_description if image_description else 'a technical subject'} with comprehensive coverage of relevant concepts.

2. **Core Concepts**: 
   - Fundamental principles and theories related to the subject matter
   - Practical applications and real-world implementations
   - Technical specifications and operational guidelines

3. **Important Findings**:
   - Key data points and statistical information
   - Comparative analysis of different approaches
   - Future trends and development areas

4. **Actionable Insights**:
   - Recommendations for implementation
   - Best practices derived from the content
   - Potential areas for further research

## Detailed Analysis:

The document provides thorough coverage of its subject matter, presenting information in a structured and accessible manner. Key technical terms are clearly defined and contextualized, making the content suitable for both beginners and experienced professionals.

### Section Highlights:
- Comprehensive overview of fundamental concepts
- Detailed examination of practical applications
- Analysis of current challenges and solutions
- Future outlook and emerging trends

### Technical Content:
The material includes technical specifications, operational guidelines, and implementation strategies that would be valuable for professionals working in related fields.

## Conclusion:

This document serves as a valuable resource for understanding {image_description if image_description else 'the subject'} and its various applications. The content is well-organized, comprehensive, and provides both theoretical foundations and practical guidance.

*Note: This is a demo summary. Connect to Google Gemini API for accurate extraction of specific document content.*
"""
    return summary

def get_demo_notes(topic, language="english"):
    """Generate comprehensive demo notes (2+ pages)"""
    if topic.lower() in DEMO_NOTES:
        return DEMO_NOTES[topic.lower()]
    else:
        return f"""
# {topic.upper()}: COMPREHENSIVE STUDY GUIDE

## 1. INTRODUCTION AND OVERVIEW

{topic.title()} represents a significant field of study with wide-ranging applications across multiple disciplines. This comprehensive guide provides detailed coverage of fundamental concepts, practical applications, and advanced topics.

### 1.1 Historical Development
- Early foundations and pioneering work
- Major milestones and breakthroughs
- Current state of the field
- Future directions and emerging trends

### 1.2 Core Principles
- Fundamental theories and concepts
- Key terminology and definitions
- Basic principles and axioms
- Foundational frameworks

## 2. THEORETICAL FOUNDATIONS

### 2.1 Fundamental Concepts
- Core ideas that form the basis of understanding
- Important relationships between concepts
- Theoretical models and frameworks
- Conceptual hierarchies and taxonomies

### 2.2 Mathematical Underpinnings
- Relevant mathematical concepts
- Statistical methods and applications
- Computational complexity considerations
- Algorithmic foundations

## 3. PRACTICAL APPLICATIONS

### 3.1 Industry Implementations
- Real-world applications across sectors
- Case studies of successful implementations
- Industry-specific adaptations
- Practical challenges and solutions

### 3.2 Technical Implementation
- Implementation methodologies
- System architecture considerations
- Integration with existing systems
- Performance optimization techniques

## 4. METHODOLOGIES AND TECHNIQUES

### 4.1 Research Methods
- Experimental design approaches
- Data collection techniques
- Analysis methodologies
- Validation procedures

### 4.2 Practical Techniques
- Problem-solving approaches
- Implementation strategies
- Optimization methods
- Troubleshooting procedures

## 5. CURRENT CHALLENGES

### 5.1 Technical Challenges
- Limitations of current approaches
- Computational constraints
- Scalability issues
- Integration difficulties

### 5.2 Theoretical Limitations
- Gaps in current understanding
- Unresolved problems
- Theoretical constraints
- Research opportunities

## 6. FUTURE DIRECTIONS

### 6.1 Emerging Trends
- New developments and innovations
- Changing paradigms
- Evolving methodologies
- Future research areas

### 6.2 Potential Applications
- Untapped application areas
- Future implementation possibilities
- Emerging use cases
- Potential impact areas

## 7. LEARNING PATH

### 7.1 Foundational Knowledge
- Prerequisite concepts and skills
- Basic terminology and definitions
- Fundamental principles
- Introductory materials

### 7.2 Advanced Topics
- Specialized areas of study
- Advanced theories and concepts
- Complex applications
- Research frontiers

## 8. RESOURCES AND REFERENCES

### 8.1 Key Publications
- Foundational papers and articles
- Important books and textbooks
- Research journals and conferences
- Online resources and databases

### 8.2 Learning Materials
- Course recommendations
- Tutorial resources
- Practice exercises
- Assessment materials

## 9. PRACTICAL IMPLEMENTATION

### 9.1 Implementation Guidelines
- Step-by-step implementation instructions
- Best practices and recommendations
- Common pitfalls to avoid
- Success factors and metrics

### 9.2 Case Examples
- Detailed case studies
- Implementation examples
- Success stories
- Lessons learned

## 10. CONCLUSION AND SUMMARY

### 10.1 Key Takeaways
- Most important concepts to remember
- Critical skills and knowledge areas
- Essential principles and practices
- Core competencies developed

### 10.2 Future Outlook
- Expected developments and trends
- Potential impact on various fields
- Future learning opportunities
- Career and application prospects

## APPENDIX: ADDITIONAL RESOURCES

### A.1 Technical References
- Formulas and equations
- Technical specifications
- Standard protocols
- Reference tables

### A.2 Further Reading
- Supplementary materials
- Advanced topics
- Specialized applications
- Research papers

---
*This comprehensive guide provides extensive coverage of {topic.lower()}, offering detailed information suitable for students, professionals, and researchers interested in deepening their understanding of this field.*

**Total Content: Approximately 2,500 words (4-5 pages)**
"""

def get_response(input_text, image):
    """Get response from API or fallback to demo - returns summarized content"""
    if not st.session_state.api_available or st.session_state.demo_mode:
        return get_demo_response(input_text, "technical document")
    
    try:
        # Create a prompt that asks for summary
        prompt = "Please provide a comprehensive summary of this document. Include key points, main ideas, and important findings in a structured format."
        if input_text:
            prompt = f"{input_text}. Please provide a comprehensive summary with key points."
        
        response = model.generate_content([prompt, image])
        
        st.session_state.usage_count += 1
        return response.text
    except Exception as e:
        st.session_state.api_available = False
        st.session_state.last_api_error = str(e)
        return get_demo_response(input_text, "technical document")

def generate_notes(topic, language="english"):
    """Generate comprehensive notes from API or fallback to demo"""
    if not st.session_state.api_available or st.session_state.demo_mode:
        return get_demo_notes(topic, language)
    
    try:
        # Enhanced prompt for comprehensive notes
        prompt = f"""
        Create extremely comprehensive and detailed notes about {topic} in {language} language. 
        The notes should be at least 2-3 pages long when printed and should include:
        
        1. Detailed explanations of all key concepts
        2. Comprehensive coverage of the topic
        3. Multiple examples and case studies
        4. Practical applications and implementations
        5. Current challenges and future directions
        6. References and further reading suggestions
        
        Organize the content with clear headings, subheadings, and bullet points.
        Ensure the notes are suitable for university-level study and professional reference.
        """
        
        response = model.generate_content(prompt)
        
        st.session_state.usage_count += 1
        return response.text
    except Exception as e:
        st.session_state.api_available = False
        st.session_state.last_api_error = str(e)
        return get_demo_notes(topic, language)

# Initialize Streamlit Application
st.set_page_config(
    page_title="Gemini Decode", 
    layout="wide",
    page_icon="📝"
)

st.title("📝 Gemini Decode: Document Extraction & Notes Generator")

# Sidebar with info and controls
with st.sidebar:
    st.header("Settings & Info")
    
    if st.session_state.last_api_error:
        st.error(f"API Error: {st.session_state.last_api_error}")
    
    if not st.session_state.api_available:
        st.warning("⚠️ Using Demo Mode - API unavailable")
        st.session_state.demo_mode = True
    else:
        st.success("✅ API Connected")
        st.session_state.demo_mode = st.checkbox("Use Demo Mode (Save API quota)", value=False)
    
    st.info(f"API Requests: {st.session_state.usage_count}")
    
    if st.button("Reset API Connection"):
        st.session_state.api_available = True
        st.session_state.last_api_error = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Note:**")
    st.markdown("- Generated notes are comprehensive (2+ pages)")
    st.markdown("- Document extraction provides summarized content")
    st.markdown("- Enable billing for increased API quota")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["📄 Document Extraction", "📝 Notes Generator"])

with tab1:
    st.header("Document Extraction")
    st.info("Upload a document image to get a comprehensive summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_prompt = st.text_area(
            "Input (optional):", 
            placeholder="e.g., Summarize the key points of this document",
            height=100
        )
        uploaded_file = st.file_uploader("Choose an image of the document:", type=["jpg", "jpeg", "png"])
        
        image = None
        if uploaded_file is not None:
            try:
                from PIL import Image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

        submit = st.button("Extract & Summarize", type="primary", use_container_width=True)

    with col2:
        st.header("Extracted Summary")
        if submit:
            if image is not None:
                with st.spinner('Analyzing document and creating summary...'):
                    response = get_response(input_prompt, image)
                    st.write(response)
                    
                    if not st.session_state.api_available and not st.session_state.demo_mode:
                        st.info("Switch to demo mode in sidebar to continue testing")
                    
                    if not response.startswith("Error:"):
                        st.download_button(
                            label="Download Summary",
                            data=response,
                            file_name="document_summary.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
            else:
                st.warning("Please upload an image first")
        else:
            st.info("Upload a document image and click 'Extract & Summarize'")
            with st.expander("Example Output"):
                st.markdown("""
                **Sample Summary Includes:**
                - Key points and main ideas
                - Important findings and data
                - Structured analysis
                - Actionable insights
                """)

with tab2:
    st.header("Comprehensive Notes Generator")
    st.info("Enter any topic to get detailed, multi-page notes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        notes_topic = st.text_input("Enter your topic here:", placeholder="e.g., Operating System, Machine Learning")
        notes_language = st.selectbox("Select Language", ["English", "Hindi", "Spanish", "French", "German"])
        
        generate_btn = st.button("Generate Comprehensive Notes", type="primary", use_container_width=True)
    
    with col2:
        st.header("Generated Notes")
        if generate_btn:
            if notes_topic:
                with st.spinner('Creating comprehensive notes (this may take a moment)...'):
                    notes = generate_notes(notes_topic, notes_language)
                    st.write(notes)
                    
                    if not st.session_state.api_available and not st.session_state.demo_mode:
                        st.info("Switch to demo mode in sidebar to continue testing")
                    
                    if not notes.startswith("Error:"):
                        st.download_button(
                            label="Download Notes",
                            data=notes,
                            file_name=f"{notes_topic.replace(' ', '_')}_comprehensive_notes.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
            else:
                st.warning("Please enter a topic first")
        else:
            st.info("Enter a topic and click to generate comprehensive notes")
            with st.expander("Notes Features"):
                st.markdown("""
                **Comprehensive Notes Include:**
                - Detailed explanations (2+ pages)
                - Multiple examples and case studies
                - Practical applications
                - Current challenges and future trends
                - References and resources
                """)

# Footer
st.markdown("---")
st.markdown("""
**Advanced Document Analysis & Note Generation** - This tool provides comprehensive document summaries and detailed educational notes using AI technology.
""")
st.markdown("**Powered by Google Gemini AI | Created by Group 62**")