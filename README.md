# Minimizing-Manipulations-in-LLMs

This project is a smart safety system for AI chatbots. Its main job is to act like a "guardian," stopping the AI from writing harmful, toxic, or inappropriate responses.
> ### ⚠️ Content Warning
> This repository contains files with examples of harmful and offensive text (specifically in `manipulativePrompts.csv` ). This data is used exclusively for the important work of testing and developing safety measures for AI. Please be aware of this before exploring the files.

---

## 📋 Table of Contents
1.  [The Problem](#-the-problem)
2.  [Our Solution](#-our-solution)
3.  [How It Works](#-how-it-works)
4.  [Key Parts of the System](#-key-parts-of-the-system)
5.  [Benchmark: Why This is Needed](#-benchmark-why-this-is-needed)
6.  [Getting Started: Setup Guide](#-getting-started-setup-guide)
7.  [How to Use the App](#-how-to-use-the-app)
8.  [Project Files Explained](#-project-files-explained)

## 🎯 The Problem

As AI chatbots become more common, we need to ensure they are safe. A big challenge is that people can sometimes trick these AIs into saying bad things—a practice called "jailbreaking." Our research and testing show that even the most advanced AIs can be vulnerable to these tricks, highlighting the need for an extra layer of security.

## ✅ Our Solution

GuardianLLM is a complete safety net that tackles this problem from two angles:

1.  **Proactive Scanning:** It first checks the user's question with a custom-built "Content Scanner" to see if it's asking for something dangerous *before* the AI even replies.
2.  **Reactive Guardrails:** After the AI writes an answer, a second "Safety Checker" AI double-checks it. If it finds anything unsafe, it tells the first AI to scrap the answer and write a new, safer one.

This entire system is wrapped in a simple web application where you can chat with the protected AI and see the safety features in action.

## ⚙️ How It Works

The system is a clever, two-step process for generating safe AI responses. It uses a main AI to write answers and a second AI to validate them.

Here is a diagram of the workflow:

```mermaid
graph TD
    A[User's Question] --> B{Main AI Writes Answer <br> (Llama 3.2)};
    B --> C{Safety Checker AI Reviews It <br> (Microsoft Phi-3.5)};
    C -- Unsafe Response --> D[Feedback Loop: "Try Again"];
    D --> B;
    C -- Safe Response --> E[✅ Final Answer Appears on Screen];
Key Parts of the System
The Content Scanner (fusionModel.py)
This is a powerful scanner trained to spot over 140 different types of unsafe text¹.

Technology: It combines the strengths of two advanced AI models (DeBERTa-v3 and RoBERTa) to be highly accurate².

Training: It learned how to spot harmful content by studying the NVIDIA Aegis AI Content Safety Dataset³.

The AI Safety Team (multiAgent.py)
This is the core of the system, where two AIs work together as a team.

The Generator AI (Llama 3.2): A creative AI that's good at writing answers to your questions⁴.

The Validator AI (Microsoft Phi-3.5): A fast, efficient AI that's excellent at spotting safety issues and giving a simple "safe" or "unsafe" verdict⁵.

The Website Dashboard (main.py)
A simple and interactive web page built with Streamlit⁶.

Chat Window: Lets you talk to the AI that is protected by the GuardianLLM system⁷.

Analysis Panels: Shows you in real-time what the Content Scanner found in your question and how many times the AI had to rewrite its own answer to make it safe⁸.

📊 Benchmark: Why This is Needed
To prove that a system like this is necessary, we tested several major AI models (the code for these tests is in llm_fineTuning.py⁹). The results clearly show that even the biggest and most expensive models can be tricked.

High Jailbreak Rates: We found that top-tier models could be jailbroken over 90% of the time¹⁰.

Toxicity and Stereotypes: Many models scored poorly on generating toxic language and relying on harmful stereotypes¹¹.

These results prove that AI models need a strong guardian system like this one to ensure they remain safe and helpful.

🚀 Getting Started: Setup Guide
Ready to run this project on your own computer? Follow these steps.

1. You Will Need:

Python 3.8+

Git for version control

2. Copy the Project (Clone)
Open your computer's terminal and run this command:

Bash

git clone <your-repository-url>
cd <repository-name>
3. Create a Clean Workspace (Virtual Environment)
This creates a separate space for the project so it doesn't interfere with other Python programs on your computer.

Bash

# For Mac/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
4. Install All Required Packages
This command installs all the tools the project needs from the requirements.txt file¹².

Bash

pip install -r requirements.txt
5. Add Your API Key
The system needs a free API key from Hugging Face to work. Open the multiAgent.py file¹³. It's best practice to load your key from an environment variable. However, for a quick start, you can find the line client = InferenceClient(api_key="hf_...") and replace the hf_... part with your own key¹⁴.

⚡ How to Use the App
1. Prepare the Test Questions
Run this command to create the manipulativePrompts.csv file¹⁵, which the app uses on the "Prompt Analysis" page.

Bash

python prompts.py
2. Launch the Website
Run this command to start the Streamlit web application¹⁶.

Bash

streamlit run main.py
Your web browser should automatically open to a local website where you can start chatting!

📁 Project Files Explained
.
├── fusionModel.py           # Code to train the AI that scans for unsafe content.
├── guardLLM.py              # Code that uses the trained scanner on new text.
├── llm_fineTuning.py        # Code for testing and fine-tuning various AIs.
├── main.py                  # The code for the main website dashboard.
├── multiAgent.py            # Code for the two AIs (Generator & Validator) that work together.
├── prompts.py               # Script that creates the list of tricky test questions.
├── labels.py                # A list of all 146 unsafe categories the scanner looks for.
├── manipulativePrompts.csv  # A file with tricky questions used for testing.
├── requirements.txt         # A list of all the tools this project needs to run.
└── README.md                # This file!
