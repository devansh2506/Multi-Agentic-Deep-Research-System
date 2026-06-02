import os
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from reporter.prompts import report_writer_prompt



# Initialize the model (using Gemini 1.5 Flash for high-capacity synthesis and Markdown formatting)
model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai", temperature=0.3)

# The reporter is a simple LCEL chain: it takes the inputs, formats the prompt, calls the model, and parses to a string.
report_writer_agent = (report_writer_prompt | model | StrOutputParser()).with_config({"run_name": "Reporter_Agent"})
