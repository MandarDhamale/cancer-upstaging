ZERO_SHOT_PROMPT = """You are a clinical NLP assistant. Your task is to determine whether the cancer described in the pathology report below was upstaged. 
Answer only with a JSON object in the following format:
{{"Upstaged": "Yes"}} or {{"Upstaged": "No"}}

Pathology Report:
{report_text}
"""

EVIDENCE_GROUNDED_PROMPT = """You are a clinical NLP assistant analyzing pathology reports. Your task is to determine whether the cancer described in the pathology report was upstaged.
To ensure your reasoning is correct, you must first extract textual evidence (exact quotes) from the report regarding the historical and current TNM stages.
Then, you must deduce whether upstaging occurred (Yes or No).

Your response must be a valid JSON object in exactly this format:
{{
    "Reasoning": "Your reasoning about the stages.",
    "Quote": "Exact substring from the report providing evidence of the stages and whether it was upstaged.",
    "Upstaged": "Yes" or "No"
}}

Ensure that the 'Quote' field contains an EXACT, verbatim copy of the text from the report. Do not modify the text of the quote in any way.

Pathology Report:
{report_text}
"""

def get_messages(prompt_type, report_text):
    if prompt_type == "zero_shot":
        user_message = ZERO_SHOT_PROMPT.format(report_text=report_text)
    elif prompt_type == "evidence_grounded":
        user_message = EVIDENCE_GROUNDED_PROMPT.format(report_text=report_text)
    else:
        raise ValueError("Invalid prompt type.")
        
    return [
        {"role": "system", "content": "You are a helpful and precise medical AI assistant."},
        {"role": "user", "content": user_message}
    ]
