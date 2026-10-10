from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from rag_tool import it_kb_retrieve

MODEL_NAME = r"D:\se_course\gex3\models\qwen3-0.6b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import Optional, List, Any

class LocalQwenLLM(LLM):
    model_pipe: Any
    @property
    def _llm_type(self) -> str:
        return "local_qwen"
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        out = self.model_pipe(
            prompt,
            max_new_tokens=180,
            temperature=0.0,
            do_sample=False
        )
        generated = out[0]["generated_text"][len(prompt):]
        if "Question:" in generated:
            generated = generated.split("Question:")[0]
        return generated.strip()


llm_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=180,
    temperature=0.0,
    do_sample=False
)
llm = LocalQwenLLM(model_pipe=llm_pipe)

prompt_text = """Answer the user's university IT‑support question.
You have access to one tool: it_kb_retrieve.
IMPORTANT RULES:
1. ALWAYS use it_kb_retrieve to look up information before answering.
2. YOU MUST NEVER invent Observation content. Observation is ONLY the output returned by the tool. You cannot write fake Observation values.
3. You MUST NOT invent any facts, phone‑numbers or policies.
4. If the retrieved tool context contains NO answer for the question, reply EXACTLY: "This information is not specified in the available IT policies."
5. Keep answer concise. Do NOT add extra "Answer:" repeated lines.

Tools:
{tools}
Use the following STRICT format, do NOT deviate:
Question: the input question
Thought: you should always think about what to do
Action: the tool name, MUST be [{tool_names}]
Action Input: input for tool
Observation: tool output
Thought: I now know the final answer
Final Answer: your final answer
Begin!
Question: {input}
Thought:{agent_scratchpad}
"""


prompt = PromptTemplate.from_template(prompt_text)
tools = [it_kb_retrieve]
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
if __name__ == "__main__":
    test_question = "What exact phone number should I call for the University IT Service Desk?"
    result = agent_executor.invoke({"input": test_question})
    print("\n==== Final Answer ====")
    print(result["output"])
