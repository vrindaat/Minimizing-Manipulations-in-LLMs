import os
import json
from huggingface_hub import InferenceClient
import yaml
import dirtyjson
client = InferenceClient(api_key="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")


class crewAI():
    def __init__(self,num_trigger,typeValidation,default=""):
        self.num_trigger=num_trigger
        self.typeValidation=typeValidation
        if default=="":
            self.default="I can't assist with that request. Is there anything else I can help you out with ?"
        else:
            self.default=default


    def response(self,userPrompt):
        count=0
        feedback=None
        original=""
        flag=-1
        while self.num_trigger>count:
            count+=1
            agent1_role="Response Generator",
            agent1_goal=f"Your task is to generate response to the following user query in 50 words or less."
            if feedback is None:
                messages = [
                    {
                        "role": "user",
                        "content": f"User Question : {userPrompt}"
                    }
                ]
            else:

                messages = [
                    {
                        "role": "user",
                        "content": f"User Question : {userPrompt} <END> Your previous response to the same user question had some issues. Here is the feedback {feedback}. Please re-generate a appropriate response."
                    }
                ]


            response= client.chat.completions.create(
                model="meta-llama/Llama-3.2-1B-Instruct",
                messages=messages,
                max_tokens=300,
                temperature=0
            )

            if count==1:
                original=response.choices[0].message['content']

            agent2_role = "Response Validator"

            agent2_goal=f"Your task is to ensure that the generated response is safe and sound and free from any toxicity and sexually explicit phrases. Completely flag any 18+ responses and any unethical answers"

            agent2_expectation="You are expected to generate a json with the following keys isSafe and reason. isSafe should be boolean and reason should be a one line justification as to why you flagged it as safe or toxic! Strictly JSON output. Format the JSON properly."
            messages = [
                {
                    "role": "user",
                    "content": f"ROLE : {agent2_role}, GOAL : {agent2_goal}, Expectations : {agent2_expectation}, LLM RESPONSE : {response.choices[0].message['content']} If the LLM Response is harmful, then return the json as mentioned. Your job is to ensure the safety and to be a responsible AI"
                }
            ]

            response_ = client.chat.completions.create(
                model="microsoft/Phi-3.5-mini-instruct",
                messages=messages,
                max_tokens=200,
                temperature=0
            )

            try:
                value=str(response_.choices[0].message['content'])
                value=yaml.safe_load(value[value.index("{"):value.index("}")+1])
                if value["isSafe"]:
                    flag=1
                    break
                else:

                    feedback=value["reason"]
                    print(feedback)
            except:
                raise NotImplementedError

        print(count)
        if flag==-1:
            return self.default,count,original
        return response.choices[0].message['content'],count,original




