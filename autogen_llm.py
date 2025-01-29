from types import SimpleNamespace
import requests
import autogen
from autogen import AssistantAgent, UserProxyAgent
import params

with open(params.anl_llm_url_path, 'r') as url_f:
    ANL_URL = url_f.read().strip()

class ArgoModelClient:
    def __init__(self, config, **kwargs):
        print(f"Argo config: {config}")
        self.model = config['model'] 
        self.temp = config['temp']

    def create(self, params):
        if params.get("stream", False) and "messages" in params:
            raise NotImplementedError("Local models do not support streaming.")

        response = SimpleNamespace()

        prompt = apply_chat_template(params['messages'])

        response.choices = []
        response.model = "model_name"

        num_of_responses = params.get("n", 1)


        for _ in range(num_of_responses):
            text = self._query_argo(prompt)
            choice = SimpleNamespace()
            choice.message = SimpleNamespace()
            choice.message.content = text
            choice.message.function_call = None
            response.choices.append(choice)
        return response

    def _query_argo(self, prompt): 
        req_obj = {'user': 'aps', 
                   'model': self.model, 
                   'prompt': [prompt], 
                   'system': "",
                   'stop': [],
                   'temperature': self.temp}
        result = requests.post(ANL_URL, json=req_obj)
        if not result.ok:
            raise ValueError(f"error {result.status_code} ({result.reason})")

        response = result.json()['response']
        return response

    def message_retrieval(self, response):
        choices = response.choices
        return [choice.message.content for choice in choices]

    def cost(self, response) -> float:
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response):
        return {}


def apply_chat_template(messages):
    output_str = ""
    for message in messages:
        output_str += f"-- {message['role']}"
        if 'name' in message:
            output_str += f"{message['name']} --\n"
        else:
            output_str += "--\n"
        output_str += f"{message['content']}\n --- \n\n"
    return output_str