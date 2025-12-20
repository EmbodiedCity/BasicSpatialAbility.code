import os
import random
import httpx
import argparse
from openai import OpenAI
import base64
import csv
import re

from tenacity import ( # A retrying library that automatically re-executes code upon exceptions
    retry, # A decorator used to retry a function, invoked with @
    stop_after_attempt, # retry stops after a specified number of attempts
    wait_random_exponential, # wait time between retries, determined by random exponential backoff
)
from config import PROXY, ATTEMPT_COUNTER, WAIT_TIME_MIN, WAIT_TIME_MAX, VLLM_URL # Configuring variables are defined in a seperate config.py
'''from utils import token_count'''

FREE_APIS=['llama3-8b', 'llama3.1-8b', 'gemma2-9b', 'mistral7bv2', 'qwen2-1.5b', 'qwen2-7b', 'glm4-9b', 'glm3-6b']

os.environ["SiliconFlow_API_KEY"] = "" # API Key
os.environ["DeepInfra_API_KEY"] = ""
os.environ["Google_API_KEY"] = ""
os.environ["OpenAI_API_KEY"] = ""

def get_api_key(platform, model_name=None):
    if platform=="OpenAI":
        return os.environ["OpenAI_API_KEY"]
    elif platform=="DeepInfra":
        return os.environ["DeepInfra_API_KEY"]
    elif platform=="vllm":
        return os.environ["vllm_KEY"]
    elif platform=="Google":
        return os.environ["Google_API_KEY"]
    elif platform=="SiliconFlow":
        if model_name in FREE_APIS:
            # Use multiple free keys
            keys = [os.environ["SiliconFlow_API_KEY"]]
            try:
                k2 = [os.environ["SiliconFlow_API_KEY_yuwei"]]
                k3 = [os.environ["SiliconFlow_API_KEY_tianhui"]]
                keys = keys+k2+k3
            except:
                pass
            return random.choice(keys)
        else:
            # Use a single paid keys
            return os.environ["SiliconFlow_API_KEY"]


class LLMAPI:
    def __init__(self, model_name, platform=None):
        self.model_name = model_name
        
        # Recommend SiliconFlow, then DeepInfra, then other platforms
        self.platform_list = ["SiliconFlow", "OpenAI", "DeepInfra", 'vllm',"Google"]
        self.model_platforms = {
                    "SiliconFlow":  [
                        'llama3-8b', 'llama3-70b', 'gemma2-9b', 'gemma2-27b', 'mistral7bv2', 'qwen2-1.5b', 'qwen2-7b', 'qwen2-14b', 'qwen2-72b', 'glm4-9b', 'glm3-6b', 'deepseekv2', 'llama3.1-8b', 'llama3.1-70b', 'llama3.1-405b'] + [
                        'llama3-8b-pro', 'gemma2-9b-pro', 'mistral7bv2-pro', 'qwen2-1.5b-pro', 'qwen2-7b-pro', 'glm4-9b-pro', 'glm3-6b-pro','qwen2-vl-72b','qwen2-vl-7b','intern-vl2-76b','intern-vl2-26b','intern-vl2-8b'
                    ],
                    "OpenAI":       ['gpt35turbo', 'gpt4turbo', 'gpt4o', 'gpt4omini'],
                    "DeepInfra":    ['llama3-8b', 'llama3-70b', 'gemma2-9b', 'gemma2-27b', 'mistral7bv2', 'qwen2-7b', 'qwen2-72b', 'llama3.1-8b', 'llama3.1-70b', 'mistral7bv3', 'llama3.1-405b','llama3.2-11bv','llama3.2-90bv'], # 徐文睿加了'llama3.2-11bv','llama3.2-90bv'
                    "vllm": ['llama3-8B-local', 'gemma2-2b-local', 'chatglm3-citygpt', 'chatglm3-6B-local'],
                    "Google": ['gemini-1.5-flash', 'gemini-1.5-flash-8b', 'gemini-1.5-pro']
                }

        self.model_mapper = {
            'gpt35turbo': 'gpt-3.5-turbo',
            'gpt4turbo': 'gpt-4-turbo-2024-04-09',
            'gpt4o': 'gpt-4o-2024-08-06',
            'gpt4omini': 'gpt-4o-mini-2024-07-18',
            'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'llama3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'llama3-8b-pro': 'Pro/meta-llama/Meta-Llama-3-8B-Instruct',
            'llama3-70b': 'meta-llama/Meta-Llama-3-70B-Instruct',
            'llama3.1-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
            'llama3.1-405b': 'meta-llama/Meta-Llama-3.1-405B-Instruct',
            'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
            'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
            'llama2-70b': 'meta-llama/Llama-2-70b-chat-hf',
            'llama3.2-11bv': 'meta-llama/Llama-3.2-11B-Vision-Instruct',
            'llama3.2-90bv': 'meta-llama/Llama-3.2-90B-Vision-Instruct',
            'gemma2-9b': 'google/gemma-2-9b-it',
            'gemma2-9b-pro': 'Pro/google/gemma-2-9b-it',
            'gemma2-27b': 'google/gemma-2-27b-it',
            'mistral7bv2': 'mistralai/Mistral-7B-Instruct-v0.2',
            'mistral7bv3': 'mistralai/Mistral-7B-Instruct-v0.3',
            'mistral7bv2-pro': 'Pro/mistralai/Mistral-7B-Instruct-v0.2',
            'qwen2-1.5b': 'Qwen/Qwen2-1.5B-Instruct',
            'qwen2-1.5b-pro': 'Pro/Qwen/Qwen2-1.5B-Instruct',
            'qwen2-7b': 'Qwen/Qwen2-7B-Instruct',
            'qwen2-7b-pro': "Pro/Qwen/Qwen2-7B-Instruct",
            'qwen2-14b': 'Qwen/Qwen2-57B-A14B-Instruct',
            'qwen2-72b': 'Qwen/Qwen2-72B-Instruct',
            'glm4-9b': 'THUDM/glm-4-9b-chat',
            'glm4-9b-pro': 'Pro/THUDM/glm-4-9b-chat',
            'glm3-6b': 'THUDM/chatglm3-6b',
            'glm3-6b-pro': 'Pro/THUDM/chatglm3-6b',
            'deepseekv2': 'deepseek-ai/DeepSeek-V2-Chat',
            'llama3-8B-local':'llama3-8B-local',
            'gemma2-2b-local': 'gemma2-2b-local',
            'chatglm3-citygpt': 'chatglm3-citygpt',
            'chatglm3-6B-local': 'chatglm3-6B-local',
            'qwen2-vl-72b':'Qwen/Qwen2-VL-72B-Instruct',
            'qwen2-vl-7b':'Pro/Qwen/Qwen2-VL-7B-Instruct',
            'gemini-1.5-flash':'gemini-1.5-flash',
            'gemini-1.5-flash-8b':'gemini-1.5-flash-8b',
            'gemini-1.5-pro':'gemini-1.5-pro',
            'intern-vl2-76b':'OpenGVLab/InternVL2-Llama3-76B',
            'intern-vl2-26b':'OpenGVLab/InternVL2-26B',
            'intern-vl2-8b':'Pro/OpenGVLab/InternVL2-8B'

        }

        # Check if the model exists
        support_models = ";".join([";".join(self.model_platforms[k]) for k in self.model_platforms])
        if self.model_name not in support_models:
            raise ValueError('Invalid model name! Please use one of the following: {}'.format(support_models))
        
        # Use specified platform or decide automatically
        if platform is not None and platform in self.platform_list:
            self.platform = platform
        else:
            for platform in self.platform_list:
                if self.model_name in self.model_platforms[platform]:
                    self.platform = platform
                    break
        # Invalid Platform
        if self.platform is None:
            raise ValueError("'Invalid API platform:{} with model:{}".format(self.platform, self.model_name))
        # The model and platform do not match
        if self.model_name not in self.model_platforms[self.platform]:
            raise ValueError('Invalid model name! Please use one of the following: {} in API platform:{}'.format(support_models, self.platform))
        
        # Generate url
        if self.platform == "OpenAI":
            self.client = OpenAI(
                api_key=get_api_key(platform),
                base_url = "https://api3.apifans.com/v1"
                #http_client=httpx.Client(proxies=PROXY),
            )
        elif self.platform == "DeepInfra":
            self.client = OpenAI(
                base_url="https://api.deepinfra.com/v1/openai",
                api_key=get_api_key(platform),
                #http_client=httpx.Client(proxies=PROXY),
            )
        elif self.platform == "SiliconFlow":
            self.client = OpenAI(
                base_url="https://api.siliconflow.cn/v1",
                api_key=get_api_key(platform, model_name)
            )
        elif self.platform == 'vllm':
            self.client = OpenAI(
                base_url=VLLM_URL,
                api_key=get_api_key(platform)
            )
        elif self.platform == 'Google':
            self.client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/",
                api_key=get_api_key(platform)
            )



    def get_client(self):
        return self.client
    
    def get_model_name(self):
        return self.model_mapper[self.model_name]
    
    def get_platform_name(self):
        return self.platform

    def get_supported_models(self):
        return self.model_platforms


class LLMWrapper:
    def __init__(self, model_name, platform=None):
        self.model_name = model_name
        self.hyperparams = {
            'temperature': 0.,  # make the LLM basically deterministic
            'max_new_tokens': 100, # not used in OpenAI API
            'max_tokens': 1000,    # The maximum number of [tokens](/tokenizer) that can be generated in the completion.
            'max_input_tokens': 2000 # The maximum number of input tokens
        }
        
        self.llm_api = LLMAPI(self.model_name, platform=platform)
        self.client = self.llm_api.get_client()
        self.api_model_name = self.llm_api.get_model_name()

    @retry(wait=wait_random_exponential(min=WAIT_TIME_MIN, max=WAIT_TIME_MAX), stop=stop_after_attempt(ATTEMPT_COUNTER))
    def get_response(self, prompt_text):

        response = self.client.chat.completions.create(
            model=self.api_model_name,
            messages=[
            {
                "role": "user",
                "content":[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{prompt_image}",
                            "detail":"low"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"{prompt_text}"
                    }
                ]
            }],
                        
            #messages=system_messages + [{"role": "user", "content": prompt_text}],
            max_tokens=self.hyperparams["max_tokens"],
            temperature=self.hyperparams["temperature"]
        )
        full_text = response.choices[0].message.content

        return full_text


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def save_response_to_csv(i_number, response_text, csv_file_path='predictions.csv'):
    # Parsing the response into Question Number and Answer
    try:
        if args.platform == 'DeepInfra':
            '''print('This model is stupid！')
            question_number = i_number

            response_text = response_text[:-1]
            words = response_text.split()
            print(words)

            answer_list = []
            for word in words:
                if word.isalpha() and len(word) == 1 and 'A' <= word <= 'Z':
                    answer_list.append(word)
            print(answer_list)
            answer = ''.join(answer_list)'''
            question_number, answer = response_text.strip().split(',')
        else:
            question_number, answer = response_text.strip().split(',')
    except:
        question_number, answer = (i_number, 'Fail!!!')

    # Write results
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        file.seek(0, 2)
        if file.tell() == 0:
            writer.writerow(["Question Number", "Answer"])  # Write headers

        writer.writerow([question_number, answer])


def compare_answers(answers_file, predictions_file, output_file):
    # Step 1: Read answer file
    correct_answers = {}
    with open(answers_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            question_number = row['Question Number']
            correct_answers[question_number] = row['Answer']

    # Step 2: Read prediction results
    predictions = {}
    with open(predictions_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            question_number = row['Question Number']
            predictions[question_number] = row['Answer']

    # Step 3: Compare answers
    comparison_results = []
    correct_count = 0

    for question_number, correct_answer in correct_answers.items():
        predicted_answer = predictions.get(question_number, "")
        is_correct = correct_answer == predicted_answer
        comparison_results.append({
            'Question Number': question_number,
            'Correct Answer': correct_answer,
            'Predicted Answer': predicted_answer,
            'Result': 'True' if is_correct else 'False'
        })
        if is_correct:
            correct_count += 1

    # Step 4: Write scores
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['Question Number', 'Correct Answer', 'Predicted Answer', 'Result']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_results)

    # Step 5: Print scores
    print(f"Total correct answers: {correct_count} out of {len(correct_answers)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='qwen2-vl-7b') # Set default model
    parser.add_argument("--platform", type=str, default="SiliconFlow", choices=["SiliconFlow", "OpenAI", "DeepInfra","Google"]) # Set default platform, proxy may needed
    args = parser.parse_args()
    llm = LLMWrapper(model_name=args.model_name, platform=args.platform)

    # instruction
    prompt_image = encode_image_to_base64(rf'') # File location
    prompt_text = rf"You are taking a spatial ability test. This test consists of 40 patterns which can be folded into figures. To the right of each pattern there are five figures. You are to decide which of these figures can be made from the pattern shown. The pattern always shows the outside of the figure. Here is an example: Which of these five figures—A, B, C, D, E—can be made from the pattern at the left?"
    llm_response = llm.get_response(prompt_text)
    print('=========INSTRUCTION=========')
    print(llm_response)

    prompt_image = encode_image_to_base64(rf'') # File location
    prompt_text = "In Example X, A and B certainly cannot be made; they are not the right shape. C and D is correct both in shape and size. You cannot make E from this pattern. So the right answers are CD. Remember: 1. In this test there will always be a row of five figures following each pattern. 2. In every row there is at least one correct figure. 3. Usually more than one is correct. In facts, in some cases, all five may be correct. Now look at Example Y and the five choices for it. Note that when the pattern is folded, the figure must have two gray surfaces. One of these is a large surface which could be either the top or bottom of a box. The other is a small surface which would be one end of the box."
    llm_response = llm.get_response(prompt_text)
    print('=========INSTRUCTION=========')
    print(llm_response)



    # tests
    for i in range(1,41):
        prompt_image = encode_image_to_base64(rf'') # File location
        if args.model_name == 'DeepInfra':
            prompt_text = rf"This is Question {i}. Each question has at least one correct answer, but may have one or more than one answer. Please output the result in pure text format as: question number,answers. For example: '1,BDE'. Please answer."
        else:
            prompt_text = rf"This is Question {i}. Each question has at least one correct answer, but may have one or more than one answer. Please output the result in pure text format as: question number,answers. For example: '1,BDE'. Please answer."
    
        llm_response = llm.get_response(prompt_text)
        print(f'========= No. {i} =========')
        print(llm_response)
        save_response_to_csv(i,llm_response,'predictions.csv') # Destination directory
    print('Test completed!')

    # Compare answers
    answers_file = '' # Destination directory
    predictions_file = 'predictions.csv'
    output_file = 'comparison_results.csv'
    compare_answers(answers_file, predictions_file, output_file)

