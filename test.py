import openai
import json

from utils.fileio import read_file_str, write_file_str, read_file_json
from argp import get_args
from test_funcs.base import get_current_detection, get_current_task


def get_functions(func_list):
    """Get the list of functions that GPT can call"""
    return read_file_json(func_list, strip=False)


def run_conversation(args):
    # Step 1: send the conversation and available functions to GPT
    base_prompt = read_file_str(args.base_prompt, strip=True)
    functions = get_functions(args.func_list)

    messages = [{"role": "user", "content": base_prompt}]
    response = openai.ChatCompletion.create(
        model=args.model,
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]  # ignore other info: includes log, usage, etc.

    ## Sample response
    # {
    #     "role": "assistant",
    #     "content": null,
    #     "function_call": {
    #         "name": "get_current_detection",
    #         "arguments": "{\n  \"camera_id\": \"right_hand_camera\"\n}"
    #     }
    # }

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_detection": get_current_detection,
            # "get_current_task": get_current_task,
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = fuction_to_call(
            camera_id=function_args.get("camera_id"),
        )

        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            functions=functions,
            function_call="auto",
        )  # get a new response from GPT where it can see the function response

        # TODO third response
        second_response_message = second_response["choices"][0]["message"]
        if second_response_message.get("function_call"):
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                # "get_current_detection": get_current_detection,
                "get_current_task": get_current_task,
            }  # only one function in this example, but you can have multiple
            function_name = second_response_message["function_call"]["name"]
            fuction_to_call = available_functions[function_name]
            function_args = json.loads(second_response_message["function_call"]["arguments"])
            function_response = fuction_to_call(
                category=function_args.get("category"),
            )

            # Step 4: send the info on the function call and function response to GPT
            messages.append(second_response_message)  # extend conversation with assistant's reply
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
            third_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
            )  # get a new response from GPT where it can see the function response

            print(third_response)


if __name__ == "__main__":
    args = get_args()
    openai.api_key = read_file_str(args.api_key, strip=True)
    run_conversation(args)
