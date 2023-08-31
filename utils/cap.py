import openai

def lmp(base_prompt, query, model_name="gpt-3.5-turbo-0613", stop_tokens=None, query_kwargs=None):
    new_prompt = f'{base_prompt}\n{query}'

    use_query_kwargs = {
        'engine': model_name,
        'max_tokens': 512,
        'temperature': 0,
    }
    if query_kwargs is not None:
      use_query_kwargs.update(query_kwargs)

    response = openai.Completion.create(
        prompt=new_prompt, stop=stop_tokens, **use_query_kwargs
    )['choices'][0]['text'].strip()

    print(query)
    print(response)

    return response