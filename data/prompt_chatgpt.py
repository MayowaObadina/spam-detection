from __future__ import annotations
import os

import pandas as pd

import openai
import asyncio
import logging
import aiolimiter
from tqdm.asyncio import tqdm_asyncio
from aiohttp import ClientSession



from typing import Any
from typing import List



def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)





async def dispatch_openai_requests(
        messages_list: List[List[dict[str, str]]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
) -> List[str]:
    """Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(100):
            try:
                return await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 20 seconds."
                )
                await asyncio.sleep(40)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 20 seconds.")
                await asyncio.sleep(40)
            except openai.error.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}



async def generate_from_openai_chat_completion(
    full_contexts: List[List[dict[str, str]]],
    model_config: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    #if "OPENAI_API_KEY" not in os.environ:
    #    raise ValueError(
    #        "OPENAI_API_KEY environment variable must be set when using OpenAI API."
    #    )
    #openai.api_key = os.environ["OPENAI_API_KEY"]
    session = ClientSession()
    openai.aiosession.set(session)
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model_config,
            messages=full_context,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for full_context in full_contexts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    await session.close()
    return [x["choices"][0]["message"]["content"] for x in responses]



if __name__ == '__main__':

    openai.api_key = "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"
    prompt_prefix = 'Classify this email as "spam" or "ham". '
    data_dir = 'test.csv'

    n_tokens = 0
    #model_name = 'gpt-3.5-turbo'
    model_name = 'gpt-4o-2024-08-06'#'gpt-4o-2024-08-06' #'gpt-3.5-turbo'
    #output_dir = 'gpt_turbo_35_results/'
    output_dir = 'gpt_4_results/'

    create_dir(output_dir)

    df = pd.read_csv(data_dir, sep=',')
    df = df.astype('U')
    df = df.dropna()

    #df = df_.iloc[200:]
    print(df.shape[0])


    all_input_messages = []
    for i in range(df.shape[0]): #df.shape[0]
        content = df['text'].iloc[i]

        message = prompt_prefix + content
        input_mes = [{"role": "system", "content": "You are an assistant able to classify email as spam or ham"},
            {"role": "user", "content": message[:500]}, ]
        all_input_messages.append(input_mes)

    responses = asyncio.run(generate_from_openai_chat_completion(all_input_messages, model_name, 0.3, 500, 1.0, 500))


    print(responses[:5])
    completions = []
    for completion_text in responses:
        #print(completion_text, n_tokens)

        completions.append(completion_text)

    df['gpt'] = completions

    df.to_csv(output_dir+"GPT-4o"+'.csv', sep=',')



'''
      completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [{"role": "system", "content" : "You are a helpful assistant."},
                    {"role": "user", "content" : message[:500]},]
                    )
      #print(completion)
      print(completion)
      '''