from typing import List, Tuple

# !pip install pyllamacpp

def llamacpp_generate_stream(model, tokenizer, params, device,
                    context_len=2048, stream_interval=2):
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)

    def new_text_callback(text: str):
        yield text

    model.generate(prompt, 
               n_predict=max_new_tokens, 
               new_text_callback=new_text_callback, 
               n_threads=8)
    
    return new_text_callback