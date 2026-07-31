try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
except ImportError:
    GPT2LMHeadModel = None
    GPT2Tokenizer = None
    torch = None

prompt = "Artificial Intelligence is"

if GPT2Tokenizer is not None and GPT2LMHeadModel is not None and torch is not None:
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_length=80,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as exc:
        generated_text = f"Fallback output: {prompt} a rapidly growing field that changes how we live and work. (Reason: {exc})"
else:
    generated_text = f"Fallback output: {prompt} a rapidly growing field that changes how we live and work."

print(generated_text)