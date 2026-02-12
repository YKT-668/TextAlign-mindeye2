import torch
from diffusers import StableDiffusionXLPipeline

print('loading pipe...')
pipe = StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0')
print('loaded')
res = pipe.encode_prompt(prompt='a photo of a cat', device='cpu', num_images_per_prompt=1, do_classifier_free_guidance=False)
print('encode_prompt returned type:', type(res))
try:
    print('len:', len(res))
    for i, r in enumerate(res):
        print(i, type(r))
        if hasattr(r, 'shape'):
            print(' shape', getattr(r, 'shape'))
        if hasattr(r, 'last_hidden_state'):
            print(' has last_hidden_state', type(r.last_hidden_state), getattr(r.last_hidden_state, 'shape', None))
except Exception as e:
    print('inspect error:', e)

# Also call internal text encoders directly
print('\nCall text_encoder outputs:')
text = 'a photo of a cat'
enc1 = pipe.tokenizer(text, return_tensors='pt')
try:
    out1 = pipe.text_encoder(enc1.input_ids)
    print('text_encoder returns', type(out1))
    if hasattr(out1, 'last_hidden_state'):
        print(' last_hidden_state shape', out1.last_hidden_state.shape)
    else:
        try:
            print(' out1[0] shape', out1[0].shape)
        except Exception as e:
            print(' cannot inspect out1 contents:', e)
except Exception as e:
    print('text_encoder call failed:', e)

print('done')
